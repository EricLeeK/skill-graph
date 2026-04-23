"""Evaluate SkillGraph on SkillsBench curated tasks."""
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from skill_graph import SkillGraph


def load_skillsbench_tasks(curated_dir: str) -> List[Dict]:
    """Load SkillsBench curated tasks."""
    curated = Path(curated_dir)
    with open(curated / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = []
    for task_name, info in manifest.items():
        if not info.get("has_skills"):
            continue
        task_dir = curated / task_name
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.exists():
            continue

        instruction = instruction_file.read_text(encoding="utf-8").strip()
        # Use first paragraph as query
        first_para = instruction.split("\n\n")[0].strip()
        if len(first_para) > 500:
            first_para = first_para[:500]

        tasks.append({
            "task_name": task_name,
            "query": first_para,
            "full_instruction": instruction,
            "ground_truth_skills": info["skill_names"],
        })

    return tasks


def evaluate_system(
    sg: SkillGraph,
    tasks: List[Dict],
    query_embeddings: np.ndarray,
    use_sacr: bool = True,
    use_sre: bool = True,
    top_k: int = 10,
) -> Dict:
    """Run evaluation for one configuration."""
    hits = 0
    hit_details = []
    latencies = []

    for task, q_emb in zip(tqdm(tasks, desc=f"sacr={use_sacr}"), query_embeddings):
        gt_skills = set(s.lower() for s in task["ground_truth_skills"])

        t0 = time.perf_counter()
        result = sg.retrieve_from_embedding(
            q_emb, query=task["query"], top_k=top_k, use_sacr=use_sacr, use_sre=use_sre
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_names = set(s.name.lower() for s in result.skills)
        if gt_skills & retrieved_names:
            hits += 1
            hit_details.append({
                "task": task["task_name"],
                "gt": list(gt_skills),
                "retrieved": list(retrieved_names)[:top_k],
                "match": list(gt_skills & retrieved_names),
            })

    recall = hits / len(tasks) if tasks else 0
    return {
        "recall_at_10": round(recall, 4),
        "task_success_rate": round(recall, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "n_tasks": len(tasks),
        "n_hits": hits,
        "hit_details": hit_details,
    }


def main():
    random.seed(42)
    np.random.seed(42)

    curated_dir = "/tmp/skill-usage-hf/skillsbench/curated"
    tasks = load_skillsbench_tasks(curated_dir)
    print(f"Loaded {len(tasks)} SkillsBench tasks")

    print("Initializing SkillGraph with 34K UCSB skills...")
    sg = SkillGraph(
        skills_path="data_real/skills_ucsb_34k.jsonl",
        hsc_path="data_real/hsc_ucsb_34k",
    )

    print("Batch encoding queries...")
    query_texts = [t["query"] for t in tasks]
    query_embeddings = sg.embedder.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)

    print("\n=== Experiment 1: Full System (SACR + SRE) ===")
    result_full = evaluate_system(sg, tasks, query_embeddings, use_sacr=True, use_sre=True)
    print(json.dumps({k: v for k, v in result_full.items() if k != "hit_details"}, indent=2))

    print("\n=== Experiment 2: Without SACR (Dense + SRE) ===")
    result_no_sacr = evaluate_system(sg, tasks, query_embeddings, use_sacr=False, use_sre=True)
    print(json.dumps({k: v for k, v in result_no_sacr.items() if k != "hit_details"}, indent=2))

    print("\n=== Experiment 3: Dense Retrieval Baseline ===")
    result_dense = evaluate_system(sg, tasks, query_embeddings, use_sacr=False, use_sre=False)
    print(json.dumps({k: v for k, v in result_dense.items() if k != "hit_details"}, indent=2))

    results = {
        "full_system": {k: v for k, v in result_full.items() if k != "hit_details"},
        "without_sacr": {k: v for k, v in result_no_sacr.items() if k != "hit_details"},
        "dense_baseline": {k: v for k, v in result_dense.items() if k != "hit_details"},
    }

    out = Path("data_real/results_skillsbench_ucsb_34k.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
