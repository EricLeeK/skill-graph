"""Compare Manual vs Auto keyword maps on SkillsBench with cross-validation.

This script directly compares:
1. Manual keyword map (135/189 mappings leaked from test set GT)
2. Auto keyword map (zero test-set leakage, generated from skill library)

Both are evaluated with 5-Fold CV for fair comparison.
"""
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from skill_graph import SkillGraph
from skill_graph.core.sacr import SparseAttentionCommunityRouter


CONFIGS: List[Tuple[int, int, int]] = [
    (8, 20, 50),
    (12, 30, 80),
    (16, 40, 100),
    (20, 50, 120),
    (24, 60, 150),
    (8, 40, 100),
    (12, 50, 120),
    (16, 60, 150),
    (20, 80, 200),
    (8, 60, 150),
    (12, 80, 200),
    (16, 100, 250),
]

N_FOLDS = 5
RANDOM_SEED = 42


def extract_query(instruction: str, max_len: int = 500) -> str:
    paragraphs = instruction.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if para.startswith("#"):
            continue
        if len(para) < 20:
            continue
        if len(para) > max_len:
            para = para[:max_len]
        return para
    for line in instruction.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and len(line) >= 10:
            return line[:max_len]
    return instruction[:max_len]


def load_skillsbench_tasks(curated_dir: str) -> List[Dict]:
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
        query = extract_query(instruction)
        tasks.append({
            "task_name": task_name,
            "query": query,
            "ground_truth_skills": info["skill_names"],
        })
    return tasks


def set_sacr_params(sg: SkillGraph, l0: int, l1: int, l2: int) -> None:
    sg.router = SparseAttentionCommunityRouter(
        sg.hsc.communities,
        top_k_level0=l0,
        top_k_level1=l1,
        top_k_level2=l2,
        temperature=1.0,
    )


def evaluate(
    sg: SkillGraph,
    tasks: List[Dict],
    query_embeddings: np.ndarray,
    use_sacr: bool,
    use_keywords: bool,
    top_k: int = 10,
) -> Dict:
    hits = 0
    latencies = []
    for task, q_emb in zip(tasks, query_embeddings):
        gt_skills = set(s.lower() for s in task["ground_truth_skills"])
        t0 = time.perf_counter()
        result = sg.retrieve_from_embedding(
            q_emb,
            query=task["query"],
            top_k=top_k,
            use_sacr=use_sacr,
            use_keywords=use_keywords,
            use_sre=False,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_names = set(s.name.lower() for s in result.skills)
        if gt_skills & retrieved_names:
            hits += 1
    return {
        "recall_at_10": round(hits / len(tasks), 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "n_hits": hits,
        "n_queries": len(tasks),
    }


def search_best_config(
    sg: SkillGraph, train_tasks: List[Dict], train_embeddings: np.ndarray, use_keywords: bool
) -> Tuple[Tuple[int, int, int], Dict]:
    best_config = None
    best_result = None
    best_recall = -1.0
    for l0, l1, l2 in CONFIGS:
        set_sacr_params(sg, l0, l1, l2)
        result = evaluate(sg, train_tasks, train_embeddings, use_sacr=True, use_keywords=use_keywords)
        recall = result["recall_at_10"]
        if recall > best_recall or (
            recall == best_recall
            and best_result is not None
            and result["avg_latency_ms"] < best_result["avg_latency_ms"]
        ):
            best_recall = recall
            best_config = (l0, l1, l2)
            best_result = result
    return best_config, best_result


def run_cv(sg: SkillGraph, tasks: List[Dict], query_embeddings: np.ndarray) -> Dict:
    n = len(tasks)
    fold_size = n // N_FOLDS
    fold_indices = []
    start = 0
    for i in range(N_FOLDS):
        end = start + fold_size if i < N_FOLDS - 1 else n
        fold_indices.append((start, end))
        start = end

    results = {
        "sacr_only": [],
        "sacr_keyword": [],
        "dense_only": [],
        "dense_keyword": [],
    }

    for fold_idx, (test_start, test_end) in enumerate(fold_indices):
        test_tasks = tasks[test_start:test_end]
        test_embeddings = query_embeddings[test_start:test_end]
        train_indices = list(range(0, test_start)) + list(range(test_end, n))
        train_tasks = [tasks[i] for i in train_indices]
        train_embeddings = query_embeddings[train_indices]

        # SACR only
        best_cfg, _ = search_best_config(sg, train_tasks, train_embeddings, use_keywords=False)
        set_sacr_params(sg, *best_cfg)
        results["sacr_only"].append(evaluate(sg, test_tasks, test_embeddings, use_sacr=True, use_keywords=False))

        # SACR + Keyword
        best_cfg, _ = search_best_config(sg, train_tasks, train_embeddings, use_keywords=True)
        set_sacr_params(sg, *best_cfg)
        results["sacr_keyword"].append(evaluate(sg, test_tasks, test_embeddings, use_sacr=True, use_keywords=True))

        # Dense only
        results["dense_only"].append(evaluate(sg, test_tasks, test_embeddings, use_sacr=False, use_keywords=False))

        # Dense + Keyword
        results["dense_keyword"].append(evaluate(sg, test_tasks, test_embeddings, use_sacr=False, use_keywords=True))

    return results


def summarize(results: Dict) -> Dict:
    summary = {}
    for variant, fold_results in results.items():
        recalls = [r["recall_at_10"] for r in fold_results]
        latencies = [r["avg_latency_ms"] for r in fold_results]
        total_hits = sum(r["n_hits"] for r in fold_results)
        total_queries = sum(r["n_queries"] for r in fold_results)
        summary[variant] = {
            "mean_recall": round(float(np.mean(recalls)), 4),
            "std_recall": round(float(np.std(recalls)), 4),
            "mean_latency": round(float(np.mean(latencies)), 2),
            "overall_recall": round(total_hits / total_queries, 4),
            "total_hits": total_hits,
            "total_queries": total_queries,
        }
    return summary


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    curated_dir = "/tmp/skill-usage-hf/skillsbench/curated"
    tasks = load_skillsbench_tasks(curated_dir)

    # Shuffle
    indices = list(range(len(tasks)))
    random.shuffle(indices)
    tasks = [tasks[i] for i in indices]

    query_texts = [t["query"] for t in tasks]

    print("=" * 60)
    print("KEYWORD MAP COMPARISON: MANUAL vs AUTO")
    print("=" * 60)

    # ---- Manual keyword map ----
    print("\n[1/2] Evaluating with MANUAL keyword map...")
    sg_manual = SkillGraph(
        skills_path="data_real/skills_ucsb_34k.jsonl",
        hsc_path="data_real/hsc_ucsb_34k",
        use_auto_keywords=False,
    )
    query_embeddings_manual = sg_manual.embedder.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
    results_manual = run_cv(sg_manual, tasks, query_embeddings_manual)
    summary_manual = summarize(results_manual)

    # ---- Auto keyword map ----
    print("\n[2/2] Evaluating with AUTO keyword map...")
    sg_auto = SkillGraph(
        skills_path="data_real/skills_ucsb_34k.jsonl",
        hsc_path="data_real/hsc_ucsb_34k",
        use_auto_keywords=True,
    )
    query_embeddings_auto = sg_auto.embedder.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
    results_auto = run_cv(sg_auto, tasks, query_embeddings_auto)
    summary_auto = summarize(results_auto)

    # ---- Print comparison ----
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    for variant in ["sacr_only", "sacr_keyword", "dense_only", "dense_keyword"]:
        name = variant.upper().replace("_", " ")
        m = summary_manual[variant]
        a = summary_auto[variant]
        print(f"\n{name}:")
        print(f"  Manual: {m['mean_recall']:.1%} +/- {m['std_recall']:.1%}  (latency: {m['mean_latency']:.1f}ms)")
        print(f"  Auto:   {a['mean_recall']:.1%} +/- {a['std_recall']:.1%}  (latency: {a['mean_latency']:.1f}ms)")
        diff = a["mean_recall"] - m["mean_recall"]
        print(f"  Diff:   {diff:+.1%} (Auto - Manual)")

    # Save
    output = {
        "manual": summary_manual,
        "auto": summary_auto,
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
    }
    out_path = Path("data_real/results_keyword_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
