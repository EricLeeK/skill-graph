"""Evaluate Dense + SynQ with skill-name deduplication in top-k.

Problem: Same skill name appears multiple times from different namespaces,
e.g. 3 copies of 'd3js-visualization' occupy top-3 slots, pushing out
the actual GT skill 'd3-visualization'.

Solution: In top-k retrieval, if multiple results share the same skill name,
keep only the highest-scoring one. This frees up slots for diverse skills.
"""
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer

RANDOM_SEED = 42
TOP_K = 10

INDEX_PATH = "data/index/multi_vector_index.pkl"
SKILLS_PATH = "data_real/skills_ucsb_34k.jsonl"


def load_multi_vector_index(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


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


def load_skill_names(path: str) -> Dict[str, str]:
    name_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            skill = json.loads(line)
            name_map[skill["id"]] = skill["name"]
    return name_map


def retrieve_dense_synq(q_emb: np.ndarray, index: Dict, top_k: int = TOP_K) -> List[str]:
    q = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    skill_sims = index["skill_embeddings"] @ q
    syn_sims = index["synthetic_embeddings"] @ q
    best_syn_sims = np.max(syn_sims, axis=1)
    combined = np.maximum(skill_sims, best_syn_sims)
    top_indices = np.argsort(combined)[::-1][:top_k]
    return [index["skill_ids"][i] for i in top_indices]


def retrieve_dense_synq_dedup(
    q_emb: np.ndarray,
    index: Dict,
    skill_id_to_name: Dict[str, str],
    top_k: int = TOP_K,
) -> List[str]:
    """Retrieve with skill-name deduplication: same name appears at most once."""
    q = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    skill_sims = index["skill_embeddings"] @ q
    syn_sims = index["synthetic_embeddings"] @ q
    best_syn_sims = np.max(syn_sims, axis=1)
    combined = np.maximum(skill_sims, best_syn_sims)

    # Sort all by score descending
    sorted_indices = np.argsort(combined)[::-1]

    # Greedily pick highest-scoring unique skill names
    result = []
    seen_names = set()
    for idx in sorted_indices:
        sid = index["skill_ids"][idx]
        name = skill_id_to_name.get(sid, sid)
        if name in seen_names:
            continue
        seen_names.add(name)
        result.append(sid)
        if len(result) >= top_k:
            break

    return result


def evaluate(
    tasks: List[Dict],
    query_embeddings: np.ndarray,
    skill_names: Dict[str, str],
    index: Dict,
    use_dedup: bool = False,
) -> Dict:
    hits = 0
    per_case = []

    for task, q_emb in zip(tasks, query_embeddings):
        gt_skills = set(s.lower() for s in task["ground_truth_skills"])

        if use_dedup:
            retrieved = retrieve_dense_synq_dedup(q_emb, index, skill_names, TOP_K)
        else:
            retrieved = retrieve_dense_synq(q_emb, index, TOP_K)

        retrieved_names = set()
        for sid in retrieved:
            name = skill_names.get(sid, sid.split("--")[-1] if "--" in sid else sid)
            retrieved_names.add(name.lower())

        hit = bool(gt_skills & retrieved_names)
        if hit:
            hits += 1

        per_case.append({
            "task_name": task["task_name"],
            "hit": hit,
            "retrieved": [skill_names.get(sid, sid) for sid in retrieved],
        })

    return {
        "recall_at_10": round(hits / len(tasks), 4),
        "hits": hits,
        "total": len(tasks),
        "per_case": per_case,
    }


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Eval: Dense + SynQ with Skill-Name Deduplication")
    print("=" * 60)

    print("\n[1/4] Loading index...")
    index = load_multi_vector_index(INDEX_PATH)
    print(f"  Skills in index: {len(index['skill_ids'])}")

    print("\n[2/4] Loading skill names...")
    skill_names = load_skill_names(SKILLS_PATH)
    print(f"  Names: {len(skill_names)}")

    # Count duplicate names
    name_counts = {}
    for sid, name in skill_names.items():
        name_counts[name] = name_counts.get(name, 0) + 1
    dup_count = sum(1 for c in name_counts.values() if c > 1)
    dup_instances = sum(c for c in name_counts.values() if c > 1)
    print(f"  Duplicate names: {dup_count} ({dup_instances} instances)")

    print("\n[3/4] Loading tasks...")
    tasks = load_skillsbench_tasks("/tmp/skill-usage-hf/skillsbench/curated")
    print(f"  Tasks: {len(tasks)}")

    print("\n[4/4] Encoding queries...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_texts = [t["query"] for t in tasks]
    query_embeddings = embedder.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    baseline = evaluate(tasks, query_embeddings, skill_names, index, use_dedup=False)
    dedup = evaluate(tasks, query_embeddings, skill_names, index, use_dedup=True)

    print(f"\nBaseline (no dedup):")
    print(f"  Recall@10: {baseline['recall_at_10']:.1%} ({baseline['hits']}/{baseline['total']})")

    print(f"\nWith deduplication:")
    print(f"  Recall@10: {dedup['recall_at_10']:.1%} ({dedup['hits']}/{dedup['total']})")

    diff = dedup['recall_at_10'] - baseline['recall_at_10']
    print(f"\nDifference: {diff:+.1%}")

    # Show cases that changed
    improved = []
    degraded = []
    for b, d in zip(baseline["per_case"], dedup["per_case"]):
        if not b["hit"] and d["hit"]:
            improved.append(b["task_name"])
        elif b["hit"] and not d["hit"]:
            degraded.append(b["task_name"])

    if improved:
        print(f"\nImproved cases ({len(improved)}):")
        for name in improved:
            print(f"  + {name}")
    if degraded:
        print(f"\nDegraded cases ({len(degraded)}):")
        for name in degraded:
            print(f"  - {name}")

    # Save detailed results
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dedup_comparison.json", "w") as f:
        json.dump({
            "baseline": baseline["recall_at_10"],
            "dedup": dedup["recall_at_10"],
            "diff": diff,
            "improved": improved,
            "degraded": degraded,
            "per_case_baseline": baseline["per_case"],
            "per_case_dedup": dedup["per_case"],
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_dir / 'dedup_comparison.json'}")


if __name__ == "__main__":
    main()
