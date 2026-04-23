"""SynQ quantity ablation study.

Tests if reducing synthetic queries per skill from 10 to 5, 3, or 1
maintains recall while reducing latency.

Variants:
1. Dense baseline (0 synQ)
2. SynQ-1 (1 synthetic query per skill)
3. SynQ-3 (3 synthetic queries per skill)
4. SynQ-5 (5 synthetic queries per skill)
5. SynQ-10 (10 synthetic queries per skill - full)
"""
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# Config
RANDOM_SEED = 42
TOP_K = 10
INDEX_PATH = "data/index/multi_vector_index.pkl"
SKILLS_PATH = "data_real/skills_ucsb_34k.jsonl"
BENCHMARK_DIR = Path("/tmp/skill-usage-hf/skillsbench/curated")


def load_index(path: str) -> Dict:
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
        return para[:max_len]
    return instruction[:max_len]


def load_benchmark_tasks() -> List[Dict]:
    with open(BENCHMARK_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = []
    for task_name, info in manifest.items():
        if not info.get("has_skills"):
            continue
        task_dir = BENCHMARK_DIR / task_name
        instruction = (task_dir / "instruction.md").read_text().strip()
        query = extract_query(instruction)
        tasks.append({
            "task_name": task_name,
            "query": query,
            "ground_truth_skills": info["skill_names"],
        })
    return tasks


def build_subsampled_index(full_index: Dict, n_syn: int) -> Dict:
    """Build index with only first N synthetic queries per skill."""
    if n_syn == 0:
        return {
            "skill_ids": full_index["skill_ids"],
            "skill_embeddings": full_index["skill_embeddings"],
            "synthetic_embeddings": None,
        }

    # Extract first n_syn synthetic queries: (N, n_syn, 384)
    syn_emb = full_index["synthetic_embeddings"][:, :n_syn, :]
    return {
        "skill_ids": full_index["skill_ids"],
        "skill_embeddings": full_index["skill_embeddings"],
        "synthetic_embeddings": syn_emb,
    }


def retrieve(q_emb: np.ndarray, index: Dict, top_k: int = TOP_K) -> List[str]:
    q = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    skill_sims = index["skill_embeddings"] @ q

    if index["synthetic_embeddings"] is not None:
        syn_sims = index["synthetic_embeddings"] @ q  # (N, n_syn)
        best_syn_sims = np.max(syn_sims, axis=1)
        combined = np.maximum(skill_sims, best_syn_sims)
    else:
        combined = skill_sims

    top_indices = np.argsort(combined)[::-1][:top_k]
    return [index["skill_ids"][i] for i in top_indices]


def evaluate_variant(tasks: List[Dict], query_embeddings: np.ndarray, index: Dict) -> Dict:
    hits = 0
    latencies = []

    for task, q_emb in zip(tasks, query_embeddings):
        gt_skills = set(s.lower() for s in task["ground_truth_skills"])

        t0 = time.perf_counter()
        retrieved = retrieve(q_emb, index, TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_names = set()
        for sid in retrieved:
            name = sid.split("--")[-1] if "--" in sid else sid
            retrieved_names.add(name.lower())

        if gt_skills & retrieved_names:
            hits += 1

    return {
        "recall_at_10": round(hits / len(tasks), 4),
        "hits": hits,
        "total": len(tasks),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
    }


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("SynQ Quantity Ablation Study")
    print("=" * 60)

    print("\n[1/4] Loading full index...")
    full_index = load_index(INDEX_PATH)
    print(f"  Skills: {len(full_index['skill_ids'])}")
    print(f"  SynQ per skill: {full_index['synthetic_embeddings'].shape[1]}")

    print("\n[2/4] Loading benchmark tasks...")
    tasks = load_benchmark_tasks()
    print(f"  Tasks: {len(tasks)}")

    print("\n[3/4] Encoding queries...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_texts = [t["query"] for t in tasks]
    query_embeddings = embedder.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    print("\n[4/4] Evaluating variants...")
    results = {}

    for n_syn in [0, 1, 3, 5, 10]:
        print(f"\n  Variant: SynQ-{n_syn}")
        index = build_subsampled_index(full_index, n_syn)
        result = evaluate_variant(tasks, query_embeddings, index)
        results[f"synq_{n_syn}"] = result
        print(f"    Recall@10: {result['recall_at_10']:.1%} ({result['hits']}/{result['total']})")
        print(f"    Latency: {result['avg_latency_ms']:.2f}ms (median: {result['median_latency_ms']:.2f}ms)")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Variant':>12s} {'Recall@10':>10s} {'Latency':>10s} {'Delta':>8s}")
    print("-" * 50)
    baseline = results["synq_0"]["recall_at_10"]
    for n_syn in [0, 1, 3, 5, 10]:
        r = results[f"synq_{n_syn}"]
        delta = r["recall_at_10"] - baseline
        print(f"{'SynQ-' + str(n_syn):>12s} {r['recall_at_10']:>9.1%} {r['avg_latency_ms']:>9.2f}ms {delta:>+7.1%}")

    # Save results
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synq_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'synq_ablation.json'}")


if __name__ == "__main__":
    main()
