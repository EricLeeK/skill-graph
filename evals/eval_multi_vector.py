"""Evaluate multi-vector index (Dense + SynQ) on SkillsBench with 5-Fold CV.

Corrects the O1 Pilot mistake: candidate pool is the FULL 34K skills, not a subset.

Variants:
1. Dense baseline: skill embedding only, full 34K pool
2. Dense + SynQ: max(skill_sim, best_syn_sim), full 34K pool
3. SACR only: existing SACR router + dense reranking
4. SACR + SynQ: SACR router + multi-vector reranking
"""
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from skill_graph import SkillGraph
from skill_graph.core.sacr import SparseAttentionCommunityRouter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_PATH = "data/index/multi_vector_index.pkl"
SKILLS_PATH = "data_real/skills_ucsb_34k.jsonl"
N_FOLDS = 5
RANDOM_SEED = 42
TOP_K = 10

# SACR config search space (same as eval_cross_validation.py)
SACR_CONFIGS: List[Tuple[int, int, int]] = [
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_multi_vector_index(path: str) -> Dict:
    """Load the pre-built multi-vector index."""
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
    """Map skill_id -> skill_name for ground truth matching."""
    name_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            skill = json.loads(line)
            name_map[skill["id"]] = skill["name"]
    return name_map


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


def retrieve_dense_baseline(
    q_emb: np.ndarray,
    index: Dict,
    top_k: int = TOP_K,
) -> List[str]:
    """Dense retrieval using only skill embeddings (full 34K pool)."""
    q = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    scores = index["skill_embeddings"] @ q  # (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [index["skill_ids"][i] for i in top_indices]


def retrieve_dense_synq(
    q_emb: np.ndarray,
    index: Dict,
    top_k: int = TOP_K,
) -> List[str]:
    """Dense retrieval using max(skill_sim, best_syn_sim) (full 34K pool)."""
    q = q_emb / (np.linalg.norm(q_emb) + 1e-10)

    # Skill similarities: (N,)
    skill_sims = index["skill_embeddings"] @ q

    # Synthetic query similarities: (N, 10)
    syn_sims = index["synthetic_embeddings"] @ q  # (N, 10)
    best_syn_sims = np.max(syn_sims, axis=1)  # (N,)

    # Combined score: max
    combined_scores = np.maximum(skill_sims, best_syn_sims)

    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    return [index["skill_ids"][i] for i in top_indices]


def retrieve_sacr_baseline(
    q_emb: np.ndarray,
    sg: SkillGraph,
    top_k: int = TOP_K,
) -> List[str]:
    """SACR routing + dense reranking on candidates."""
    result = sg.retrieve_from_embedding(
        q_emb,
        top_k=top_k,
        use_sacr=True,
        use_keywords=False,
        use_sre=False,
    )
    return [s.id for s in result.skills]


def retrieve_sacr_synq(
    q_emb: np.ndarray,
    sg: SkillGraph,
    index: Dict,
    top_k: int = TOP_K,
) -> List[str]:
    """SACR routing + multi-vector reranking on candidates."""
    # Step 1: SACR route to get candidate communities
    route_result = sg.router.route(q_emb)
    candidate_ids = route_result["skill_ids"]

    if not candidate_ids:
        candidate_ids = sg.skill_ids

    # Step 2: Build id -> index mapping
    id_to_idx = {sid: i for i, sid in enumerate(index["skill_ids"])}

    # Step 3: Compute multi-vector scores for candidates
    q = q_emb / (np.linalg.norm(q_emb) + 1e-10)

    candidate_indices = []
    for sid in candidate_ids:
        if sid in id_to_idx:
            candidate_indices.append(id_to_idx[sid])

    if not candidate_indices:
        return []

    cand_skill_sims = index["skill_embeddings"][candidate_indices] @ q
    cand_syn_sims = index["synthetic_embeddings"][candidate_indices] @ q  # (M, 10)
    cand_best_syn_sims = np.max(cand_syn_sims, axis=1)
    cand_scores = np.maximum(cand_skill_sims, cand_best_syn_sims)

    # Step 4: Rank and return top-k
    if len(cand_scores) <= top_k:
        top_local = np.argsort(-cand_scores)
    else:
        top_local = np.argpartition(cand_scores, -top_k)[-top_k:]
        top_local = top_local[np.argsort(-cand_scores[top_local])]

    return [index["skill_ids"][candidate_indices[i]] for i in top_local[:top_k]]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_variant(
    tasks: List[Dict],
    query_embeddings: np.ndarray,
    skill_names: Dict[str, str],
    retrieve_fn,
    top_k: int = TOP_K,
) -> Dict:
    """Evaluate a retrieval variant on a set of tasks."""
    hits = 0
    latencies = []

    for task, q_emb in zip(tasks, query_embeddings):
        gt_skills = set(s.lower() for s in task["ground_truth_skills"])

        t0 = time.perf_counter()
        retrieved_ids = retrieve_fn(q_emb)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_names = set()
        for sid in retrieved_ids:
            name = skill_names.get(sid, sid.split("--")[-1] if "--" in sid else sid)
            retrieved_names.add(name.lower())

        if gt_skills & retrieved_names:
            hits += 1

    recall = hits / len(tasks) if tasks else 0.0
    return {
        "recall_at_10": round(recall, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "n_queries": len(tasks),
        "n_hits": hits,
    }


def search_best_sacr_config(
    sg: SkillGraph,
    train_tasks: List[Dict],
    train_embeddings: np.ndarray,
    skill_names: Dict[str, str],
    use_synq: bool,
    index: Dict,
) -> Tuple[Tuple[int, int, int], Dict]:
    """Grid-search SACR configs on training fold."""
    best_config = None
    best_result = None
    best_recall = -1.0

    for l0, l1, l2 in SACR_CONFIGS:
        sg.router = SparseAttentionCommunityRouter(
            sg.hsc.communities,
            top_k_level0=l0,
            top_k_level1=l1,
            top_k_level2=l2,
            temperature=1.0,
        )

        if use_synq:
            result = evaluate_variant(
                train_tasks, train_embeddings, skill_names,
                lambda q: retrieve_sacr_synq(q, sg, index, TOP_K)
            )
        else:
            result = evaluate_variant(
                train_tasks, train_embeddings, skill_names,
                lambda q: retrieve_sacr_baseline(q, sg, TOP_K)
            )

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


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------


def run_cross_validation(
    tasks: List[Dict],
    query_embeddings: np.ndarray,
    index: Dict,
    sg: SkillGraph,
    skill_names: Dict[str, str],
) -> Dict:
    """Run 5-fold cross-validation for all variants."""
    n = len(tasks)
    fold_size = n // N_FOLDS

    fold_indices = []
    start = 0
    for i in range(N_FOLDS):
        end = start + fold_size if i < N_FOLDS - 1 else n
        fold_indices.append((start, end))
        start = end

    results = {
        "dense_baseline": [],
        "dense_synq": [],
        "sacr_baseline": [],
        "sacr_synq": [],
    }
    best_configs = {
        "sacr_baseline": [],
        "sacr_synq": [],
    }

    for fold_idx, (test_start, test_end) in enumerate(fold_indices):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'=' * 60}")

        test_tasks = tasks[test_start:test_end]
        test_embeddings = query_embeddings[test_start:test_end]

        train_indices = list(range(0, test_start)) + list(range(test_end, n))
        train_tasks = [tasks[i] for i in train_indices]
        train_embeddings = query_embeddings[train_indices]

        print(f"  Train: {len(train_tasks)} | Test: {len(test_tasks)}")

        # --- Dense Baseline ---
        print("  [1/4] Dense baseline...")
        result = evaluate_variant(
            test_tasks, test_embeddings, skill_names,
            lambda q: retrieve_dense_baseline(q, index, TOP_K)
        )
        results["dense_baseline"].append(result)
        print(f"        Recall@10: {result['recall_at_10']:.1%}")

        # --- Dense + SynQ ---
        print("  [2/4] Dense + SynQ...")
        result = evaluate_variant(
            test_tasks, test_embeddings, skill_names,
            lambda q: retrieve_dense_synq(q, index, TOP_K)
        )
        results["dense_synq"].append(result)
        print(f"        Recall@10: {result['recall_at_10']:.1%}")

        # --- SACR Baseline ---
        print("  [3/4] SACR baseline (searching config)...")
        best_cfg, _ = search_best_sacr_config(
            sg, train_tasks, train_embeddings, skill_names, use_synq=False, index=index
        )
        best_configs["sacr_baseline"].append({
            "fold": fold_idx + 1,
            "l0": best_cfg[0], "l1": best_cfg[1], "l2": best_cfg[2]
        })
        sg.router = SparseAttentionCommunityRouter(
            sg.hsc.communities,
            top_k_level0=best_cfg[0],
            top_k_level1=best_cfg[1],
            top_k_level2=best_cfg[2],
            temperature=1.0,
        )
        result = evaluate_variant(
            test_tasks, test_embeddings, skill_names,
            lambda q: retrieve_sacr_baseline(q, sg, TOP_K)
        )
        result["best_config"] = best_cfg
        results["sacr_baseline"].append(result)
        print(f"        Best: L0={best_cfg[0]} L1={best_cfg[1]} L2={best_cfg[2]}")
        print(f"        Recall@10: {result['recall_at_10']:.1%}")

        # --- SACR + SynQ ---
        print("  [4/4] SACR + SynQ (searching config)...")
        best_cfg, _ = search_best_sacr_config(
            sg, train_tasks, train_embeddings, skill_names, use_synq=True, index=index
        )
        best_configs["sacr_synq"].append({
            "fold": fold_idx + 1,
            "l0": best_cfg[0], "l1": best_cfg[1], "l2": best_cfg[2]
        })
        sg.router = SparseAttentionCommunityRouter(
            sg.hsc.communities,
            top_k_level0=best_cfg[0],
            top_k_level1=best_cfg[1],
            top_k_level2=best_cfg[2],
            temperature=1.0,
        )
        result = evaluate_variant(
            test_tasks, test_embeddings, skill_names,
            lambda q: retrieve_sacr_synq(q, sg, index, TOP_K)
        )
        result["best_config"] = best_cfg
        results["sacr_synq"].append(result)
        print(f"        Best: L0={best_cfg[0]} L1={best_cfg[1]} L2={best_cfg[2]}")
        print(f"        Recall@10: {result['recall_at_10']:.1%}")

    return results, best_configs


def compute_summary(results: List[Dict]) -> Dict:
    recalls = [r["recall_at_10"] for r in results]
    latencies = [r["avg_latency_ms"] for r in results]
    total_hits = sum(r["n_hits"] for r in results)
    total_queries = sum(r["n_queries"] for r in results)
    return {
        "mean_recall_at_10": round(float(np.mean(recalls)), 4),
        "std_recall_at_10": round(float(np.std(recalls)), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "total_hits": total_hits,
        "total_queries": total_queries,
        "overall_recall_at_10": round(total_hits / total_queries, 4),
        "per_fold": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Multi-Vector Index Evaluation (34K Full Pool)")
    print("=" * 60)

    # Load multi-vector index
    print("\n[1/4] Loading multi-vector index...")
    index = load_multi_vector_index(INDEX_PATH)
    print(f"  Skills: {len(index['skill_ids'])}")
    print(f"  Embeddings: skill={index['skill_embeddings'].shape}, syn={index['synthetic_embeddings'].shape}")

    # Load SkillGraph (for SACR)
    print("\n[2/4] Loading SkillGraph...")
    sg = SkillGraph(
        skills_path=SKILLS_PATH,
        hsc_path="data_real/hsc_ucsb_34k",
        use_auto_keywords=False,
    )

    # Load skill names
    print("\n[3/4] Loading skill names...")
    skill_names = load_skill_names(SKILLS_PATH)
    print(f"  Names: {len(skill_names)}")

    # Load SkillsBench
    print("\n[4/4] Loading SkillsBench tasks...")
    curated_dir = "/tmp/skill-usage-hf/skillsbench/curated_terminus"
    tasks = load_skillsbench_tasks(curated_dir)
    print(f"  Tasks: {len(tasks)}")

    # Shuffle
    indices = list(range(len(tasks)))
    random.shuffle(indices)
    tasks = [tasks[i] for i in indices]

    # Encode queries
    print("\nEncoding benchmark queries...")
    query_texts = [t["query"] for t in tasks]
    query_embeddings = sg.embedder.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Run CV
    print(f"\n{'=' * 60}")
    print(f"Running {N_FOLDS}-Fold Cross-Validation")
    print(f"{'=' * 60}")
    print("Candidate pool: FULL 34,396 skills")
    print("Variants: Dense, Dense+SynQ, SACR, SACR+SynQ")

    results, best_configs = run_cross_validation(tasks, query_embeddings, index, sg, skill_names)

    # Summarize
    summaries = {k: compute_summary(v) for k, v in results.items()}

    # Print
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    for variant, stats in summaries.items():
        name = variant.upper().replace("_", " ")
        print(f"\n{name}:")
        print(f"  Recall@10: {stats['mean_recall_at_10']:.1%} +/- {stats['std_recall_at_10']:.1%}")
        print(f"  Latency:   {stats['mean_latency_ms']:.2f} +/- {stats['std_latency_ms']:.2f} ms")
        print(f"  Overall:   {stats['total_hits']}/{stats['total_queries']} = {stats['overall_recall_at_10']:.1%}")

    # Compare
    print(f"\n{'=' * 60}")
    print("COMPARISONS")
    print(f"{'=' * 60}")

    dense_base = summaries["dense_baseline"]["mean_recall_at_10"]
    dense_synq = summaries["dense_synq"]["mean_recall_at_10"]
    sacr_base = summaries["sacr_baseline"]["mean_recall_at_10"]
    sacr_synq = summaries["sacr_synq"]["mean_recall_at_10"]

    print(f"\nDense + SynQ vs Dense:     {dense_synq - dense_base:+.1%} ({dense_synq:.1%} - {dense_base:.1%})")
    print(f"SACR + SynQ vs SACR:       {sacr_synq - sacr_base:+.1%} ({sacr_synq:.1%} - {sacr_base:.1%})")
    print(f"SACR + SynQ vs Dense:      {sacr_synq - dense_base:+.1%} ({sacr_synq:.1%} - {dense_base:.1%})")

    # Save
    output = {
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "n_tasks": len(tasks),
        "n_skills": len(index["skill_ids"]),
        "candidate_pool": "full_34k",
        "best_configs": best_configs,
        "summaries": summaries,
    }

    out_path = Path("data/results_multi_vector_cv.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
