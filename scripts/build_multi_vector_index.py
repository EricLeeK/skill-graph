"""Build multi-vector index from 34K synthetic queries.

Maps SynQ data (namespace--repo--name format) to original skills
(namespace--name format) and embeds all synthetic queries.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

SYNQ_PATH = "34k处理后/synthetic_queries_merged.jsonl"
SKILLS_PATH = "data_real/skills_ucsb_34k.jsonl"
OUTPUT_PATH = "data/index/multi_vector_index.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("=" * 60)
    print("Building Multi-Vector Index")
    print("=" * 60)

    # Load original skills in order
    print("\n[1/5] Loading original skills...")
    skills = []
    skill_ids = []
    skill_embeddings = []
    with open(SKILLS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            skill = json.loads(line)
            skills.append(skill)
            skill_ids.append(skill["id"])
            skill_embeddings.append(np.array(skill["embedding"], dtype=np.float32))

    n_skills = len(skills)
    print(f"  Loaded {n_skills} skills")

    # Load synthetic queries, keyed by namespace--name
    print("\n[2/5] Loading synthetic queries...")
    synq_map = {}
    with open(SYNQ_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # Map to original skill ID: namespace--name
            skill_name = r.get("skill_name") or r.get("name", "")
            fixed_id = r["namespace"] + "--" + skill_name
            synq_map[fixed_id] = r["synthetic_queries"]

    print(f"  Loaded {len(synq_map)} SynQ records")

    # Extract all queries in original skill order
    print("\n[3/5] Extracting queries in skill order...")
    all_queries = []
    query_counts = []
    missing_count = 0
    short_count = 0

    for skill in skills:
        sid = skill["id"]
        queries = synq_map.get(sid, [])

        if not queries:
            missing_count += 1
            queries = [""] * 10
        elif len(queries) < 10:
            short_count += 1
            queries = queries + [""] * (10 - len(queries))

        all_queries.extend(queries)
        query_counts.append(len(queries))

    print(f"  Total queries: {len(all_queries)}")
    print(f"  Missing skills: {missing_count}")
    print(f"  Short queries: {short_count}")

    # Embed all queries
    print(f"\n[4/5] Embedding queries with {MODEL_NAME}...")
    embedder = SentenceTransformer(MODEL_NAME)
    t0 = time.perf_counter()
    syn_embeddings = embedder.encode(
        all_queries,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=256,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Embedded {len(all_queries)} queries in {elapsed:.1f}s")

    # Reshape to (n_skills, 10, 384)
    syn_embeddings = syn_embeddings.reshape(n_skills, 10, -1)
    print(f"  Shape: {syn_embeddings.shape}")

    # Stack skill embeddings
    skill_embeddings = np.stack(skill_embeddings)
    print(f"  Skill embeddings shape: {skill_embeddings.shape}")

    # Normalize
    skill_embeddings_normed = skill_embeddings / (
        np.linalg.norm(skill_embeddings, axis=1, keepdims=True) + 1e-10
    )
    syn_embeddings_normed = syn_embeddings / (
        np.linalg.norm(syn_embeddings, axis=2, keepdims=True) + 1e-10
    )

    # Build index
    print("\n[5/5] Saving index...")
    index = {
        "skill_ids": skill_ids,
        "skill_embeddings": skill_embeddings_normed,
        "synthetic_embeddings": syn_embeddings_normed,
        "metadata": {
            "n_skills": n_skills,
            "queries_per_skill": 10,
            "embed_dim": syn_embeddings.shape[-1],
            "model": MODEL_NAME,
            "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "synq_source": "34k处理后/synthetic_queries_34k.jsonl",
            "missing_skills": missing_count,
            "short_skills": short_count,
        },
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(index, f)

    file_size_mb = Path(OUTPUT_PATH).stat().st_size / (1024 * 1024)
    print(f"  Saved to {OUTPUT_PATH}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Skills: {n_skills}")
    print(f"  Embeddings per skill: 1 skill + 10 synthetic")

    print("\nDone.")


if __name__ == "__main__":
    main()
