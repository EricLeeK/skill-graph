"""Build SkillGraph dataset from UCSB Skill-Usage 34K skills."""
import json
import random
import uuid
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_skills(meta_path: str) -> list:
    """Load skills from skills_meta.jsonl."""
    skills = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            skills.append({
                "id": data.get("skill_id", data.get("id", str(uuid.uuid4())[:8])),
                "name": data.get("skill_name", data.get("name", "unknown")),
                "description": data.get("description", ""),
                "category_tags": ["Agent Skill"],
                "source": data.get("source", ""),
                "owner": data.get("owner", ""),
                "repo": data.get("repo", ""),
                "installs": data.get("installs", 0),
                "github_stars": data.get("github_stars", 0),
                "license": data.get("github_license", ""),
            })
    return skills


def add_embeddings(skills, model_name="all-MiniLM-L6-v2", batch_size=256):
    model = SentenceTransformer(model_name)
    texts = [f"{s['name']}: {s['description']}" for s in skills]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    for skill, emb in zip(skills, embeddings):
        skill["embedding"] = emb.tolist()
    return skills


def add_relationships(skills, top_k=5):
    embeddings = np.array([s["embedding"] for s in skills], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    n = len(skills)
    chunk_size = 1000
    for i in tqdm(range(0, n, chunk_size), desc="Building relationships"):
        end = min(i + chunk_size, n)
        sims = embeddings[i:end] @ embeddings.T
        for offset, row_sims in enumerate(sims):
            idx = i + offset
            row_sims[idx] = -1.0
            top_indices = np.argpartition(row_sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-row_sims[top_indices])]
            skills[idx]["related_skill_ids"] = [skills[j]["id"] for j in top_indices]
    return skills


def main():
    random.seed(42)
    np.random.seed(42)

    meta_path = "/tmp/skill-usage-hf/skills-34k/skills_meta.jsonl"
    print(f"Loading skills from {meta_path}...")
    skills = load_skills(meta_path)
    print(f"Loaded {len(skills)} skills")

    print("\nComputing embeddings...")
    skills = add_embeddings(skills)

    print("\nBuilding relationships...")
    skills = add_relationships(skills)

    out_path = Path("data_real/skills_ucsb_34k.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in skills:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(skills)} skills to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
