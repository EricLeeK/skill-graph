"""Clean manual benchmark: keep only tasks with GT rank <= 100."""
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

BENCHMARK_DIR = Path("/tmp/skill-usage-hf/skillsbench/curated")
INDEX_PATH = "data/index/multi_vector_index.pkl"

# Load data
with open(BENCHMARK_DIR / "manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)

name_to_indices = {}
for i, sid in enumerate(index["skill_ids"]):
    name = sid.split("--")[-1] if "--" in sid else sid
    name_to_indices.setdefault(name, []).append(i)

def extract_query(instruction, max_len=500):
    paragraphs = instruction.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if para.startswith("#"):
            continue
        if len(para) < 20:
            continue
        return para[:max_len]
    return instruction[:max_len]

# Get manual tasks
task_list = list(manifest.keys())
manual_tasks = task_list[87:]

print(f"Manual tasks: {len(manual_tasks)}")

# Encode queries
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
queries = []
for task_name in manual_tasks:
    task_dir = BENCHMARK_DIR / task_name
    instr = (task_dir / "instruction.md").read_text().strip()
    queries.append(extract_query(instr))

query_embeddings = embedder.encode(queries, convert_to_numpy=True, show_progress_bar=True)
query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

# Compute ranks
keep_tasks = []
delete_tasks = []

for task_name, q_emb in zip(manual_tasks, query_embeddings):
    gt_skills = manifest[task_name]["skill_names"]
    q = q_emb / (np.linalg.norm(q_emb) + 1e-10)

    skill_sims = index["skill_embeddings"] @ q
    syn_sims = index["synthetic_embeddings"] @ q
    best_syn_sims = np.max(syn_sims, axis=1)
    combined = np.maximum(skill_sims, best_syn_sims)

    best_rank = 99999
    for gt in gt_skills:
        indices = name_to_indices.get(gt, [])
        for idx in indices:
            gt_score = combined[idx]
            rank = int(np.sum(combined > gt_score) + 1)
            if rank < best_rank:
                best_rank = rank

    if best_rank <= 100:
        keep_tasks.append((task_name, best_rank))
    else:
        delete_tasks.append((task_name, best_rank))

print(f"\nTasks to keep (rank <= 100): {len(keep_tasks)}")
print(f"Tasks to delete (rank > 100): {len(delete_tasks)}")

# Delete poor-quality tasks
for task_name, rank in delete_tasks:
    del manifest[task_name]
    task_dir = BENCHMARK_DIR / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)

# Save updated manifest
with open(BENCHMARK_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\nCleaned benchmark: {len(manifest)} tasks ({len(manifest) - 87} manual)")
print(f"Kept tasks median rank: {np.median([r for _, r in keep_tasks]):.0f}")

# Show kept tasks
print(f"\nKept tasks (sorted by rank):")
keep_tasks.sort(key=lambda x: x[1])
for task_name, rank in keep_tasks:
    print(f"  {task_name:40s} rank={rank:4d}")
