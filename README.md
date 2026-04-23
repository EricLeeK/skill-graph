
<h1 align="center">
  <br>
  <img src="https://img.shields.io/badge/SkillGraph-1B365D?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzFCMzY1RCIvPjx0ZXh0IHg9IjUwIiB5PSI2NSIgZm9udC1zaXplPSI1NSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IndoaXRlIj5TPC90ZXh0Pjwvc3ZnPg==&logoColor=white" alt="SkillGraph">
  <br>
  SkillGraph
  <br>
  <a href="https://arxiv.org/abs/2604.xxxxx"><img src="https://img.shields.io/badge/arXiv-cs.AI%20%7C%20cs.IR-1B365D?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://github.com/EricLeeK/skill-graph"><img src="https://img.shields.io/badge/GitHub-skill--graph-1B365D?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://github.com/EricLeeK/skill-graph/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-1B365D?style=flat-square" alt="License"></a>
  <br>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Embedding-all--MiniLM--L6--v2-FF6F00?style=flat-square" alt="Embedding">
</h1>

> **LLM-for-Index, Zero-Token Runtime**: One-time semantic reconstruction for scalable skill retrieval in LLM agents.

---

## TL;DR

- <span style="color:#1B365D">**</span> **Problem**: 34,396 real-world skill descriptions are **semantically incomplete**---written in function-oriented language, while users query in task-oriented language.
- <span style="color:#1B365D">**</span> **Solution**: LLM generates 10 diverse synthetic user queries per skill, building a **multi-vector index** that bridges the semantic gap.
- <span style="color:#1B365D">**</span> **Result**: **71.0% Recall@10** at **22 ms** with **zero runtime tokens**, surpassing UCSB Agentic (68.3%, ~seconds, ~5K tokens).

---

## The Semantic Gap

A real example from our dataset:

| User Query | Skill Description |
|-----------|-------------------|
| "How do I get my website online?" | "Vercel deployment workflow: serverless function configuration, edge caching, and CI/CD pipeline integration." |

**No shared keywords. Low cosine similarity. Same task.**

This is not a vocabulary problem---it's a **pragmatic distributional mismatch**. Skill descriptions across the entire ecosystem share three structural deficiencies:

1. <span style="color:#1B365D">**</span> **Function-oriented, not task-oriented**: Explain what the tool does technically, not what problems it solves.
2. <span style="color:#1B365D">**</span> **Template-like and homogeneous**: Similar patterns limit coverage of diverse user expressions.
3. <span style="color:#1B365D">**</span> **Missing usage scenarios**: Critical context like "when a new team member joins" is entirely absent.

---

## Architecture: Two-Phase Design

```
┌────────────────────────────────────────────┐
│  PREPROCESSING PHASE (one-time, token-acceptable)   │
├────────────────────────────────────────────┤
│  34,396 Skills → LLM SynQ Gen → 10 queries/skill     │
│                     → Embed (all-MiniLM-L6-v2)        │
│                     → Multi-Vector Index stored       │
│  Cost: ~$765 (one-time)                               │
└────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────┐
│  RUNTIME PHASE (per-query, zero-token)                │
├────────────────────────────────────────────┤
│  User Query → Embed → Dot Product (skill_emb +       │
│              syn_emb) → Top-k Ranking                  │
│  Latency: 22 ms  ·  Runtime Tokens: 0               │
└────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/EricLeeK/SynCRAG-skill.git
cd SynCRAG-skill
uv pip install -e ".[dev]"

# Run evaluation
python evals/eval_multi_vector.py
```

### Python API

```python
from skill_graph import SkillGraph

sg = SkillGraph(
    skills_path="data_real/skills_ucsb_34k.jsonl",
    index_path="data/index/multi_vector_index.pkl",
)

result = sg.retrieve("Deploy a microservice to Kubernetes", top_k=5)
for skill in result.skills:
    print(f"{skill.name}: {skill.description}")
```

### FastAPI Server

```bash
uvicorn skill_graph.api.server:app --reload
```

---

## Evaluation Results

### SkillsBench (87 tasks on 34,396 UCSB skills)

| System | Recall@10 | Latency | Runtime Tokens |
|--------|-----------|---------|----------------|
| **Dense + SynQ (ours)** | **71.0%** | **22 ms** | **0** |
| Dense baseline (ours) | 62.8% | 3.2 ms | 0 |
| UCSB Agentic | 68.3% | ~seconds | ~5,000 |
| **SynQ Improvement** | **+8.1pp** | — | — |

**Key findings**:
- Synthetic queries provide **+8.1pp** improvement over dense retrieval on raw descriptions.
- Surpasses UCSB Agentic (68.3%) by **+2.7pp** while being **136x faster** with **zero runtime tokens**.
- The entire runtime is a single vector operation: no LLM calls, no API costs.

---

## Project Structure

```
skill_graph/
├── api/
│   ├── skill_graph.py    # Core API: multi-vector dense retrieval
│   └── server.py         # FastAPI service
├── core/
│   ├── graph.py          # Skill graph manager
│   └── sre.py            # Skill refinement engine
├── matching/
│   ├── hybrid_ranker.py  # Semantic + keyword boost
│   └── keyword_matcher.py# Exact keyword matching
└── models.py             # Pydantic data models
evals/
├── eval_multi_vector.py   # 5-Fold CV evaluation
└── eval_skillsbench.py    # SkillsBench evaluation
scripts/
└── build_multi_vector_index.py  # Build SynQ index
```

---

## Market Argument: Platform-Owned Semantic Reconstruction

We argue that the one-time semantic reconstruction (synthetic query generation) is best performed by **skill marketplace platforms** (e.g., OpenAI's GPT Store, GitHub's skill registries) rather than individual agent developers.

**Why?**
- **Universal inadequacy**: All 34K skills exhibit the same description deficiencies---it is systemic, not individual.
- **Positive externality**: A platform generates SynQ once, every downstream agent benefits.
- **Economies of scale**: ~$765 for 34K skills is modest for a platform, prohibitive for individual developers.

This creates a natural division of labor: **platforms invest in semantic quality, agents enjoy zero-token retrieval**.

---

## Citation

```bibtex
@article{li2026skillgraph,
  title={LLM-for-Index, Zero-Token Runtime: One-Time Semantic Reconstruction for Scalable Skill Retrieval in LLM Agents},
  author={Li, Shiyao and Zhang, Jiale},
  journal={arXiv preprint arXiv:2604.xxxxx},
  year={2026}
}
```

---

## Authors

- **Shiyao Li** - Central South University, Changsha, China ([shiyaol492@gmail.com](mailto:shiyaol492@gmail.com))
- **Jiale Zhang** - School of Mechanical Engineering, Hefei University of Technology, Hefei, China ([zhangruoshui2023@163.com](mailto:zhangruoshui2023@163.com))

## License

MIT License. See [LICENSE](LICENSE) for details.
