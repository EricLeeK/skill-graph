# SkillGraph

Sparse Attention-based Skill Retrieval for LLM Agents.

## Overview

SkillGraph is a universal skill retrieval module designed to plug into any LLM-based agent framework (ReAct, LangChain, AutoGPT, etc.). It combines:

- **SKG (Skill Knowledge Graph)**: ~34,000 real-world skills from UCSB Skill-Usage with embeddings and relationships
- **HSC (Hierarchical Skill Communities)**: 3-level tree (8 super / 64 functional / 512 leaf)
- **SACR (Sparse Attention Community Routing)**: Query-aware dynamic top-k community filtering
- **SRE (Skill Refinement Engine)**: Query-specific skill prompt optimization
- **Synthetic Query Index**: LLM-generated user queries per skill to bridge the query-skill semantic gap

## Quick Start

```bash
# Install
pip install -e .

# Or with uv
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run evaluation
python evals/eval_multi_vector.py
```

## Python API

```python
from skill_graph import SkillGraph

sg = SkillGraph()
result = sg.retrieve("Deploy a microservice to Kubernetes", top_k=5)
for skill in result.skills:
    print(skill.name, skill.description)
```

## FastAPI Server

```bash
uvicorn skill_graph.api.server:app --reload
```

## Integrations

- **ReAct**: [`examples/react_integration.py`](examples/react_integration.py)
- **LangChain**: [`examples/langchain_integration.py`](examples/langchain_integration.py)

## Evaluation Results

### SkillsBench (87 tasks on 34,396 UCSB skills)

Benchmark: 87 real-world agent tasks from [SkillsBench](https://github.com/UCSB-NLP-Chang/Skill-Usage/tree/main/skillsbench/curated), using task instructions as queries and required skills as ground truth. 92.7% of required skills (177/191) are present in the UCSB dataset.

| System | Recall@10 | Latency | Token/Query |
|--------|-----------|---------|-------------|
| **Dense + SynQ (ours)** | **72.1%** | **17.5 ms** | **0** |
| Dense (ours) | 62.8% | 2.8 ms | 0 |
| SACR (ours) | 59.4% | 10.0 ms | 0 |
| SACR + SynQ (ours) | 72.1% | 26.5 ms | 0 |
| UCSB Agentic Hybrid | 68.3% | ~seconds | ~5000 |

**Key findings**:
- Synthetic Query Generation bridges the query-skill semantic gap, providing **+9.3pp** improvement over dense baseline
- Dense + SynQ (72.1%) exceeds UCSB Agentic (68.3%) while being **~170x faster** and **zero token** at runtime
- SACR maintains **3.5x speedup** over dense while preserving **95%** of its recall
- All components are built during preprocessing; runtime is fully zero-token

## Project Structure

```
skill_graph/
├── api/           # FastAPI + Python SDK
├── core/          # SACR, SRE, HSC, Graph
├── data/          # Skill generation pipeline
evals/             # Benchmark + evaluation
examples/          # ReAct + LangChain demos
tests/             # Unit tests
```
