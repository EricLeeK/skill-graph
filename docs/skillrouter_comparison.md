# SkillRouter vs. SkillGraph: Comparative Analysis

## Project Context

This document compares our approach (SkillGraph) with SkillRouter (Zheng et al., 2026), the current state-of-the-art in skill routing for LLM agents.

**SkillRouter**: https://github.com/zhengyanzhao1997/SkillRouter

---

## Overview

| Aspect | SkillRouter (Zheng et al., 2026) | SkillGraph (Our Work) |
|--------|----------------------------------|----------------------|
| **Core Method** | Retrieve-and-Rerank with dedicated 0.6B models | Dense retrieval + Synthetic Query (SynQ) augmentation |
| **Embedding Model** | SkillRouter-Embedding-0.6B (custom trained) | all-MiniLM-L6-v2 (general purpose, 22M params) |
| **Reranker** | SkillRouter-Reranker-0.6B (custom trained) | None (single-stage retrieval) |
| **Skill Pool** | ~80K open-source skills (51 categories) | 34K UCSB real-world skills |
| **Primary Metric** | Hit@1 = 74.0% | Recall@10 = 72.4% (87 tasks) |
| **Training Cost** | Requires GPU training for 0.6B models | Zero training (uses off-the-shelf model) |
| **Runtime Latency** | Two-stage inference (retrieve + rerank) | Single matrix multiplication |
| **Memory Footprint** | ~2.4GB (FP16) for two 0.6B models | ~80MB for MiniLM embedding |

---

## Key Methodological Differences

### 1. Signal Source

**SkillRouter**: Uses the **complete skill body** as the routing signal. Their core insight is that "the full skill body is the decisive routing signal in large, highly overlapping skill pools."

- First stage: Embed full skill text for retrieval
- Second stage: Rerank top-20 candidates using "flat-full" prompts with complete skill text

**SkillGraph**: Uses **skill description + synthetic queries** as the signal.

- Offline: Use LLM to generate 10 diverse user queries per skill
- Runtime: Max(skill_embedding, best_syn_embedding) for retrieval
- Key insight: Bridge the query-skill semantic gap by precomputing user perspectives

### 2. Model Approach

**SkillRouter**: Train dedicated models on skill routing data

- Pros: Higher accuracy, optimized for the specific task
- Cons: Requires training data, GPU resources, ongoing model maintenance

**SkillGraph**: Use general-purpose model with index augmentation

- Pros: No training required, works with any embedding model
- Cons: Accuracy ceiling limited by general model quality

### 3. Cost Structure

| Cost Type | SkillRouter | SkillGraph |
|-----------|-------------|------------|
| Training | High (0.6B model training) | Zero |
| Index Build | Moderate (embedding 80K skills) | High (LLM generates 343K syn queries) |
| Runtime/Query | Moderate (two 0.6B model inferences) | Very Low (single vector dot product) |
| Maintenance | High (retrain model when skills change) | Low (regenerate SynQ for new skills) |

---

## Performance Analysis

### Metric Strictness

**Important**: The metrics are NOT directly comparable.

- **Hit@1** (SkillRouter): Correct skill must be rank #1. Much stricter.
- **Recall@10** (SkillGraph): Correct skill must be in top-10. More forgiving.

Estimated conversion:
- If SkillRouter achieves 74% Hit@1 on 80K pool, their Recall@10 is likely >90%
- Our 72.4% Recall@10 on 34K pool with a 22M model is substantially different territory

### Scale Factor

- SkillRouter: 80K skills (2.3x larger pool)
- SkillGraph: 34K skills

Larger pool makes retrieval harder (more candidates to distinguish).

### Evaluation Scenarios

- SkillRouter: 75 expert queries (24 single-skill, 51 **multi-skill**)
- SkillGraph: 87 SkillsBench tasks (primarily **single-skill**)

SkillRouter's multi-skill evaluation is more challenging and representative of real-world agent scenarios.

---

## Our Unique Contribution

Despite SkillRouter's superior raw performance, our approach has distinct value:

### 1. Zero-Training-Cost Retrieval

> **"Achieve comparable recall without training any dedicated model."**

SkillRouter requires training two 0.6B parameter models. Our method works with any off-the-shelf embedding model.

### 2. Offline Intelligence / Zero-Runtime-Token

> **"All LLM computation happens offline; runtime is pure vector arithmetic."**

SkillRouter's reranker consumes tokens at runtime (feeding full skill body to 0.6B model). Our SynQ generation is fully offline.

### 3. Speed Advantage

| Metric | SkillRouter (est.) | SkillGraph |
|--------|-------------------|------------|
| Latency | ~100-500ms (two model passes) | ~17ms (matrix ops) |
| Speedup | 1x | **6-30x faster** |

For latency-sensitive applications (real-time agent routing), this is significant.

### 4. Complementary Insight

Both approaches independently confirm the same insight:

> **"Skill name + description alone are insufficient for accurate routing at scale."**

- SkillRouter solves this by using the **full skill body** at runtime (reranker)
- We solve this by **expanding the index offline** with synthetic queries

These are **complementary** strategies, not competing ones.

---

## Recommended Positioning for Paper

### Story 1: Different Trade-off Point

> "While SkillRouter (Zheng et al., 2026) achieves superior Hit@1 through dedicated 0.6B models, we explore the opposite end of the design space: **maximizing recall per unit cost**. Our approach achieves 72.4% Recall@10 using only a 22M general-purpose model with zero training overhead, making it suitable for resource-constrained deployments."

### Story 2: SynQ as "Precomputed Reranking"

> "SkillRouter's reranker uses full skill body at runtime to overcome the limitation of name/description embeddings. Our Synthetic Query method achieves a similar effect **offline**: by having an LLM generate diverse user queries per skill, we effectively precompute the 'full body relevance' into the index itself. This trades index size for runtime speed."

### Story 3: Multi-Skill Extension

SkillRouter's evaluation includes 51 multi-skill queries (68% of benchmark). We have NOT evaluated multi-skill scenarios. This is a **critical gap**.

**Recommendation**: Add multi-skill evaluation before submission.

---

## Action Items

### High Priority
1. **Multi-skill evaluation**: Test on queries requiring multiple skills (like SkillRouter's 51 multi-skill cases)
2. **SynQ quantity ablation**: Test 10 -> 5 -> 3 -> 1 syn queries per skill to find optimal recall/latency trade-off
3. **Scale test**: Evaluate on larger skill pools (e.g., merge with more UCSB skills to reach 50K+)

### Medium Priority
4. **SynQ compression**: Try PCA/compression on 10 syn embeddings per skill to reduce index size
5. **Different aggregation**: Test max vs mean vs learned weighting for combining skill_sim and syn_sim

### Low Priority
6. **Hybrid with keyword**: Re-evaluate auto-generated keyword matching with proper leakage prevention
7. **Query difficulty analysis**: Categorize benchmark queries by difficulty and analyze failure modes

---

## Conclusion

**SkillRouter does not invalidate our work.** It sets a new accuracy bar using dedicated models, but our approach occupies a distinct niche:

- **SkillRouter**: Maximum accuracy, accepts training cost and runtime overhead
- **SkillGraph**: Good accuracy, zero training cost, minimal runtime overhead

The key question for our paper becomes: **"What is the best recall achievable without training any dedicated model?"** This is a well-defined research question with clear practical value.
