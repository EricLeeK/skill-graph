# SkillGraph 技术发展路线图 v2.0

> **定位**: 大规模 Agent Skill 检索的基础设施层
> **核心范式**: Preprocessing Intelligence → Zero-Token Query-Time Efficiency
> **目标场景**: 34K+ Skills 的毫秒级检索，供 MCP Client / Claude Code / LangChain 等生态集成

---

## 0. 现状诊断

### 0.1 当前系统能力

| 组件 | Recall@10 | 延迟 | Token/Query |
|------|-----------|------|-------------|
| Dense (纯向量) | 62.8% ± 12.7% | 17ms | 0 |
| SACR (稀疏路由) | 59.4% ± 9.2% | 10ms | 0 |
| Dense + SynQ | 72.1% ± 16.2% | 17.5ms | 0 |
| SACR + SynQ | 72.1% ± 16.2% | 26.5ms | 0 |

### 0.2 核心瓶颈

1. **Query-Skill 分布 Gap**: 用户 query 的词汇分布与 skill name/description 的分布不匹配
2. **Keyword Matching 效果不佳**: Auto-generated keyword map 召回率较低，手动维护不可扩展
3. **图结构未利用**: `related_skill_ids` 仅存储，从未参与检索
4. **社区质量受限**: HSC 用 KMeans 聚类，未利用 skill 间已知关系
5. **无增量能力**: 新增 skill 需重建整个 HSC

### 0.3 对标基准

| 系统 | Recall@10 | 延迟 | Token/Query | 规模 |
|------|-----------|------|-------------|------|
| UCSB Agentic | 68.3% | ~3s | ~5000 | 34K |
| Claude Code ToolSearch | 未公开 | 毫秒 | 有 | 受 context 限制 |
| MCP `tools/list` | N/A | 毫秒 | 全量加载 | 受 context 限制 |
| **目标: SkillGraph v2** | **> 75%** | **< 10ms** | **0** | **34K+** |

---

## 1. 架构总览: LLM-for-Index + Sparse Attention + Graph RAG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           预处理阶段 (Preprocessing)                          │
│                        允许任意 Token 消耗，一次性执行                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌───────────────┐ │
│  │ 1.1 Synthetic Query  │    │ 1.2 Graph Relation   │    │ 1.3 Community │ │
│  │    Generation        │    │    Learning          │    │    Building   │ │
│  │                      │    │                      │    │               │ │
│  │ LLM 为每个 skill     │    │ LLM 判断 skill 间    │    │ 图聚类替代    │ │
│  │ 生成用户可能查询     │    │ 功能关系(依赖/互补)  │    │ KMeans        │ │
│  │                      │    │                      │    │               │ │
│  │ 输出: N 个 synthetic │    │ 输出: 有向关系图     │    │ 输出: 社区树  │ │
│  │       queries/skill  │    │       (带边类型)     │    │       + 层次  │ │
│  └──────────┬───────────┘    └──────────┬───────────┘    └───────┬───────┘ │
│             │                           │                       │         │
│             └───────────────────────────┼───────────────────────┘         │
│                                         ↓                                 │
│                           ┌─────────────────────────┐                     │
│                           │ 1.4 Multi-Vector Index  │                     │
│                           │                         │                     │
│                           │  skill_vector            │                     │
│                           │  synthetic_query_vectors │                     │
│                           │  tag_vectors             │                     │
│                           │  community_vectors       │                     │
│                           │                         │                     │
│                           │  输出: 持久化索引文件    │                     │
│                           └─────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           运行时阶段 (Runtime)                                │
│                          零 Token，毫秒级延迟                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query                                                                 │
│      ↓                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2.1 Query Embedding (MiniLM, 本地)                                   │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2.2 Sparse Attention Community Routing (SACR v2)                     │   │
│  │                                                                      │   │
│  │  - 图注意力增强: 在社区内聚合邻居信息                                 │   │
│  │  - 多向量路由: query 与 skill_emb + syn_emb + comm_emb 分别计算      │   │
│  │  - 输出: Candidate Communities (缩减到 ~5% 的 skills)                │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2.3 Graph RAG Expansion                                              │   │
│  │                                                                      │   │
│  │  - 种子扩展: 沿 related 边 1-2 跳扩展                                 │   │
│  │  - 关系过滤: 只走 prerequisite / complementary 边                    │   │
│  │  - 输出: Expanded Candidate Set                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2.4 Multi-Vector Reranking                                           │   │
│  │                                                                      │   │
│  │  - 融合多个向量空间的相似度                                           │   │
│  │  - 图 PageRank 重要性加权                                             │   │
│  │  - 输出: Top-k Skills                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: LLM-Augmented Synthetic Query Generation

### 2.1 问题定义

当前 Dense 检索的 query embedding 与 skill embedding 存在分布 gap：
- Skill embedding: `embedding(name + ": " + description)`
- Query embedding: `embedding(user query)`
- Gap: 用户 query 通常是任务描述，skill description 是功能说明，语义空间不同

### 2.2 方法

**输入**: 每个 skill 的 `name`, `description`
**输出**: 每个 skill 的 K 个 synthetic queries + 嵌入

```python
# 预处理（一次性，可消耗 Token）
for skill in skills:
    synthetic_queries = llm_generate(
        system="""You are an expert at understanding software tools and APIs.
        Given a skill's name and description, generate 10-20 different ways
        a user might naturally ask for this skill in an agent task.
        
        Guidelines:
        - Use natural language, not technical jargon
        - Include variations: direct request, implicit need, question form
        - Cover different contexts where this skill would be useful
        - Each query should be 5-30 words
        """,
        user=f"Skill name: {skill.name}\nDescription: {skill.description}\n\nGenerate diverse user queries."
    )
    
    # 嵌入 synthetic queries
    syn_embeddings = embedder.encode(synthetic_queries)
    
    # 存储到索引
    index.add_skill(
        skill_id=skill.id,
        skill_embedding=skill_embedding,
        synthetic_embeddings=syn_embeddings,
        synthetic_queries=synthetic_queries,
    )
```

### 2.3 检索时的多向量匹配

```python
def multi_vector_search(query_embedding, index, top_k=100):
    """Query 与所有向量空间同时匹配，取最佳匹配。"""
    
    # 空间 1: skill 本身的 embedding
    skill_sims = cosine(query_embedding, index.skill_embeddings)
    
    # 空间 2: 每个 synthetic query 的 embedding
    # 对于每个 skill，取与它最相似的 synthetic query 的相似度
    syn_sims = []
    for skill_id, syn_embs in index.synthetic_embeddings.items():
        best_sim = max(cosine(query_embedding, emb) for emb in syn_embs)
        syn_sims[skill_id] = best_sim
    
    # 融合: 取两个空间的最大值（或加权平均）
    final_scores = {
        skill_id: max(skill_sims[skill_id], syn_sims[skill_id])
        for skill_id in index.skill_ids
    }
    
    return top_k(final_scores, k=top_k)
```

### 2.4 预期效果

| 基线 | 改进后 (预估) |
|------|--------------|
| Dense: 62.8% | Dense + SynQ: 70-78% |
| SACR: 59.4% | SACR + SynQ: 68-75% |

**原理**: Synthetic queries 缩小了 query 与 skill 之间的语义 gap，因为 synthetic queries 是用 LLM "翻译"过的用户语言。

### 2.5 资源估算

| 项目 | 数值 |
|------|------|
| 34K skills × 10 queries | 340K LLM calls |
| 平均每个 query 500 tokens | 170M input tokens |
| Claude Haiku 成本 | ~$50-85 |
| Claude Sonnet 成本 | ~$500-850 |
| 嵌入成本 (MiniLM 本地) | $0 |
| 总存储增量 | ~340K × 384dim × 4byte ≈ 522MB |

### 2.6 与现有工作的关系

- **不同于 UCSB**: UCSB 用 LLM 在**查询时**改写 query（每次查询都调用），我们用 LLM 在**预处理时**生成 synthetic queries（一次性）
- **类似于 REALM / DPR 的 query generation**: 但应用于 skill retrieval 场景
- **类似于 BEIR 的 doc2query**: 但应用于非文档场景

---

## 3. Phase 2: Graph RAG 关系增强

### 3.1 问题定义

当前 `related_skill_ids` 是**纯语义相似度**生成的（cosine top-5），未利用技能间的**功能关系**。

但功能关系对检索很重要：
- `docker` 和 `kubernetes` 向量不接近，但功能相关（部署依赖）
- `pdf-parser` 和 `excel-exporter` 向量不接近，但经常一起用（互补）
- `spring-boot` 和 `spring-security` 向量接近，但一个是框架一个是安全模块（子集关系）

### 3.2 方法: LLM-Based 关系图构建

**输入**: 所有 skill pairs（或采样 pairs）
**输出**: 有向关系图，边带有关系类型

```python
# 预处理（一次性）
relation_types = [
    "prerequisite",      # A 是 B 的前置条件（如 docker → kubernetes）
    "complementary",     # A 和 B 经常一起用（如 pdf-parser + excel-exporter）
    "alternative",       # A 和 B 是替代品（如 ffmpeg vs handbrake）
    "subskill",          # A 是 B 的子功能（如 spring-security ⊂ spring-boot）
    "unrelated",
]

for skill_a, skill_b in sampled_pairs:
    relation = llm_judge(
        system="""Judge the functional relationship between two skills.
        Consider: Would a user who needs skill A also likely need skill B?
        Is one a prerequisite for the other? Are they alternatives?""",
        user=f"Skill A: {skill_a.name}\n{skill_a.description}\n\n"
             f"Skill B: {skill_b.name}\n{skill_b.description}"
    )
    
    if relation != "unrelated":
        graph.add_edge(skill_a.id, skill_b.id, relation=relation)
```

**采样策略**（避免 O(N²)）：
1. 先用向量相似度找出 top-100 邻居候选
2. 只在候选对之间做 LLM 判断
3. 采样数量: ~34K × 100 = 3.4M pairs，但只选 top 相似度的

### 3.3 运行时: 图注意力增强

```python
def graph_attention_enhance(query_embedding, seeds, graph, alpha=0.15):
    """Enhance seed skill embeddings with neighbor information."""
    
    enhanced = {}
    for skill_id in seeds:
        # 获取邻居
        neighbors = graph.neighbors(skill_id, 
            relation_filter=["prerequisite", "complementary"])
        
        if not neighbors:
            enhanced[skill_id] = skill_embeddings[skill_id]
            continue
        
        # Attention 权重: query 与邻居的相似度
        neighbor_embs = [skill_embeddings[n] for n in neighbors]
        attn_weights = softmax([
            dot(query_embedding, emb) for emb in neighbor_embs
        ])
        
        # 增强 embedding = 自身 + 邻居加权
        enhanced_emb = skill_embeddings[skill_id] + alpha * sum(
            w * emb for w, emb in zip(attn_weights, neighbor_embs)
        )
        enhanced[skill_id] = enhanced_emb
    
    return enhanced
```

### 3.4 运行时: 种子扩展

```python
def graph_expand(seeds, graph, max_hops=1, 
                 relation_filter=["prerequisite", "complementary"]):
    """Expand seed set along graph edges."""
    
    candidates = set(seeds)
    frontier = set(seeds)
    
    for _ in range(max_hops):
        new_frontier = set()
        for skill_id in frontier:
            for neighbor in graph.neighbors(skill_id):
                if graph[skill_id][neighbor]["relation"] in relation_filter:
                    new_frontier.add(neighbor)
        candidates.update(new_frontier)
        frontier = new_frontier
    
    return candidates
```

### 3.5 图社区替代 KMeans

```python
# 用图算法替代 KMeans 构建 HSC
import networkx as nx

communities = nx.community.louvain_communities(
    graph, 
    weight="similarity",
    resolution=1.0
)

# 社区分层
# Level 0: 大社区（超级社区）
# Level 1: 中等社区（功能组）
# Level 2: 小社区（具体领域）
```

**优势**:
- KMeans 只考虑向量距离，可能把功能无关但向量接近的技能聚在一起
- 图社区考虑实际功能关系，语义更一致
- `docker` 和 `kubernetes` 会被聚到同一社区（通过 prerequisite 边）

### 3.6 预期效果

| 场景 | 无 Graph RAG | 有 Graph RAG |
|------|-------------|-------------|
| 单 GT 任务 | 已找到 | 可能找到更多相关 skills |
| **多 GT 任务**（如 travel-planning 6 skills）| 只找到 2-3 个 | **能找到更多互补 skills** |
| 冷启动 skill | 依赖 embedding | 通过关系被已有 skill 带到 |

### 3.7 资源估算

| 项目 | 数值 |
|------|------|
| 采样 pairs | ~100K-500K |
| 每对 LLM 判断 | ~200 tokens |
| Claude Haiku 成本 | ~$10-50 |
| 图存储 | 邻接表，< 10MB |

---

## 4. Phase 3: 稀疏注意力路由升级 (SACR v2)

### 4.1 当前 SACR 的问题

1. **社区表示**: 用 KMeans 中心点作为 community embedding，质量有限
2. **路由策略**: 纯 top-k 选择，没有考虑社区间关系
3. **注意力计算**: 简单的 scaled dot-product，没有利用图结构

### 4.2 升级方案

#### 4.2.1 社区表示增强

```python
# 当前: community_emb = KMeans center
# 升级: community_emb = 成员 embeddings 的注意力加权平均

def compute_community_embedding(community, query_embedding):
    """Compute community embedding with query-aware attention."""
    
    members = community.skill_ids
    member_embs = [skill_embeddings[m] for m in members]
    
    # Attention 权重: query 与每个成员的相似度
    weights = softmax([
        dot(query_embedding, emb) for emb in member_embs
    ])
    
    # 加权平均
    community_emb = sum(w * emb for w, emb in zip(weights, member_embs))
    return community_emb
```

#### 4.2.2 图注意力路由

```python
class GraphAttentionSACR:
    """SACR with graph attention for community selection."""
    
    def route(self, query_embedding):
        # Level 0: 选 top-k L0 社区（与 SACR v1 相同）
        l0_selected = self._select_l0(query_embedding)
        
        # Level 1: 在 L0 内部，用图注意力选择 L1
        l1_candidates = []
        for l0_comm in l0_selected:
            # 获取 L0 内所有 L1 社区
            l1_comms = l0_comm.children
            
            # 图注意力: L1 社区的表示包含其成员的功能关系
            l1_scores = []
            for l1 in l1_comms:
                # L1 的 embedding = 成员 embedding + 邻居信息
                enhanced_emb = self._graph_attention_enhance(
                    l1, query_embedding
                )
                sim = cosine(query_embedding, enhanced_emb)
                l1_scores.append((l1, sim))
            
            # 选 top-k L1
            top_l1 = sorted(l1_scores, key=lambda x: x[1], reverse=True)[:self.top_k_l1]
            l1_candidates.extend([l1 for l1, _ in top_l1])
        
        # Level 2: 同理
        l2_selected = self._select_l2(l1_candidates, query_embedding)
        
        # 收集所有 skills
        skill_ids = []
        for l2 in l2_selected:
            skill_ids.extend(l2.skill_ids)
        
        return skill_ids
```

#### 4.2.3 社区关系图

```python
# 构建社区之间的关系图
community_graph = nx.DiGraph()

for comm_a in communities:
    for comm_b in communities:
        if comm_a.level != comm_b.level:
            continue
        
        # 计算两个社区之间的边密度
        edges_between = count_edges_between(comm_a, comm_b, skill_graph)
        density = edges_between / (len(comm_a) * len(comm_b))
        
        if density > threshold:
            community_graph.add_edge(comm_a.id, comm_b.id, weight=density)
```

### 4.3 预期效果

| 指标 | SACR v1 | SACR v2 (预估) |
|------|---------|---------------|
| Recall@10 | 59.4% | 62-68% |
| Candidate set quality | 纯语义 | 语义 + 功能关系 |
| 路由准确性 | 中等 | 提升 |

---

## 5. 三阶段融合后的完整检索流程

```python
def retrieve_v2(query: str, top_k: int = 10) -> List[Skill]:
    """Complete retrieval pipeline with all three phases."""
    
    # 1. Query embedding (零 Token)
    q_emb = embedder.encode(query)
    
    # 2. SACR v2: Graph Attention Community Routing
    candidate_communities = sacr_v2.route(q_emb)
    seed_skills = extract_skills_from_communities(candidate_communities)
    
    # 3. Graph RAG: Seed Expansion
    expanded_skills = graph_expand(
        seed_skills, 
        skill_graph,
        max_hops=1,
        relation_filter=["prerequisite", "complementary"]
    )
    
    # 4. Graph Attention: Enhance embeddings
    enhanced_embs = graph_attention_enhance(q_emb, expanded_skills, skill_graph)
    
    # 5. Multi-Vector Reranking
    scores = {}
    for skill_id in expanded_skills:
        # 5.1 Skill vector similarity
        skill_sim = cosine(q_emb, enhanced_embs[skill_id])
        
        # 5.2 Synthetic query similarity
        syn_sims = [cosine(q_emb, emb) for emb in index.synthetic_embeddings[skill_id]]
        best_syn_sim = max(syn_sims) if syn_sims else 0
        
        # 5.3 Graph PageRank importance
        pagerank_score = graph_pagerank[skill_id]
        
        # 5.4 融合
        scores[skill_id] = (
            0.5 * skill_sim + 
            0.4 * best_syn_sim + 
            0.1 * pagerank_score
        )
    
    # 6. Return top-k
    top_skill_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [skill_index[id] for id in top_skill_ids]
```

---

## 6. 评估方法论

### 6.1 核心原则

- **所有超参数通过交叉验证确定**，在训练 fold 上搜索，在测试 fold 上报告
- **Keyword map（如果有）必须从技能库本身自动生成**，不查看 benchmark 任务内容
- **所有报告结果附带 5-Fold CV 的测试集均值 ± 标准差**

### 6.2 评估指标

| 指标 | 定义 | 为什么重要 |
|------|------|-----------|
| **Recall@10** | top-10 中命中至少一个 GT skill | SkillsBench 标准 |
| **Recall@5** | top-5 中命中至少一个 GT skill | 更严格 |
| **Average Precision@10** | 考虑排名的精确度 | 比 Recall 更严格 |
| **nDCG@10** | 考虑多 GT skill 的排序质量 | 多 GT 任务的公平性 |
| **Latency (ms)** | 单次检索延迟 | 速度是核心卖点 |
| **Index Build Time** | 预处理阶段耗时 | 可接受性 |
| **Index Size (MB)** | 持久化索引大小 | 存储成本 |

### 6.3 对比基线

| 基线 | 说明 |
|------|------|
| Dense (纯向量) | 当前实现的 dense retrieval |
| SACR v1 | 当前实现的稀疏路由 |
| UCSB Agentic | 论文方法，~68.3% |
| BM25 | 传统词频检索 |
| FAISS (IVF) | 工业级近似最近邻 |

### 6.4 消融实验设计

```
Full System: SACR v2 + Graph RAG + SynQ
    ↓
- Graph RAG: SACR v2 + SynQ (无图扩展)
    ↓
- SynQ: SACR v2 (无 synthetic queries)
    ↓
- SACR v2 → SACR v1: 纯 SACR v1
    ↓
- SACR → Dense: 纯 Dense
```

---

## 7. 论文叙事框架

### 7.1 核心故事

> **"预处理阶段的智能投资可以换取运行时的零 Token 高效率"**

当前 Agent 系统的两难：
- **LLM-based 方法**（UCSB）：高精度但慢（秒级）且贵（每次查询耗 Token）
- **传统检索方法**（Dense/SACR）：快（毫秒级）且便宜（零 Token）但精度不够

我们的 insight：
- **把 LLM 的智能从"查询时"移到"预处理时"**
- 预处理阶段：用 LLM 深度理解 skills，构建智能索引（一次性，可接受成本）
- 运行时阶段：零 Token，毫秒级，基于构建好的索引检索

### 7.2 技术贡献

| 贡献 | 说明 |
|------|------|
| **Synthetic Query Generation for Skill Retrieval** | 首次将 doc2query 范式应用到 skill retrieval |
| **Graph Attention Community Routing** | 结合图结构和稀疏注意力的层次路由 |
| **LLM-Based Relation Graph for Skills** | 用 LLM 构建 skill 功能关系图 |
| **Zero-Token Retrieval at Scale** | 在 34K skills 上实现毫秒级零 Token 检索 |

### 7.3 与 UCSB 的关系

不是竞争，是**互补**：
- UCSB: 适合"精度优先，延迟不敏感"的场景（离线批处理）
- 我们: 适合"实时响应，高频调用"的场景（在线服务）

类比：
- UCSB 像 Google Search（重排序，多轮交互）
- 我们像 Elasticsearch（快速初筛，基础设施）

---

## 8. 风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| Synthetic queries 提升有限 | 中 | 高 | 尝试不同 prompt 策略；增加 query 数量；尝试 query 聚类 |
| Graph RAG 对单 GT 任务帮助小 | 高 | 中 | 聚焦多 GT 任务的评估；作为辅助组件而非核心 |
| 预处理成本过高 | 低 | 中 | 使用 Claude Haiku 替代 Sonnet；增量更新避免全量重建 |
| 5-Fold CV 结果仍 < 68.3% | 中 | 极高 | 这是 project killer，必须提前做 pilot 验证 |
| 图社区构建太慢 | 中 | 中 | Louvain 算法是 O(N log N)，34K 应该很快；可采样加速 |

---

## 9. 时间线

| 阶段 | 时间 | 产出 |
|------|------|------|
| **Pilot** | 1-2 天 | 1000 skills synthetic queries + 评估 |
| **Phase 1** | 1 周 | 完整 34K synthetic query index + 评估 |
| **Phase 2** | 1 周 | Graph relation learning + Graph RAG + 评估 |
| **Phase 3** | 1 周 | SACR v2 + 三阶段融合 + 完整评估 |
| **论文写作** | 2-3 周 | 初稿 → 迭代 → 投稿 |
| **投稿** | 目标: 2026年Q2 | NeurIPS / ICML / ICLR |

---

## 10. 附录: 关键术语

| 术语 | 定义 |
|------|------|
| **SACR** | Sparse Attention Community Routing，稀疏注意力社区路由 |
| **Graph RAG** | Retrieval-Augmented Generation using graph structure |
| **Synthetic Query** | 用 LLM 生成的、模拟用户可能查询的文本 |
| **Multi-Vector Index** | 每个 skill 对应多个向量（skill vector, syn query vectors, tag vectors） |
| **Graph Attention** | 用 attention 机制聚合图邻居信息 |
| **HSC** | Hierarchical Skill Communities，层次化技能社区 |
| **SkillsBench** | UCSB 提出的 87-task skill retrieval benchmark |
| **MCP** | Model Context Protocol，Anthropic 主导的工具连接标准 |
