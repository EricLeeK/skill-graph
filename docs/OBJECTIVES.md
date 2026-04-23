# SkillGraph 下一阶段目标文档

> **版本**: 1.0
> **日期**: 2026-04-22
> **约束**: 本文档只定义"目标"与"完成标准"，不包含任何实现路线或方法论指导。

---

## O1: Pilot 验证 — Synthetic Query Generation 的有效性

### O1.1 生成 1000 个技能的 Synthetic Queries

**完成标准**:
- [ ] 从 34K 技能库中**随机采样** 1000 个技能（seed=42，可复现）
- [ ] 每个技能生成 **10 个** synthetic queries
- [ ] 所有 synthetic queries 保存到 `data/pilot/synthetic_queries_1k.jsonl`
- [ ] 文件格式: 每行一个 JSON，包含 `skill_id`, `skill_name`, `synthetic_queries: List[str]`
- [ ] 生成日志保存到 `data/pilot/generation_log.json`，包含: 总调用次数、总 token 数、总耗时、失败次数

**可评判标准**:
- 文件 `data/pilot/synthetic_queries_1k.jsonl` 存在且行数 = 1000
- 每个记录的 `synthetic_queries` 列表长度 = 10
- 无空字符串或长度 < 5 个字符的 query
- 生成日志中失败率 < 5%

---

### O1.2 评估 Pilot 效果

**完成标准**:
- [ ] 用 1000 个技能的 synthetic queries 构建 **multi-vector index**
- [ ] 在 SkillsBench 全部 87 个任务上运行 Dense 检索（仅用这 1000 个技能作为候选池）
- [ ] 同时运行 Baseline: Dense 检索（不用 synthetic queries，仅用 skill embedding）
- [ ] 记录每个任务的 Recall@10、Latency

**可评判标准**:
- Baseline (无 SynQ) Recall@10 在 1000-skill 子集上可复现（预期 ~55-65%）
- SynQ 版本 Recall@10 **绝对值提升 ≥ 5 个百分点**（即 ≥ Baseline + 5%）
- 结果保存到 `data/pilot/results_pilot.json`，格式:
  ```json
  {
    "baseline": {"recall_at_10": 0.XX, "avg_latency_ms": X.XX},
    "with_synthetic_queries": {"recall_at_10": 0.XX, "avg_latency_ms": X.XX},
    "improvement_pp": X.X,
    "n_skills": 1000,
    "n_queries_per_skill": 10
  }
  ```

**决策点**: 如果 improvement < 5pp，O2 不启动。

---

## O2: 完整 Synthetic Query Index 构建

### O2.1 全量 34K 技能 Synthetic Query 生成

**完成标准**:
- [ ] 全部 **34,396** 个 UCSB 技能均生成 10 个 synthetic queries
- [ ] 输出文件 `data/synthetic_queries_34k.jsonl`（34,396 行）
- [ ] 生成日志 `data/synthetic_generation_log.json`，包含:
  - 总调用次数: 34,396
  - 总 input tokens
  - 总 output tokens
  - 总耗时（分钟）
  - 失败次数及 skill_id 列表
  - 使用的 LLM 模型名称和版本

**可评判标准**:
- `data/synthetic_queries_34k.jsonl` 行数 = 34,396
- 每个记录 `synthetic_queries` 列表长度 = 10
- 失败率 < 3%（即失败 < 1032 个）
- 如果失败 > 0，必须有 `data/synthetic_retry_log.json` 记录重试结果

---

### O2.2 Multi-Vector Index 构建

**完成标准**:
- [ ] 为所有 synthetic queries 计算 embedding（MiniLM-L6-v2）
- [ ] 构建持久化索引文件 `data/index/multi_vector_index.pkl` 或 `.npy` 格式
- [ ] 索引包含:
  - `skill_ids`: List[str], 长度 34,396
  - `skill_embeddings`: np.ndarray, shape (34396, 384)
  - `synthetic_embeddings`: Dict[str, np.ndarray], 每个 skill 对应 (10, 384)
  - `metadata`: 构建时间、模型版本、数据版本

**可评判标准**:
- 索引文件可加载，加载后 `len(skill_ids) == 34396`
- 每个 skill 的 synthetic embeddings shape = (10, 384)
- 索引文件大小 < 1GB（预期 ~600MB）
- 加载时间 < 30 秒

---

### O2.3 全量评估

**完成标准**:
- [ ] 在 SkillsBench 87 任务上，用 **5-Fold 交叉验证** 评估 Dense + Synthetic Query 检索
- [ ] 每折用 80% 任务调参（如果有），20% 测试
- [ ] 记录每折的 Recall@10、Latency
- [ ] 计算并报告: mean ± std

**可评判标准**:
- 结果保存到 `data/results_dense_synq_cv.json`
- 格式包含:
  ```json
  {
    "n_folds": 5,
    "per_fold": [
      {"fold": 1, "recall_at_10": 0.XX, "n_hits": X, "avg_latency_ms": X.XX},
      ...
    ],
    "summary": {
      "mean_recall_at_10": 0.XX,
      "std_recall_at_10": 0.XX,
      "mean_latency_ms": X.XX
    }
  }
  ```
- **Mean Recall@10 ≥ 70%**

---

## O3: Graph RAG 模块

### O3.1 Skill 关系图构建

**完成标准**:
- [ ] 构建有向图 `graph`，节点为 34,396 个 skill，边为功能关系
- [ ] 边类型属于以下之一: `prerequisite`, `complementary`, `alternative`, `subskill`
- [ ] 图持久化到 `data/graph/skill_relation_graph.json`
- [ ] 图统计信息保存到 `data/graph/graph_stats.json`:
  ```json
  {
    "n_nodes": 34396,
    "n_edges": X,
    "edge_type_distribution": {
      "prerequisite": X,
      "complementary": X,
      "alternative": X,
      "subskill": X
    },
    "avg_degree": X.XX,
    "max_degree": X,
    "connected_components": X
  }
  ```

**可评判标准**:
- 图文件可加载，`n_nodes == 34396`
- 边的类型只能是上述 4 种之一
- 图是连通的（connected_components = 1）或有明确的最大连通分量说明

---

### O3.2 图社区构建（替代 KMeans）

**完成标准**:
- [ ] 用图算法（Louvain 或类似）构建层次化社区
- [ ] 社区树持久化到 `data/graph/hierarchical_communities.json`
- [ ] 每个社区包含: `id`, `level`, `skill_ids`, `parent_id`, `child_ids`
- [ ] 社区统计:
  ```json
  {
    "n_level0": X,
    "n_level1": X,
    "n_level2": X,
    "avg_community_size": {
      "level0": X.X,
      "level1": X.X,
      "level2": X.X
    }
  }
  ```

**可评判标准**:
- 所有 34,396 个 skills 都被分配到某个 L2 社区（无遗漏）
- 社区层次结构是无环树（验证 parent-child 关系无循环）

---

### O3.3 Graph RAG 检索评估

**完成标准**:
- [ ] 在 SkillsBench 87 任务上，用 5-Fold CV 评估 **Dense + SynQ + Graph RAG**
- [ ] Graph RAG 配置:
  - 种子扩展: 沿 `prerequisite` 和 `complementary` 边 1 跳
  - 图注意力: alpha = 0.15（可调整）
- [ ] 同时运行消融: 无 Graph RAG（Dense + SynQ only）

**可评判标准**:
- 结果保存到 `data/results_graph_rag_cv.json`
- **Graph RAG 版本 Mean Recall@10 ≥ Dense + SynQ 版本**
- 记录 Graph RAG 带来的候选集扩展比例（expanded / seeds）

---

## O4: SACR v2 升级

### O4.1 图注意力社区路由实现

**完成标准**:
- [ ] 实现 `SparseAttentionCommunityRouterV2` 类
- [ ] 社区表示使用图注意力增强（非 KMeans 中心点）
- [ ] 支持 multi-vector 路由（skill_emb + syn_emb）
- [ ] 代码位于 `skill_graph/core/sacr_v2.py`

**可评判标准**:
- 单元测试通过: `pytest tests/test_sacr_v2.py -v`
- 测试覆盖:
  - 初始化加载
  - 单次路由返回 skill_ids
  - 路由结果非空
  - 候选集大小 < 全量的 10%

---

### O4.2 SACR v2 评估

**完成标准**:
- [ ] 在 SkillsBench 87 任务上，用 5-Fold CV 评估 SACR v2
- [ ] 配置: 与 SACR v1 相同的参数搜索空间（12 个配置）
- [ ] 同时运行 SACR v1 作为 baseline（相同 CV 流程）

**可评判标准**:
- 结果保存到 `data/results_sacr_v2_cv.json`
- **SACR v2 Mean Recall@10 ≥ SACR v1 Mean Recall@10**
- **SACR v2 Latency ≤ SACR v1 Latency × 1.2**（不显著变慢）

---

## O5: 完整系统融合评估

### O5.1 三阶段融合系统

**完成标准**:
- [ ] 实现完整检索流程: SACR v2 → Graph RAG 扩展 → Multi-Vector Reranking
- [ ] 支持配置开关: `use_sacr`, `use_graph_rag`, `use_synthetic_queries`
- [ ] API 接口: `SkillGraph.retrieve(query, config)`

---

### O5.2 完整消融实验

**完成标准**:
- [ ] 在 SkillsBench 87 任务上，用 5-Fold CV 运行以下配置:
  1. Dense（纯向量，baseline）
  2. Dense + SynQ
  3. Dense + SynQ + Graph RAG
  4. SACR v1
  5. SACR v2
  6. SACR v2 + SynQ
  7. SACR v2 + SynQ + Graph RAG（完整系统）
- [ ] 每个配置记录: Recall@10, Recall@5, AP@10, Latency

**可评判标准**:
- 结果保存到 `data/results_full_ablation_cv.json`
- **完整系统 Mean Recall@10 ≥ 75%**
- **完整系统 Mean Latency ≤ 15ms**
- **完整系统 Mean Recall@10 > UCSB Agentic (68.3%)**
- 消融表格可生成，显示每个组件的独立贡献

---

## O6: 公平对比与基准复现

### O6.1 UCSB Agentic 方法复现

**完成标准**:
- [ ] 在 SkillsBench 87 任务上复现 UCSB Agentic Hybrid 方法
- [ ] 使用相同的 34K 技能库和相同的评估流程
- [ ] 记录 Recall@10 和 Latency

**可评判标准**:
- 复现结果保存到 `data/results_ucsb_reproduction.json`
- 复现的 Recall@10 在 65-72% 范围内（论文报告 68.3%，允许 ±3pp 误差）
- 如果复现失败（偏差 > 5pp），记录差异分析

---

### O6.2 公平对比表格

**完成标准**:
- [ ] 生成对比表格，包含以下系统:
  - UCSB Agentic
  - Dense ( ours )
  - SACR v1 ( ours )
  - SACR v2 + SynQ + Graph RAG ( ours, full )
- [ ] 每个系统报告: Recall@10 (mean ± std), Latency, Token/Query
- [ ] 所有结果均来自 **5-Fold CV 测试集**（非训练集调优结果）

**可评判标准**:
- 表格保存到 `docs/comparison_table.md`
- 我们的完整系统在 Recall@10 上 **严格优于** UCSB Agentic（非相等）
- 我们的完整系统在 Latency 上 **严格优于** UCSB Agentic（至少 10x faster）

---

## O7: 代码与测试

### O7.1 代码质量

**完成标准**:
- [ ] 所有新代码通过 `pytest`（现有 11 个测试 + 新增测试全部通过）
- [ ] 新增测试覆盖:
  - Synthetic query generation pipeline
  - Multi-vector index loading and search
  - Graph RAG expansion
  - SACR v2 routing
  - Full system integration
- [ ] 新增测试数量 ≥ 10 个

**可评判标准**:
- `.venv/bin/python3 -m pytest tests/ -v` 全部通过
- 测试覆盖率（行覆盖率）≥ 80%（新代码）

---

### O7.2 文档

**完成标准**:
- [ ] `README.md` 更新，包含:
  - 评估方法论说明
  - 5-Fold CV 结果
  - 与 UCSB 的公平对比
  - 组件级消融结果
- [ ] `docs/METHODOLOGY.md`: 详细的方法论文档（给审稿人看的）
- [ ] `docs/LIMITATIONS.md`: 诚实的局限性说明

**可评判标准**:
- README 中包含明确的"评估方法论"章节
- 所有报告的数字均标注"5-Fold CV"或标准差

---

## O8: 论文写作

### O8.1 论文初稿

**完成标准**:
- [ ] 完整论文初稿，包含以下章节:
  - Abstract
  - Introduction（问题定义 + 核心 insight）
  - Related Work（UCSB + 其他 tool learning 工作）
  - Method（三阶段方法，每阶段有公式/算法伪代码）
  - Experiments（数据集、评估指标、对比基线、消融实验）
  - Results（表格 + 图表）
  - Discussion（局限性 + 未来工作）
  - Conclusion
- [ ] 论文长度: 8-10 页（NeurIPS/ICML 格式）
- [ ] 图表数量: ≥ 4 个（含消融柱状图、延迟对比图、召回率曲线）

**可评判标准**:
- 文件 `paper/main.tex` 或 `paper/draft.md` 存在
- 包含所有上述章节
- 所有实验数字可追溯（对应 `data/results_*.json` 文件）

---

### O8.2 论文迭代

**完成标准**:
- [ ] 内部 review: 至少 2 轮 self-review
- [ ] 外部 review: 至少 1 位同行（非作者）阅读并给出反馈
- [ ] 修改记录保存到 `paper/revision_log.md`

**可评判标准**:
- `paper/revision_log.md` 包含至少 3 条修改记录
- 每条记录包含: 问题描述、修改位置、修改前、修改后

---

## O9: 投稿准备

### O9.1 目标会议选择

**候选会议**（按优先级）:
1. NeurIPS 2026（截稿: ~2026年5月）
2. ICML 2026（截稿: ~2026年1月/2月，可能已错过）
3. ICLR 2027（截稿: ~2026年9月）
4. ACL 2026（截稿: ~2026年7月）
5. EMNLP 2026（截稿: ~2026年6月）

**完成标准**:
- [ ] 确定目标会议
- [ ] 确认截稿日期
- [ ] 确认格式要求（LaTeX 模板）

---

### O9.2 投稿材料

**完成标准**:
- [ ] 论文 PDF（符合目标会议格式）
- [ ] Supplementary Material（包含完整实验细节、代码链接）
- [ ] Code Release: GitHub repo 公开，包含:
  - 完整代码
  - 预处理脚本
  - 评估脚本
  - README 使用说明
- [ ] Data Release: 预处理后的索引文件（如允许）

**可评判标准**:
- GitHub repo 可 clone
- `pip install -e .` 成功
- `pytest tests/` 通过
- `python evals/eval_cross_validation.py` 可复现论文结果（±2pp 误差）

---

## 目标优先级与依赖关系

```
O1 (Pilot) ──→ 决策点 ──→ O2 (全量 SynQ) ──→ O3 (Graph RAG) ──→ O4 (SACR v2)
   │              │              │                    │                │
   │              │              └────────────────────┴────────────────┘
   │              │                                     │
   │              └───────── 如果 O1 improvement < 5pp，停止 ──────┘
   │                                                    │
   └────────────────────────────────────────────────────┘
                                                        ↓
                                                  O5 (完整评估)
                                                        ↓
                                                  O6 (公平对比)
                                                        ↓
                                                  O7 (代码+测试)
                                                        ↓
                                                  O8 (论文写作)
                                                        ↓
                                                  O9 (投稿)
```

| 目标 | 优先级 | 阻塞条件 | 预期耗时 |
|------|--------|---------|---------|
| O1 | P0 | 无 | 1-2 天 |
| O2 | P0 | O1 通过决策点 | 3-5 天 |
| O3 | P1 | O2 完成 | 5-7 天 |
| O4 | P1 | O2 完成 | 3-5 天 |
| O5 | P0 | O3 + O4 完成 | 2-3 天 |
| O6 | P0 | O5 完成 | 1-2 天 |
| O7 | P1 | O5 完成 | 2-3 天 |
| O8 | P1 | O6 + O7 完成 | 10-14 天 |
| O9 | P2 | O8 完成 | 3-5 天 |

---

## 关键决策点

| 决策点 | 条件 | 通过标准 | 失败标准 |
|--------|------|---------|---------|
| **DP1: Pilot 通过?** | O1.2 完成 | SynQ 提升 ≥ 5pp | SynQ 提升 < 5pp → **项目终止或 pivot** |
| **DP2: 全量 SynQ 有效?** | O2.3 完成 | Mean Recall@10 ≥ 70% | < 70% → 需改进方法 |
| **DP3: Graph RAG 有效?** | O3.3 完成 | Graph RAG ≥ 无 Graph RAG | < → Graph RAG 降级为可选组件 |
| **DP4: 完整系统达标?** | O5.2 完成 | Mean Recall@10 ≥ 75% 且 > 68.3% | 不达标 → **不投稿** |

---

## 评估方法论规范

### 数据使用原则

1. **系统配置可目标集校准**：SACR 的 `top_k_level0/1/2` 是系统配置（类似 FAISS 的 `nprobe`），可在目标数据集上校准。
2. **所有报告指标附带方差**：使用 5-Fold CV 报告 mean ± std。
3. **消融实验完整**：报告所有组件组合的独立贡献。
4. **对比公平**：与 UCSB 的对比基于相同 benchmark、相同指标定义。

---

## 完成定义 (Definition of Done)

整个阶段完成的唯一标准是:

> **O5.2 的完整消融实验显示：SACR v2 + SynQ + Graph RAG 在 SkillsBench 的 5-Fold CV 上达到 Mean Recall@10 ≥ 75%，且严格优于 UCSB Agentic 的 68.3%。**

如果此标准未达成，O8 和 O9 不启动。
