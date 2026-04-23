# SkillGraph 项目修复报告

**日期**: 2026-04-21
**修复目标**: 消除测试集泄露，建立公平的评估方法论

---

## 一、发现的问题总结

### 1.1 致命问题：测试集超参数调优（已修复）

**原问题**: `evals/eval_skillsbench_sacr_tuning.py` 直接在 SkillsBench 全部 87 个任务上搜索 SACR 最佳参数 (L0/L1/L2 top_k)，然后用这些参数报告测试结果。

**修复**: 创建 `evals/eval_cross_validation.py`，实现 5-Fold 交叉验证：
- 每轮使用 4 折（~70 任务）搜索参数，在剩余 1 折（~17 任务）上测试
- 报告 5 折测试结果的均值和标准差

### 1.2 致命问题：Keyword Map 基于测试集 Ground Truth 构建（已修复）

**原问题**: `config/keywords.json` 中 135/189 (71%) 的映射直接对应 SkillsBench 的 ground truth skills。构建者查看了测试集的任务名称和 GT skills，手动编写了 keyword -> skill 映射。

**修复**:
- 创建 `scripts/generate_auto_keywords.py`，**纯自动**从 UCSB 34K 技能库生成 keyword map
- 创建 `config/keywords_auto.json`，零测试集泄露
- 修改 `skill_graph/config/keywords.py` 添加 `load_auto_keyword_map()`
- 修改 `skill_graph/matching/keyword_matcher.py` 支持选择 auto map
- 修改 `skill_graph/api/skill_graph.py` 支持 `use_auto_keywords` 参数

### 1.3 严重问题："Dense Baseline" 标签误导（已修复）

**原问题**: `eval_skillsbench_v2.py` 中的 "Dense Baseline" 实际上也启用了 keyword matching（默认 `use_keywords=True`）。

**修复**: 交叉验证脚本明确测试 4 种系统变体：
- 纯 SACR（无 keyword）
- SACR + Keyword
- 纯 Dense（无 keyword）
- Dense + Keyword

---

## 二、修复后的真实评估结果

### 2.1 5-Fold 交叉验证结果（Manual Keyword Map）

| 系统变体 | Recall@10 | Latency | 备注 |
|---------|-----------|---------|------|
| **SACR Only** | **59.4% ± 9.2%** | 9.7 ms | 纯稀疏路由 |
| **SACR + Keyword** | **94.1% ± 5.3%** | 5.0 ms | 完整系统 |
| **Dense Only** | **62.8% ± 12.7%** | 17.0 ms | 全量相似度 |
| **Dense + Keyword** | **94.1% ± 5.3%** | 17.6 ms | 全量+关键词 |

**关键发现**:
- Keyword matching 提供 **+31~35 个百分点** 的提升
- SACR + Keyword 与 Dense + Keyword 的 Recall **几乎相同** (94.1%)
- 但 SACR 比 Dense **快 3.5 倍** (5.0ms vs 17.6ms)

### 2.2 Manual vs Auto Keyword Map 对比（5-Fold CV）

| 系统变体 | Manual Map | Auto Map | 差距 |
|---------|-----------|---------|------|
| SACR Only | 59.4% ± 9.2% | 59.4% ± 9.2% | 0% |
| **SACR + Keyword** | **94.1% ± 5.3%** | **39.0% ± 6.1%** | **-55.1%** |
| Dense Only | 62.8% ± 12.7% | 62.8% ± 12.7% | 0% |
| **Dense + Keyword** | **94.1% ± 5.3%** | **39.0% ± 6.1%** | **-55.1%** |

**关键发现**:
- Auto keyword map 的表现**极差**，比 Manual map 低了 **55 个百分点**
- 这不是因为泛化关键词（已过滤掉映射 >5 个技能的关键词）
- 根本原因是：**Auto map 从技能名提取的关键词与 query 中实际出现的词汇分布不匹配**

### 2.3 为什么 Auto Keyword Map 效果差？

**命中分析**:
- Manual map 在 87 个任务中命中 **63 个** (72.4%)
- Auto map 在 87 个任务中只命中 **14 个** (16.1%)
- Manual 有 **53 个独有命中**，Auto 只有 **4 个独有命中**

**根本原因**:
1. Manual map 的构建者**知道 query 中会出现什么词**（通过查看 SkillsBench 任务描述）
2. 例如 "pdf", "xlsx", "csv", "video", "audio" 等是任务描述中的高频词
3. Auto map 只能从技能名提取关键词，但很多 query 中的词汇不在技能名中
4. 例如任务 "court-form-filling" 的 query 提到 "PDF"，但技能名可能是 "pdf-processing" 或 "document-parser"

---

## 三、关于"在目标数据上调参"合理性的讨论

用户的辩护是合理的：**如果系统的目的就是针对特定技能库优化，那么在目标数据上做一次性预处理（包括调参）是合理的**。这类似于 FAISS 索引构建或搜索引擎的索引优化。

**合理的前提**:
1. 调参过程**不使用测试集的 ground truth 标签**
2. 对比时明确说明这是"针对目标集合优化" vs "通用方法"
3. Keyword map 从技能库**自动生成**，而非手动对照测试集编写

**当前项目的违规点**:
| 操作 | 是否使用 GT 标签 | 是否合理 |
|------|-----------------|---------|
| SACR 参数在 SkillsBench 上调优 | **是**（用 Recall 调参） | 违规 |
| Keyword map 手动编写 | **是**（135/189 映射直接对应 GT） | 严重违规 |
| HSC 聚类在全部 34K 技能上 | 否（无监督） | 合理 |
| Embedding 用 MiniLM 生成 | 否（预训练模型） | 合理 |

**修复后的状态**:
- SACR 参数：已通过 5-Fold CV 消除泄露
- Keyword map：Auto map 零泄露，但效果差
- 仍需讨论：手动 keyword map 的合理性

---

## 四、修复后的代码变更

### 4.1 新增文件

| 文件 | 用途 |
|------|------|
| `scripts/generate_auto_keywords.py` | 自动生成零泄露 keyword map |
| `config/keywords_auto.json` | 自动生成的 keyword map (10,131 keywords) |
| `evals/eval_cross_validation.py` | 5-Fold 交叉验证评估 |
| `evals/eval_compare_keywords.py` | Manual vs Auto keyword map 对比 |

### 4.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `skill_graph/config/keywords.py` | 添加 `load_auto_keyword_map()` |
| `skill_graph/matching/keyword_matcher.py` | 支持 `use_auto` 参数 |
| `skill_graph/api/skill_graph.py` | 支持 `use_auto_keywords` 参数 |

### 4.3 保留但标记为不推荐的文件

| 文件 | 说明 |
|------|------|
| `evals/eval_skillsbench_sacr_tuning.py` | 直接在测试集上调参，数据泄露 |
| `evals/eval_skillsbench.py` | v1 版本，查询提取有缺陷 |
| `config/keywords.json` | 手动编写，含测试集泄露 |

---

## 五、与 UCSB 论文的公平对比

### 5.1 原项目的误导性对比

| 指标 | 原项目声称 | 实际情况 |
|------|-----------|---------|
| 项目 Hybrid | **94.3%** | 用测试集调参 + 手动 keyword map |
| UCSB Hybrid | **68.3%** | 通用方法，无针对 benchmark 优化 |
| 优势 | **+26pp** | 不公平对比 |

### 5.2 公平对比框架

**场景 A：通用方法对比（零泄露）**
- 项目：SACR Only (5-Fold CV) = **59.4%**
- UCSB：Agentic Hybrid = **68.3%**
- **UCSB 领先 +8.9pp**

**场景 B：可优化方法对比**
- 项目：SACR + Manual Keyword = **94.1%**（但 manual keyword map 含泄露）
- 项目：SACR + Auto Keyword = **39.0%**（零泄露，但效果差）
- UCSB：如果也允许针对 SkillsBench 优化 keyword map，结果未知

**场景 C：速度对比（公平）**
- 项目 SACR：~6ms（稀疏路由）
- 项目 Dense：~17ms（全量相似度）
- UCSB Agentic：~秒级（多次 LLM 调用）
- **项目在速度上有 100-1000 倍优势**

---

## 六、存在的问题和限制

### 6.1 Auto Keyword Map 效果差（已确认）

- 从技能名机械提取的关键词无法捕捉 query 中的词汇分布
- 这是一个根本性的挑战，不仅仅是参数调优问题
- **可能的改进方向**:
  - 分析技能描述（而非仅名称）提取关键词
  - 使用 TF-IDF 从技能描述中找出最具区分性的词汇
  - 结合 query 日志（如果有的话）学习关键词

### 6.2 手动 Keyword Map 的伦理问题

- 如果手动 map 的构建者**没有查看测试集**，只是基于领域知识编写，这是合理的
- 但实际上 71% 的映射直接对应 GT skills，说明**确实查看了测试集**
- **建议**：保留手动 map 作为"oracle 上限"，但明确标注其构建方式

### 6.3 SACR 参数对数据分布敏感

- 5-Fold CV 中，不同 fold 的最佳 SACR 参数差异较大
- Fold 1: (8, 60, 150)，Fold 5: (20, 80, 200)
- 说明 SACR 的泛化能力有限，参数需要针对具体数据分布调整

### 6.4 评估指标过于宽松

- Recall@10 的定义：只要 top-10 中包含**至少一个** GT skill 即算成功
- 对于多 GT 任务（如 travel-planning 有 6 个 GT skills），这个指标非常容易满足
- 建议未来使用更严格的指标，如 Average Precision 或 nDCG

---

## 七、结论

### 7.1 修复成果

1. **消除了 SACR 参数的数据泄露** — 通过 5-Fold 交叉验证
2. **建立了 Auto Keyword Map 生成流程** — 零测试集泄露
3. **实现了组件级评估** — 可以单独评估 SACR、Keyword、Dense 的贡献
4. **发现了手动 keyword map 的严重泄露** — 71% 映射直接对应 GT

### 7.2 真实能力评估

| 能力 | 修复前（作弊） | 修复后（真实） |
|------|---------------|---------------|
| SACR 路由 Recall | 59.8%（含 keyword 作弊） | **59.4%**（纯 SACR，CV） |
| Hybrid 系统 Recall | 94.3%（手动 keyword 作弊） | **39.0%**（Auto keyword，CV） |
| 速度优势 | 2x faster than Dense | **3.5x faster** than Dense（真实） |

### 7.3 诚实的结果

- **纯 SACR 路由**：59.4% Recall@10，比 UCSB 的 68.3% 低 **8.9pp**
- **SACR + Auto Keyword**：39.0% Recall@10，远低于 UCSB 的 68.3%
- **速度优势**：SACR 比 Dense 快 3.5 倍，比 UCSB Agentic 快 100-1000 倍

### 7.4 核心价值

项目的**真实价值**在于速度，而非 recall：
- SACR 提供了与 Dense 相似的 recall（59% vs 63%），但快 3.5 倍
- 如果需要 keyword matching 来提升 recall，需要针对 query 分布优化
- 这与 UCSB 的 Agentic 方法形成互补：SACR 用于快速初筛，Agentic 用于精确优化

---

## 八、下一步建议

1. **改进 Auto Keyword Map**：从技能描述（而非仅名称）提取关键词，使用 TF-IDF 加权
2. **多指标评估**：除了 Recall@10，增加 AP、nDCG、MRR 等指标
3. **消融分析**：系统分析哪些组件对哪些类型的任务最有效
4. **与 UCSB 的公平对比**：在相同评估设置下（都使用 5-Fold CV）对比两种方法
5. **文档更新**：在 README 中明确标注评估方法论和限制
