# SkillGraph 论文图表设计指南

> 本文档详细说明论文中每个 Figure 的设计要求，包括内容、布局、配色、数据标注等。

---

## Figure 1: 系统架构总览图 (Two-Phase Architecture)

**引用位置**: Section 3.1 (Methodology), line 167

**当前状态**: 论文中是一个文字框占位符

**替换位置**: 替换 `fig:architecture` 环境内的 `\fbox{\parbox{...}}` 内容

### 设计目标
直观展示"预处理烧 Token，运行时零 Token"的两阶段架构，让读者一眼理解核心创新。

### 推荐尺寸
- 宽度: 论文单栏宽度 (约 16cm)
- 高度: 8-10cm
- DPI: 300+ (矢量图优先: PDF/SVG)

### 布局设计 (从左到右，分两大列)

```
+----------------------------------------------------------+
|                    SkillGraph Architecture                  |
+----------------------------------------------------------+
|                                                            |
|  +------------------------+    +------------------------+  |
|  |  PREPROCESSING PHASE   |    |   RUNTIME PHASE        |  |
|  |  (One-time,            |    |   (Per-query,          |  |
|  |   Token-Acceptable)    |    |    Zero-Token)         |  |
|  +------------------------+    +------------------------+  |
|  |                        |    |                        |  |
|  |  [Skill Library]       |    |  [User Query]          |  |
|  |      34,396 skills     |    |       "deploy app"     |  |
|  |           |            |    |           |            |  |
|  |           v            |    |           v            |  |
|  |  +---------------+     |    |  +---------------+     |  |
|  |  | LLM SynQ Gen  |     |    |  |   Encoder     |     |  |
|  |  | 10 queries/   |     |    |  | all-MiniLM    |     |  |
|  |  | skill         |     |    |  | 384-dim emb   |     |  |
|  |  +---------------+     |    |  +---------------+     |  |
|  |           |            |    |           |            |  |
|  |           v            |    |           v            |  |
|  |  +---------------+     |    |  +---------------+     |  |
|  |  |  Skill Graph  |     |    |  |    SACR       |     |  |
|  |  | (relations)   |     |    |  |  Router       |     |  |
|  |  +---------------+     |    |  +---------------+     |  |
|  |           |            |    |           |            |  |
|  |           v            |    |           v            |  |
|  |  +---------------+     |    |  +---------------+     |  |
|  |  |  HSC (3-level)|     |    |  | Multi-Vector  |     |  |
|  |  |  8/64/512 com |     |    |  | Scoring       |     |  |
|  |  +---------------+     |    |  +---------------+     |  |
|  |           |            |    |           |            |  |
|  |           v            |    |           v            |  |
|  |  [Indices stored]      |    |  [Top-k Skills]        |  |
|  |  on disk/memory        |    |  ranked by similarity  |  |
|  |                        |    |                        |  |
|  |  Token cost: ~765M     |    |  Token cost: 0         |  |
|  |  (one-time)            |    |  Latency: 17.5ms       |  |
|  +------------------------+    +------------------------+  |
|                                                            |
+----------------------------------------------------------+
```

### 视觉风格建议
- **两列分色**: 左列用暖色调 (橙色/琥珀色 #F59E0B) 表示"可烧Token"，右列用冷色调 (青色/蓝色 #06B6D4) 表示"零Token"
- **阶段标签**: 顶部用色块标注，左"PREPROCESSING"右"RUNTIME"
- **箭头方向**: 左列自上而下表示预处理流水线，右列自上而下表示运行时流水线
- **成本标注**: 底部用对比字体突出 Token 成本差异
- **数据标注**: 34,396、10 queries/skill、8/64/512、384-dim、17.5ms 等关键数字用粗体

### 工具推荐
- Python: `matplotlib` + `patches` 绘制流程图
- 在线工具: draw.io / Excalidraw / Figma
- 推荐输出: PDF 矢量图 (插入 LaTeX 不会失真)

---

## Figure 2: 延迟对比柱状图 (Latency Breakdown)

**引用位置**: Section 4.4 (Experiments), line 426

**当前状态**: 论文中只有文字描述，没有实际 figure 环境

**插入位置**: 在 `\subsection{Latency Analysis}` 段落之前插入 `\begin{figure}...\end{figure}`

### 设计目标
用柱状图展示每种配置的时间分解，突出两阶段架构的速度优势。

### 推荐尺寸
- 宽度: 论文单栏宽度 (约 16cm)
- 高度: 7-8cm
- DPI: 300+

### 数据

| 配置 | Query Encoding | SACR Routing | Similarity Computation | Total |
|------|---------------|--------------|----------------------|-------|
| Dense | 2.0ms | 0ms | 0.8ms | 2.8ms |
| SACR | 2.0ms | 7.0ms | 1.0ms | 10.0ms |
| Dense + SynQ | 2.0ms | 0ms | 15.5ms | 17.5ms |
| SACR + SynQ | 2.0ms | 7.0ms | 17.5ms | 26.5ms |
| UCSB Agentic | N/A | N/A | N/A | ~3000ms |

注: Query Encoding 时间相同 (约 2ms，all-MiniLM-L6-v2 编码)

### 布局设计

**方案 A: 堆叠柱状图 (推荐)**
```
Latency (ms, log scale)
  |
  |                                          [UCSB]
  |                                          3000ms
  |
  |  [SACR+SynQ]       [Dense+SynQ]
  |  26.5ms            17.5ms
  |
  |  [SACR]            [Dense]
  |  10.0ms            2.8ms
  |_____________________________________________
       Dense    SACR   Dense+   SACR+   UCSB
                        SynQ     SynQ   Agentic
```

- X轴: 5种配置 (Dense, SACR, Dense+SynQ, SACR+SynQ, UCSB Agentic)
- Y轴: 延迟 (毫秒)，对数刻度更佳 (因为 UCSB 是 3000ms，其他在 30ms 以内)
- 柱子颜色:
  - Query Encoding: 灰色 (#9CA3AF)
  - SACR Routing: 蓝色 (#3B82F6)
  - Similarity Computation: 绿色 (#10B981)
  - UCSB (整体): 红色 (#EF4444)
- 每个柱子上方标注总延迟数值
- UCSB Agentic 柱子用不同颜色 (红色) 并标注 ~3000ms
- 添加虚线标注 30ms 阈值线

**方案 B: 分组柱状图 + 对数坐标**
- 适合更清晰地展示 5 个配置之间的倍数关系

### 关键视觉元素
1. **对数 Y 轴**: 必须用对数刻度，否则 UCSB 的柱子会压扁其他所有柱子
2. **颜色图例**: 明确标注每种颜色代表的时间组成部分
3. **数值标签**: 每个柱子上方标注总毫秒数
4. **对比标注**: 在 UCSB 柱子和其他柱子之间画双箭头，标注 "170x faster"

### Python 绘图代码框架
```python
import matplotlib.pyplot as plt
import numpy as np

configs = ['Dense', 'SACR', 'Dense+SynQ', 'SACR+SynQ', 'UCSB\nAgentic']
encoding = [2.0, 2.0, 2.0, 2.0, 0]
routing = [0, 7.0, 0, 7.0, 0]
similarity = [0.8, 1.0, 15.5, 17.5, 3000]

colors = ['#9CA3AF', '#3B82F6', '#10B981']
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(configs))
width = 0.6

# 堆叠柱状图
p1 = ax.bar(x, encoding, width, label='Query Encoding', color=colors[0])
p2 = ax.bar(x, routing, width, bottom=encoding, label='SACR Routing', color=colors[1])
p3 = ax.bar(x, similarity, width, bottom=np.array(encoding)+np.array(routing),
            label='Similarity Computation', color=colors[2])

ax.set_yscale('log')
ax.set_ylabel('Latency (ms, log scale)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.legend(loc='upper left')
ax.set_ylim(1, 5000)

# 添加数值标签
for i, (e, r, s) in enumerate(zip(encoding, routing, similarity)):
    total = e + r + s
    if total > 0:
        ax.text(i, total * 1.2, f'{total:.1f}ms', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_latency.pdf', dpi=300, bbox_inches='tight')
```

---

## Figure 3: 召回率对比图 (Recall@10 Comparison)

**引用位置**: 建议插入到 Section 4.2 (Main Results) 之后

**插入位置**: 在 `Table~\ref{tab:main-results}` 之后、Section 4.3 之前

### 设计目标
直观展示各系统的 Recall@10 性能，突出 SkillGraph 的优势和 SynQ 的贡献。

### 推荐尺寸
- 宽度: 论文单栏宽度 (约 16cm)
- 高度: 6-7cm

### 数据

| 系统 | Recall@10 | 颜色 |
|------|-----------|------|
| Dense | 62.8% | 浅灰 |
| SACR | 59.4% | 浅蓝 |
| UCSB Agentic | 68.3% | 橙色 |
| Dense + SynQ (Ours) | 72.1% | 深绿 |
| SACR + SynQ (Ours) | 72.1% | 深蓝 |
| SkillRouter | 74.0% | 紫色 |

### 布局设计

**方案: 水平柱状图 (推荐)**
```
Recall@10 (%)
  |
  |  SkillRouter (74.0%)      [======|=======|=======|====] 74.0
  |  Dense+SynQ (72.1%)       [======|=======|=======|=== ] 72.1
  |  SACR+SynQ (72.1%)        [======|=======|=======|=== ] 72.1
  |  UCSB Agentic (68.3%)     [======|=======|=====     ] 68.3
  |  Dense (62.8%)            [======|=======|===       ] 62.8
  |  SACR (59.4%)             [======|=======|=         ] 59.4
  |_________________________________________________________
       0     20     40     60     80    100
```

- 水平柱状图，Y轴为系统名称，X轴为 Recall@10 百分比
- 颜色编码:
  - 灰色系: 基线方法 (Dense, SACR)
  - 橙色: 对比方法 (UCSB Agentic)
  - 绿色/蓝色: 我们的方法 (Dense+SynQ, SACR+SynQ)
  - 紫色: SOTA (SkillRouter)
- 柱子上方/右侧标注精确百分比
- 添加虚线标注 70% 阈值线
- 在 Dense 和 Dense+SynQ 之间用箭头标注 "+9.3pp"

### 关键视觉元素
1. **性能分组**: 用背景色块区分三组 (基线/对比/我们的)
2. **提升标注**: 用箭头和文字标注关键提升幅度
3. **SOTA 对比**: SkillRouter 作为当前最佳，用虚线框标注

---

## Figure 4: SACR 路由过程示意图

**引用位置**: 建议插入到 Section 3.5 (SACR) 中

**插入位置**: 在 Algorithm 1 之后，用 figure 环境包裹

### 设计目标
用树状图展示 SACR 如何从 34K skills 稀疏路由到候选集。

### 推荐尺寸
- 宽度: 论文单栏宽度 (约 16cm)
- 高度: 9-10cm

### 布局设计

```
+----------------------------------------------------------+
|                    SACR Routing Example                   |
+----------------------------------------------------------+
|                                                            |
|  Query: "Deploy app to Kubernetes"                          |
|       |                                                    |
|       v                                                    |
|  +------------------+                                      |
|  |   Query Embedding |  384-dim vector                     |
|  +------------------+                                      |
|       |                                                    |
|       v                                                    |
|  [L0: 8 Super Communities]                                 |
|  +-----+-----+-----+                                       |
|  | Dev | Data| AI  | ...  (Top-3 selected)                  |
|  | 0.45| 0.12| 0.08|                                       |
|  +-----+-----+-----+                                       |
|       |                                                    |
|       v                                                    |
|  [L1: 8 x 3 = 24 Functional Communities]                   |
|  +--------+--------+--------+                              |
|  | K8s    | CI/CD  | Cloud  | ... (Top-5 selected)         |
|  | 0.38   | 0.32   | 0.15   |                              |
|  +--------+--------+--------+                              |
|       |                                                    |
|       v                                                    |
|  [L2: 8 x 5 = 40 Leaf Communities]                         |
|  +----------+----------+----------+                        |
|  | Deploy   | Monitor  | Scale    | ... (Top-10 selected)  |
|  | 0.41     | 0.28     | 0.19     |                        |
|  +----------+----------+----------+                        |
|       |                                                    |
|       v                                                    |
|  [Candidate Set]                                           |
|  ~1,500 skills (from 10 L2 communities)                    |
|                                                            |
|  Compute: 8 + 24 + 40 = 72 similarities                    |
|  vs Dense: 34,000 similarities                             |
|  Reduction: 472x                                           |
+----------------------------------------------------------+
```

### 视觉风格
- **树形结构**: 从上到下的层级展开
- **注意力权重**: 用数字和颜色深浅表示注意力分数 (高分数=深色)
- **Top-k 高亮**: 被选中的社区用实线框，未选中的用虚线/灰色
- **计算对比**: 底部用对比框展示计算量减少
- **颜色方案**:
  - L0: 深青色 (#0E7490)
  - L1: 中蓝色 (#3B82F6)
  - L2: 浅蓝色 (#93C5FD)
  - 候选集: 绿色 (#10B981)

---

## Figure 5: 语义鸿沟示意图 (The Semantic Gap)

**引用位置**: 建议插入到 Section 3.2 (Synthetic Query Generation) 开头

**插入位置**: 在 `\subsection{Synthetic Query Generation}` 之后、生成策略段落之前

### 设计目标
用具体例子展示用户查询和技能描述之间的语义鸿沟，以及 SynQ 如何桥接它。

### 推荐尺寸
- 宽度: 论文单栏宽度 (约 16cm)
- 高度: 6-7cm

### 布局设计

```
+----------------------------------------------------------+
|                The Semantic Gap Problem                   |
+----------------------------------------------------------+
|                                                            |
|   USER QUERY (Task-oriented)                               |
|   +-------------------------------------------+            |
|   | "How do I get my website online?"         |            |
|   +-------------------------------------------+            |
|                    |                                       |
|                    |  Cosine Sim: 0.23  (LOW!)             |
|                    v                                       |
|   SKILL DESCRIPTION (Function-oriented)                    |
|   +-------------------------------------------+            |
|   | "Vercel deployment: serverless functions, |            |
|   |  edge caching, CI/CD pipeline"            |            |
|   +-------------------------------------------+            |
|                                                            |
|   ================= BRIDGE ===================             |
|                                                            |
|   SYNTHETIC QUERIES (Generated by LLM)                     |
|   +-------------------------------------------+            |
|   | "Deploy my website to a hosting service"  | Sim: 0.78  |
|   | "How to put a web app live on internet"   | Sim: 0.82  |
|   | "Best way to host a React app"            | Sim: 0.71  |
|   | ... (10 queries total)                    |            |
|   +-------------------------------------------+            |
|                    |                                       |
|                    |  Best Sim: 0.82  (HIGH!)              |
|                    v                                       |
|   [RETRIEVAL SUCCESS]                                      |
+----------------------------------------------------------+
```

### 视觉风格
- **上部 (红色调)**: 表示"失败路径" - 用户查询和技能描述之间相似度低
  - 红色箭头标注 "Cosine Sim: 0.23 (LOW!)"
- **中部 (中性色)**: 桥接部分，标注 "Synthetic Queries Bridge the Gap"
- **下部 (绿色调)**: 表示"成功路径" - 合成查询和技能之间相似度高
  - 绿色箭头标注 "Best Sim: 0.82 (HIGH!)"
- **对比设计**: 上下两部分用镜像对称布局，中间用分割线或桥梁图标

---

## Figure 6: 两阶段成本对比图 (Two-Phase Cost Amortization)

**引用位置**: 建议插入到 Section 5.1 (Discussion) 或 Section 1.2

**插入位置**: 在 `\subsection{The Preprocessing-Runtime Trade-off}` 之后

### 设计目标
展示两阶段架构的成本优势：一次性预处理投资 vs 每次查询的运行时成本。

### 推荐尺寸
- 宽度: 论文单栏宽度 (约 16cm)
- 高度: 7-8cm

### 数据

假设每月 1M 次查询：

| 方案 | 预处理成本 | 每次查询成本 | 1M 查询总成本 |
|------|-----------|-------------|--------------|
| LLM Routing (UCSB) | $0 | $0.005 (5000 tokens) | $5,000 |
| Dense Retrieval | $0 | $0.000001 | ~$0 |
| SkillGraph | $765 (one-time) | $0 | $765 |

### 布局设计

**方案: 面积图 / 累积成本对比**
```
Cumulative Cost ($)
  |
  |     UCSB Agentic
  |     /$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $5,000
  |    /
  |   /  SkillGraph
  |  /=====================------------  $765 (flat!)
  | /
  |/_____________________________________
  0      100K    500K    1M      2M     Queries
```

- X轴: 查询次数 (0 到 2M)
- Y轴: 累积成本 (美元)
- 两条线:
  - UCSB Agentic: 斜线，随查询次数线性增长
  - SkillGraph: 水平线，预处理成本后保持平坦
- 交叉点标注: "Break-even at ~150K queries"
- 阴影区域: 标注 "Savings: $4,235 at 1M queries"

### 关键视觉元素
1. **交叉点**: 明确标注盈亏平衡点 (~15万次查询)
2. **面积填充**: 两条线之间的区域用浅绿色填充，标注节省金额
3. **注释**: 添加注释说明 "After 1M queries, SkillGraph saves $4,235 (85%)"

---

## 已完成的表格 (无需图表设计)

以下表格已在论文中用 LaTeX 实现，无需额外设计：

| 表格 | 位置 | 内容 |
|------|------|------|
| Table 1 | tab:comparison | 系统对比表 (7行 x 5列) |
| Table 2 | tab:hsc | HSC 层次结构 (4行 x 4列) |
| Table 3 | tab:main-results | 主实验结果 (6行 x 4列) |
| Table 4 | tab:ablation | 消融实验 (6行 x 4列) |

---

## 设计通用规范

### 配色方案
推荐统一使用以下配色，保持论文视觉一致性：

```
主色调:
- 我们的方法 (Ours): #10B981 (翡翠绿)
- SACR: #3B82F6 (蓝色)
- SynQ: #8B5CF6 (紫色)
- 基线: #9CA3AF (灰色)
- 对比方法 (UCSB): #F59E0B (琥珀色)
- SOTA (SkillRouter): #EF4444 (红色)

辅助色:
- 背景浅色: #F3F4F6
- 文字: #1F2937
- 边框: #D1D5DB
```

### 字体规范
- 图表标题: 12pt, 粗体
- 坐标轴标签: 10pt
- 图例: 9pt
- 数据标注: 9pt, 粗体

### 输出格式
- 优先: PDF 矢量图 (\includegraphics[width=\textwidth]{figure.pdf})
- 备选: PNG 300DPI+
- 避免: JPG (有损压缩)

### LaTeX 插入代码模板
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure1_architecture.pdf}
\caption{The two-phase architecture of \ours{}. Left: preprocessing phase where LLMs generate synthetic queries and build indices (one-time cost). Right: runtime phase where retrieval executes with zero LLM tokens.}
\label{fig:architecture}
\end{figure}
```

---

## 优先级建议

按论文阅读体验重要性排序：

1. **Figure 1 (架构图)**: 最高优先级 - 核心创新，必须在正文中出现
2. **Figure 3 (召回率对比)**: 高优先级 - 实验结果是论文核心
3. **Figure 2 (延迟柱状图)**: 高优先级 - 与召回率一起构成完整实验证据
4. **Figure 5 (语义鸿沟)**: 中优先级 - 帮助读者理解核心问题
5. **Figure 4 (SACR路由)**: 中优先级 - 技术细节，可选
6. **Figure 6 (成本对比)**: 低优先级 - 补充论证，可选

建议至少完成 Figure 1-3，论文就具备了完整的视觉支撑。
