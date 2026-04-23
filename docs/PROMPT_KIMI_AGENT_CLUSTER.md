# Kimi Agent Cluster Prompt: 34K Skills Synthetic Query Generation

> 这是一个自包含的提示词，用于指挥 Kimi Agent 集群完成大规模 Synthetic Query 生成任务。
> 将此提示词完整复制到 Kimi 的 Agent 集群中执行。

---

## 任务概述

你需要为一个包含 **34,396** 个技能的语料库生成 synthetic queries（模拟用户查询）。

**为什么做这件事**：技能检索系统的瓶颈在于"用户查询"和"技能描述"的语义分布不同。用户说"帮我处理这个 PDF 文件"，但技能描述写的是"PDF 文档解析与内容提取引擎"。通过让 LLM 为每个技能生成"用户可能会怎么问"的多样化查询，可以缩小这个分布 gap，提升检索召回率。

**核心目标**：
- 每个 skill 生成 **10 条** synthetic queries
- 总计产出：**343,960 条** synthetic queries
- 输出为可下载的 JSONL 文件
- 失败率 < 3%

---

## 第一步：数据获取

### 1.1 下载 UCSB 34K Skills 数据集

在你的环境中执行以下命令下载数据：

```bash
# 安装 huggingface_hub
pip install -U "huggingface_hub[cli]"

# 下载数据集（约 1.9GB）
huggingface-cli download UCSB-NLP-Chang/skill-usage-hf --local-dir /tmp/skill-usage-hf --repo-type dataset

# 验证下载
cd /tmp/skill-usage-hf/skills-34k
ls -la skills_meta.jsonl  # 应该看到 1.9GB 左右的文件
```

如果 huggingface-cli 下载失败，用 Python 脚本：

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="UCSB-NLP-Chang/skill-usage-hf",
    local_dir="/tmp/skill-usage-hf",
    repo_type="dataset"
)
```

### 1.2 数据格式说明

数据文件路径：`/tmp/skill-usage-hf/skills-34k/skills_meta.jsonl`

每行是一个 JSON，格式如下：

```json
{
  "namespace": "01000001-01001110",
  "repo": "agent-jira-skills",
  "name": "jira-project-management",
  "description": "Administer Jira projects, manage backlogs, create and track issues...",
  "category": "Project Management",
  "owner": "...",
  "readme": "...",
  "installs": 0,
  "github_stars": 0,
  "license": "..."
}
```

**你需要使用的字段**：
- `name`: skill 的名称（如 `jira-project-management`）
- `description`: skill 的功能描述
- `namespace`: 命名空间（用作 skill_id 的一部分）
- `repo`: 仓库名（用作 skill_id 的一部分）

**生成 skill_id 的规则**：`{namespace}--{repo}--{name}`

示例：`01000001-01001110--agent-jira-skills--jira-project-management`

### 1.3 验证数据完整性

```python
import json

path = "/tmp/skill-usage-hf/skills-34k/skills_meta.jsonl"
count = 0
with open(path) as f:
    for line in f:
        count += 1

print(f"Total skills: {count}")  # 应该是 34396
```

---

## 第二步：Synthetic Query 生成策略（关键！）

### 2.1 为什么之前的策略失败了

直接用 skill 的 name + description 生成 queries，会导致 queries 的 embedding 与 skill embedding 高度重合（都在同一个语义邻域），检索时无法提供额外信号。

**正确的生成策略**：让 LLM 扮演"用户"角色，基于 skill 的功能**反向推断**用户会如何表达需求，而不是改写 skill 描述。

### 2.2 Prompt 模板（必须严格使用）

对每个 skill，调用 LLM 时使用以下 prompt：

**System Prompt**：

```
You are an expert at understanding how users naturally express their needs to AI agents.

Given a software tool/skill's name and description, generate 10 diverse, natural user queries that would lead an AI agent to select this skill.

CRITICAL RULES:
1. Write from the USER'S perspective, not the tool's perspective
2. Use natural, conversational language (NOT technical jargon)
3. Do NOT paraphrase or summarize the skill description
4. Imagine the user's actual task/scenario, then express it naturally
5. Include diverse intent patterns: direct requests, implicit needs, questions, vague expressions
6. Each query should be 5-30 words
7. Queries should feel like real user messages to an AI assistant

EXAMPLES:
Skill: "pdf-to-text"
Description: "Extract text content from PDF files"

BAD queries (too close to description):
- "Extract text from PDF documents"
- "Parse PDF files to get text content"

GOOD queries (user's natural expression):
- "I have a scanned contract PDF, can you pull out the text for me?"
- "Convert this PDF resume to editable text"
- "Read the contents of the attached PDF"
- "How do I get text out of a PDF file?"
- "Extract all text from the PDF report"
- "This PDF won't let me copy text, help?"
- "Turn this PDF into plain text I can edit"
- "What's written in this PDF document?"
- "Pull text from the PDF attachment"
- "I need the text content from these PDFs"

OUTPUT FORMAT: Return ONLY a JSON array of 10 strings. No markdown, no explanation.
```

**User Prompt**：

```
Skill name: {name}
Description: {description}

Generate 10 diverse, natural user queries for this skill.
```

### 2.3 API 调用参数

```python
import anthropic  # 或你使用的 LLM 客户端

response = client.messages.create(
    model="claude-sonnet-4-6",  # 或 kimi 支持的等价模型
    max_tokens=2048,
    temperature=0.8,  # 稍高的温度以增加多样性
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": USER_PROMPT}]
)
```

### 2.4 批处理策略（效率优化）

**不要为了效率把多个 skill 塞进一个 prompt！**
每个 skill 单独调用 LLM，但可以并行。

建议配置：
- 每个 agent 实例处理 500-1000 个 skills
- 并发数根据你的 API rate limit 调整
- 每个 skill 生成 10 条 queries
- 失败重试 3 次

---

## 第三步：输出格式

### 3.1 主输出文件

文件名：`synthetic_queries_34k.jsonl`

格式：每行一个 JSON 对象

```json
{
  "skill_id": "01000001-01001110--agent-jira-skills--jira-project-management",
  "skill_name": "jira-project-management",
  "namespace": "01000001-01001110",
  "repo": "agent-jira-skills",
  "synthetic_queries": [
    "I need to set up a new project in Jira and manage the backlog",
    "How do I create and track issues in Jira for my team?",
    "Can you help me administer our Jira project?",
    "I want to organize tasks in Jira for my development team",
    "Set up Jira for tracking bugs and feature requests",
    "Manage Jira project workflows and issue tracking",
    "Help me configure Jira for agile project management",
    "I need to create sprints and manage backlogs in Jira",
    "How to set up Jira boards for my team's projects?",
    "Can Jira help me track project progress and deadlines?"
  ]
}
```

### 3.2 验证要求

每个记录必须满足：
- `skill_id` 不为空
- `synthetic_queries` 是长度为 10 的字符串列表
- 每个 query 长度 >= 5 个字符
- 没有空字符串
- 10 条 queries 之间有足够的多样性（不应该是彼此的轻微改写）

### 3.3 日志文件

文件名：`generation_log.json`

```json
{
  "total_skills": 34396,
  "queries_per_skill": 10,
  "total_api_calls": 34396,
  "total_input_tokens": 12345678,
  "total_output_tokens": 23456789,
  "total_time_seconds": 7200,
  "n_failures": 45,
  "failure_rate": 0.0013,
  "failed_skill_ids": ["id1", "id2", ...],
  "model_name": "claude-sonnet-4-6",
  "timestamp": "2026-04-22T10:00:00Z"
}
```

### 3.4 重试日志（如果有失败）

文件名：`synthetic_retry_log.json`

记录对失败 skills 的重试结果。

---

## 第四步：并行化方案

### 4.1 任务切分

将 34,396 个 skills 分成 N 个批次：

```python
import json
from pathlib import Path

# 读取所有 skills
skills = []
with open("/tmp/skill-usage-hf/skills-34k/skills_meta.jsonl") as f:
    for line in f:
        skills.append(json.loads(line))

# 分成 N 个批次（根据你的 agent 数量调整）
N = 10  # 例如 10 个 agent 并行
batch_size = len(skills) // N

for i in range(N):
    start = i * batch_size
    end = start + batch_size if i < N - 1 else len(skills)
    batch = skills[start:end]

    # 保存为 batch 文件
    with open(f"batch_{i:02d}.jsonl", "w") as f:
        for skill in batch:
            f.write(json.dumps(skill) + "\n")

    print(f"Batch {i}: skills {start}-{end-1} ({len(batch)} skills)")
```

### 4.2 每个 Agent 的任务

每个 agent 执行：
1. 读取分配的 batch 文件
2. 对 batch 中的每个 skill，调用 LLM 生成 10 条 synthetic queries
3. 保存结果到 `synthetic_queries_batch_{i:02d}.jsonl`
4. 保存该 batch 的日志到 `generation_log_batch_{i:02d}.json`

### 4.3 结果合并

所有 batch 完成后，合并：

```python
import json
from pathlib import Path

all_results = []
all_failed = []

total_input_tokens = 0
total_output_tokens = 0
total_time = 0

for i in range(N):
    batch_file = f"synthetic_queries_batch_{i:02d}.jsonl"
    with open(batch_file) as f:
        for line in f:
            all_results.append(json.loads(line))

    log_file = f"generation_log_batch_{i:02d}.json"
    with open(log_file) as f:
        log = json.load(f)
        total_input_tokens += log.get("total_input_tokens", 0)
        total_output_tokens += log.get("total_output_tokens", 0)
        total_time += log.get("total_time_seconds", 0)
        all_failed.extend(log.get("failed_skill_ids", []))

# 保存合并结果
with open("synthetic_queries_34k.jsonl", "w") as f:
    for r in all_results:
        f.write(json.dumps(r) + "\n")

# 保存合并日志
with open("generation_log.json", "w") as f:
    json.dump({
        "total_skills": len(all_results),
        "queries_per_skill": 10,
        "total_api_calls": len(all_results),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_time_seconds": total_time,
        "n_failures": len(all_failed),
        "failure_rate": len(all_failed) / 34396,
        "failed_skill_ids": all_failed,
        "model_name": "claude-sonnet-4-6",
    }, f, indent=2)

print(f"Total results: {len(all_results)}")
print(f"Total failed: {len(all_failed)}")
print(f"Failure rate: {len(all_failed)/34396:.2%}")
```

---

## 第五步：质量检查脚本

运行以下脚本验证输出质量：

```python
import json
from collections import Counter

def validate_output(filepath):
    """Validate the generated synthetic queries."""
    issues = []
    total = 0
    query_lengths = []

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                total += 1

                # Check skill_id
                if not record.get("skill_id"):
                    issues.append(f"Line {line_num}: missing skill_id")

                # Check synthetic_queries
                queries = record.get("synthetic_queries", [])
                if len(queries) != 10:
                    issues.append(f"Line {line_num}: expected 10 queries, got {len(queries)}")

                for q in queries:
                    if len(q) < 5:
                        issues.append(f"Line {line_num}: query too short: '{q}'")
                    query_lengths.append(len(q))

                # Check for duplicates within a skill
                if len(set(queries)) != len(queries):
                    issues.append(f"Line {line_num}: duplicate queries")

            except json.JSONDecodeError:
                issues.append(f"Line {line_num}: invalid JSON")

    print(f"Total records: {total}")
    print(f"Total issues: {len(issues)}")
    print(f"Failure rate: {len(issues)/total:.2%}")
    if query_lengths:
        print(f"Avg query length: {sum(query_lengths)/len(query_lengths):.1f} chars")
        print(f"Min query length: {min(query_lengths)} chars")
        print(f"Max query length: {max(query_lengths)} chars")

    if issues:
        print("\nFirst 10 issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")

    return len(issues) == 0

# Run validation
is_valid = validate_output("synthetic_queries_34k.jsonl")
print(f"\nValidation {'PASSED' if is_valid else 'FAILED'}")
```

---

## 第六步：最终交付物

完成后，请提供以下文件供下载：

| 文件名 | 大小预估 | 说明 |
|--------|---------|------|
| `synthetic_queries_34k.jsonl` | ~500-800 MB | 主输出文件，34,396 行 |
| `generation_log.json` | ~50 KB | 生成日志 |
| `synthetic_retry_log.json` | 可选 | 如果有失败 |
| `validation_report.txt` | ~10 KB | 质量检查报告 |

**交付标准**：
- [ ] `synthetic_queries_34k.jsonl` 行数 = 34,396
- [ ] 每个记录的 `synthetic_queries` 列表长度 = 10
- [ ] 无空字符串或长度 < 5 个字符的 query
- [ ] 失败率 < 3%（即失败 < 1,032 个 skills）
- [ ] `generation_log.json` 包含完整的统计信息

---

## 第七步：如果失败率过高

如果失败率 > 3%，请：
1. 分析失败原因（API 错误？JSON 解析失败？格式不对？）
2. 对失败的 skills 进行重试（最多 3 次）
3. 将重试结果保存到 `synthetic_retry_log.json`
4. 确保最终失败率 < 3%

---

## 成本估算

| 项目 | 数值 |
|------|------|
| 34,396 skills × 10 queries | 343,960 synthetic queries |
| API calls | 34,396 |
| 平均 input tokens/skill | ~150 |
| 平均 output tokens/skill | ~400 |
| 总 input tokens | ~5.1M |
| 总 output tokens | ~13.8M |
| Claude Sonnet 成本 | ~$150-250 |
| Claude Haiku 成本 | ~$15-25 |

**建议**：先用 100 个 skills 做小规模测试，确认输出质量后再全量跑。

---

## 开始执行

请按以下顺序执行：

1. **下载数据**（第一步）
2. **小规模测试**（先跑 100 skills 验证 prompt 效果）
3. **并行生成**（第三步 + 第四步）
4. **质量检查**（第五步）
5. **交付结果**（第六步）

**有任何问题或异常情况，立即停止并报告给我。**
