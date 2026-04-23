# 34K Skills Synthetic Query Generation Plan

## 目标
为 34,396 个技能各生成 10 条高质量 synthetic queries，总计 ~344,000 条，失败率 < 3%。

## Stage 1 — 数据获取与验证
**目标**: 下载 UCSB 34K Skills 数据集并验证完整性
- 使用 huggingface-cli 下载数据集到 `/tmp/skill-usage-hf/skills-34k/`
- 验证 `skills_meta.jsonl` 存在且大小约 1.9GB
- 运行验证脚本，确认总行数 = 34,396
- 抽样检查数据格式（name, description, namespace, repo, installs 字段）

## Stage 2 — 小规模测试（100 skills）
**目标**: 验证 prompt 效果，确保生成质量符合要求
- 选取 100 个具有代表性的 skills（涵盖不同 popularity、不同 category）
- 使用系统 prompt + user prompt 生成 synthetic queries
- 严格按照 10 个 diversity dimensions 检查输出
- 验证失败率、query 质量、格式正确性
- 如发现问题，调整 prompt 后再测试

## Stage 3 — 并行生成（核心）
**目标**: 将 34,396 skills 分批次并行处理
- 将数据集分成 N 个 batch（建议 N=20-30，每批 ~1,200-1,700 skills）
- 每个子代理处理一个 batch，独立生成 synthetic queries
- 每个 skill 单独调用 LLM，temperature=0.8
- 每个 skill 失败时重试 3 次
- 每个 batch 输出 `synthetic_queries_batch_XX.jsonl` 和 `generation_log_batch_XX.json`

### 子代理任务设计
每个子代理接收：
1. batch 文件路径
2. 系统 prompt（含 strict prohibitions 和 diversity requirements）
3. 生成规则和质量检查清单
4. 输出格式要求

子代理输出：
1. `synthetic_queries_batch_XX.jsonl`
2. `generation_log_batch_XX.json`

## Stage 4 — 结果合并与质量验证
**目标**: 合并所有 batch 结果，验证最终质量
- 合并所有 batch 文件为 `synthetic_queries_34k.jsonl`
- 合并所有日志为 `generation_log.json`
- 运行质量检查脚本，验证：
  - 总行数 = 34,396
  - 每个记录的 queries 长度 = 10
  - 无空字符串或长度 < 5 的 query
  - 失败率 < 3%
  - 生成日志完整性
- 如有失败 skills > 3%，进入重试阶段

## Stage 5 — 最终交付
**目标**: 提供可下载的最终文件
- `synthetic_queries_34k.jsonl`（主输出）
- `generation_log.json`（统计日志）
- `synthetic_retry_log.json`（如有重试）
- `validation_report.txt`（质量报告）

## 关键约束
- 每个 skill 单独 LLM 调用（不要把多个 skill 塞进一个 prompt）
- 严格遵循 10 个 diversity dimensions
- 避免 description copying、template repetition、technical jargon
- 从用户视角出发，自然语言
- 最终失败率 < 3%
