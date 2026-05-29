# run_search_benchmark.py 使用说明

执行搜索和评估，支持五种搜索策略：`atomic`、`multi`、`multi1`、`hopllm`、`vector`。
展示逻辑和计分逻辑与 `pipeline/evaluation/benchmark.py` 完全一致。

## 参数说明

### 必填参数

**`--strategy {atomic,multi,multi1,hopllm,vector}`**
搜索策略，五选一：

| 策略 | 说明 |
|------|------|
| `atomic` | 原子检索，先拆分实体再逐跳扩展，适合多跳推理问题 |
| `multi` | 多路检索，同时走多条检索链后合并，召回更全面 |
| `multi1` | 多路检索增强版，双阶段扩跳（固定1跳 + 动态扩跳直到满足 max_events） |
| `hopllm` | HopLLM 双阶段多跳检索，阶段A粗排后以粗排结果为种子进行阶段B扩跳 |
| `vector` | 纯向量检索，速度最快，适合语义相似度匹配场景 |

### 数据集（必填二选一）

**`--dataset-name <名称>`**
数据集简称，如 `hotpotqa`、`musique` 等。
脚本会根据 `.env` 文件中的 `LLM_MODEL` 配置，自动在
`./pipeline/evaluation/source/SAG/<LLM_MODEL>/<数据集名>/`
目录下查找最新时间戳的 `source_info.json`。

> 注意：确保 `.env` 中的 `LLM_MODEL` 对应的模型目录下存在该数据集。
> 同时作为输出目录和 MLflow run 名称的前缀。

**`--source-config-id <id>`**
直接指定 `source_config_id`，跳过自动文件查找。
适合已知 `source_config_id` 或需要跨模型复用数据源的场景。

### 输出

**`--output-dir <路径>`**
结果输出目录。
默认：`output/<数据集名>/<策略>/<时间戳>/`

目录下会生成两个文件：
- `search_results.json` — 每条问题的检索结果（`retrieved_docs` 文本列表）
- `benchmark_results.json` — Recall@K 评估指标及分布统计

### 检索参数

**`--top-k <整数>`**
每次查询返回的最大文档数，同时也是 Recall 评估的上限 K。默认：`10`

**`--k-values "<k1,k2,...>"`**
计算 Recall@K 时使用的 K 值列表，逗号分隔。默认：`"1,2,5,10"`（与 `benchmark.py` 一致）

**`--max-concurrency <整数>`**
并发查询数，控制同时发出的搜索请求数量。默认：`1`（顺序执行，最稳定）

**`--limit <整数> [<整数>]`**
限制处理范围，用于快速调试：
- `--limit N` — 只处理前 N 条（索引 0 ~ N-1）
- `--limit S E` — 处理第 S 条到第 E 条（含两端，0-based 索引）

示例：`--limit 7 8` 处理 question_index 7 和 8，共 2 条

**`--bench-size <整数>`**
每处理 N 个问题打印一次累积统计（与 `benchmark.py --bench-size` 一致）。默认：`5`

### MLflow 跟踪

**`--use-mlflow`**
开关参数，加上即启用 MLflow 实验跟踪（默认关闭）。

**`--mlflow-url <URL>`**
MLflow Tracking Server 地址。
如果不指定，将使用 `.env` 文件中的 `MLFLOW_URL` 配置。

**`--mlflow-experiment <名称>`**
MLflow 实验名称。默认：`"lgxbenchmark"`

## 使用示例

```bash
# 最简用法（使用 .env 中配置的 LLM_MODEL）
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy atomic

# 只跑第 7、8 条做快速验证
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy vector \
    --limit 7 8

# 只跑前 20 条做快速验证
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy vector \
    --limit 20

# 开启并发，自定义 K 值
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy multi \
    --top-k 10 \
    --k-values "1,2,5,10" \
    --max-concurrency 5

# 使用 multi1 双阶段扩跳策略
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy multi1 \
    --top-k 10

# 启用 MLflow 跟踪
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy atomic \
    --use-mlflow \
    --mlflow-url http://192.168.1.100:5000
```
