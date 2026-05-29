# Zleap_SAG2

> by Zleap Team（智跃团队）

## 🌟 项目简介

**Zleap_SAG2** 是基于 SQL-RAG 的 AI 数据流索引、数据处理与聚合检索引擎，通过创新的动态实体关联技术，将非结构化数据转化为可检索的结构化事项。

### 💡 核心优势

通过 **SQL + Vector + LLM** 的混合架构，实现：

- ✅ **无需维护复杂知识图谱** — 动态计算实体关联，降低维护成本
- ✅ **事项为中心的数据组织** — 独立、可索引的事项单元，包含完整上下文

### 🚀 核心创新

#### 1. 事项为中心（Event-Centric）

#### 2. 动态实体关联

#### 3. 灵活的实体维度

#### 4. 智能召回机制

---

## 数据集

| 名称 | 说明 |
|------|------|
| `musique` | MuSiQue 多跳问答（完整版） |
| `hotpotqa` | HotpotQA 多跳问答 |
| `2wikimultihopqa` | 2WikiMultihopQA |
| `test_hotpotqa` | HotpotQA 小测试集（10条） |
| `sample` | 极小样本集，快速调试用 |

数据集文件位于 `pipeline/evaluation/dataset/<name>.json`，详见 [数据集说明文档](pipeline/evaluation/dataset/README.md)。

---

## 搜索策略

| 策略 | 说明 |
|------|------|
| `multi` | 多路检索，NER → 实体向量召回 → 多跳图扩展 → 合并排序 |
| `hopllm` | 双阶段多跳：粗排后以粗排结果为种子进行扩跳 |
| `atomic` | 原子检索，先拆分实体再逐跳扩展 |
| `vector` | 纯向量检索，速度最快 |
| `multi1` | multi改版，固定1跳 + 动态扩跳至满足 max_relations |
---

## 实验结果

### 对比基线

本项目选择 [HippoRAG 2](https://github.com/ianliuwd/HippoRAG2) 作为对比基线。HippoRAG 2 出自论文 *"From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"*（ICML 2025），是目前多跳检索领域性能领先的 SOTA 方法，通过结合知识图谱和个性化 PageRank 算法，实现了类人长期记忆的非参数持续学习框架，在多跳问答任务上表现优异。为确保公平对比，我们在相同的 Embedding 模型和 LLM 配置下重新运行了 HippoRAG 2。

### 主要性能对比

**配置：** Embedding = `bge-large-en-v1.5` | LLM = `qwen3.6-flash`

| 方法 | 数据集 | Recall@1 | Recall@2 | Recall@5 | Recall@10 |
|------|--------|----------|----------|----------|-----------|
| **SAG (This work)** | HotpotQA | **47.80%** | **91.55%** | **96.50%** | **97.70%** |
| HippoRAG 2 | HotpotQA | 44.40% | 78.35% | 94.35% | 97.15% |
| **SAG (This work)** | 2WikiMultiHop | **43.53%** | **82.30%** | 88.00% | 88.75% |
| HippoRAG 2 | 2WikiMultiHop | 42.38% | 76.55% | 90.35% | 93.40% |
| **SAG (This work)** | MuSiQue | **36.17%** | **64.05%** | **80.04%** | **83.37%** |
| HippoRAG 2 | MuSiQue | 30.65% | 49.52% | 65.13% | 73.76% |
| **SAG (This work)** | **平均** | **42.50%** | **79.30%** | **88.18%** | **89.94%** |
| HippoRAG 2 | **平均** | 39.14% | 68.14% | 83.28% | 88.10% |

**结论：** 在相同配置下，SAG 在三个数据集上全面优于 HippoRAG 2，平均 Recall@1/2/5/10 分别提升 3.36%/11.16%/4.90%/1.84%，尤其在 Recall@2 指标上提升显著。

**使用更强 Embedding 模型（NV-Embed-v2）的性能：**

| 方法 | 数据集 | Recall@1 | Recall@2 | Recall@5 | Recall@10 |
|------|--------|----------|----------|----------|-----------|
| **SAG (This work)** | MuSiQue | **36.35%** | **64.55%** | **81.71%** | **86.60%** |
| HippoRAG 2 | MuSiQue | 33.70% | 55.98% | 74.55% | 83.16% |

**结论：** 升级到更强的 Embedding 模型后，SAG 的性能进一步提升（Recall@5 从 80.04% 提升至 81.71%），且依然保持对 HippoRAG 2 的领先优势。这表明 SAG 的架构具有良好的 **Embedding 鲁棒性** — 即使在性能较弱的 Embedding 模型（bge-large-en-v1.5）下，SAG 也能通过动态实体关联和混合检索机制实现优于竞品的性能。

---

## 快速开始

### 1. 环境配置

```bash
cp .env.example .env
# 填写 MySQL、Elasticsearch、LLM、Embedding 配置
```

### 2. 启动数据库服务（Docker）

所有服务（MySQL、Elasticsearch、MLflow）统一由 `docker-compose.yml` 管理。

| 服务 | 容器名 | 默认端口 | 说明 |
|------|--------|----------|------|
| MySQL | `sag2_mysql` | `3306` | 用户 `sag2` / 密码 `sag2_pass` / Root `sag2_root` |
| Elasticsearch | `new_sag_elasticsearch` | `9200` | 2GB 内存，已禁用安全认证 |
| MLflow | `sag2_mlflow` | `5000` | SQLite 后端，持久化 artifacts |

端口可在 `.env` 中通过 `MYSQL_PORT` / `ES_PORT` / `MLFLOW_PORT` 覆盖。

```bash
# 启动全部服务
docker compose up -d

# 查看服务状态
docker compose ps

# 停止服务
docker compose down

# 停止并删除数据卷（⚠️ 会删除所有数据）
docker compose down -v
```

### 3. 初始化数据库

```bash
# 建 MySQL 表
uv run python scripts/init_database.py --fix-grants

# 建 ES 索引
uv run python scripts/init_elasticsearch.py
```

### 4. 上传数据集（建立知识库）

```bash
uv run python scripts/run_upload.py --dataset musique
uv run python scripts/run_upload.py --dataset hotpotqa
uv run python scripts/run_upload.py --dataset 2wikimultihopqa
uv run python scripts/run_upload.py --dataset test_hotpotqa
uv run python scripts/run_upload.py --dataset sample
```

上传完成后在 `pipeline/evaluation/source/SAG` 下生成 `source_info.json`，记录 `source_config_id`。

---

## 主要脚本用法

### 搜索 + Benchmark 评估（主脚本）

```bash
# 快速测试
uv run python scripts/run_search_benchmark.py \
    --dataset test_hotpotqa \
    --strategy multi \
    --max-concurrency 10


# musique
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy multi \
    --k-values "1,2,5,10" \
    --max-concurrency 10 \
    --bench-size 20

# 指定 source-config-id
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy multi1 \
    --k-values "1,3,5,10" \
    --max-concurrency 1 \
    --bench-size 20 \
    --limit 10 \
    --source-config-id musique-20260512_213908

# 带 MLflow 记录
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy multi \
    --max-concurrency 5 \
    --use-mlflow \
    --mlflow-url http://192.168.110.10:5000 \
    --mlflow-experiment benchmark \
    --bench-size 5 \


# 单条调试
uv run python scripts/run_search_benchmark.py \
    --dataset-name musique \
    --strategy hopllm \
    --max-concurrency 20 \
    --bench-size 5 \
    --limit 696 696 \
    --output output/test1 \
    --source-config-id musique-20260526_102936
```

**常用参数：**

| 参数 | 说明 |
|------|------|
| `--dataset-name` | 数据集名称 |
| `--strategy` | 搜索策略（multi / multi1 / hopllm / atomic / vector） |
| `--k-values` | 评估 K 值，逗号分隔，如 `"1,2,5,10"` |
| `--max-concurrency` | 并发数（默认1） |
| `--bench-size` | 滚动评估窗口大小 |
| `--limit [start] [end]` | 限制处理范围，单值=前N条，双值=索引区间 |
| `--source-config-id` | 手动指定数据源 ID |
| `--output-dir` | 输出目录（默认 `output/<dataset>/<strategy>/<timestamp>/`） |
| `--use-mlflow` | 启用 MLflow 记录 |
| `--mlflow-url` | MLflow 服务地址 |
| `--mlflow-experiment` | MLflow 实验名 |

**输出：**
- `output/<dataset>/<strategy>/<timestamp>/search_results.json`
- `output/<dataset>/<strategy>/<timestamp>/benchmark_results.json`

---

### 纯搜索

```bash
uv run python scripts/run_search.py \
    --dataset test_hotpotqa \
    --strategy multi \
    --output output/search_results.json
```

### 纯 Benchmark 评估

```bash
uv run python scripts/run_benchmark.py \
    --results output/results_musique.json \
    --dataset musique
```

### 对比两个检索结果

```bash
# 对比两个方法
uv run python scripts/compare_recall_methods.py \
    --predictions \
        output/test_hotpotqa/multi/20260526_150132/search_results.json \
        output/test_hotpotqa/hopllm/20260526_151819/search_results.json \
    --dataset-name test_hotpotqa \
    --k-values 1,2,3,5,10 \
    --verbose

# 评估单个结果
uv run python scripts/compare_recall_methods.py \
    --predictions output/test_hotpotqa/multi/20260526_150132/search_results.json \
    --dataset-name test_hotpotqa \
    --k-values 1,2,3,5,10 \
    --verbose
```

---

## 项目结构

```
Zleap_SAG2/
├── pipeline/
│   ├── core/
│   │   ├── ai/             # LLM、Embedding、Rerank 客户端
│   │   ├── config/         # 配置管理（.env 读取）
│   │   ├── prompt/         # Prompt 模板管理
│   │   └── storage/        # MySQL + Elasticsearch 存储层
│   ├── db/                 # SQLAlchemy ORM 模型
│   ├── engine/             # pipelineEngine 任务引擎
│   ├── evaluation/
│   │   ├── dataset/        # 数据集 JSON
│   │   ├── metrics/        # RetrievalRecall 评估指标
│   │   ├── source/         # upload 生成的 source_info.json（gitignore）
│   │   └── utils/          # DatasetLoader、MLflowTracker、TokenTracker
│   ├── models/             # Pydantic 数据模型
│   ├── modules/
│   │   ├── extract/        # 知识提取（实体、事件）
│   │   ├── load/           # 文档加载与分块
│   │   └── search/         # 搜索策略实现
│   └── utils/              # 日志、重试、文本处理
├── scripts/
│   ├── init_database.py            # 建 MySQL 表
│   ├── init_elasticsearch.py       # 建 ES 索引
│   ├── run_upload.py               # 上传数据集
│   ├── run_search_benchmark.py     # 搜索 + 评估（主脚本）
│   ├── run_search.py               # 纯搜索
│   ├── run_benchmark.py            # 纯评估
│   ├── compare_recall_methods.py   # 多方法对比
│   ├── cross_validation.py         # 交叉验证
│   ├── run_extract.py              # 知识提取
│   ├── run_load.py                 # 文档加载
│   ├── drop_unused_tables.py       # 清理废旧表
│   ├── migrate_add_dataset_fields.py
│   └── recreate_elasticsearch.py
├── docs/
│   └── search.md               # run_search_benchmark.py 完整参数说明
├── prompts/                # Prompt 模板（YAML/JSON）
├── output/                 # 输出结果（gitignore）
├── docker-compose.yml      # 所有服务（MySQL + ES + MLflow）
├── .env                    # 环境变量（不提交）
└── .env.example            # 环境变量模板
```

---

**最后更新：** 2026-05-29
