#!/usr/bin/env python3
"""
搜索 + Benchmark 脚本

执行搜索和评估，支持五种搜索策略：atomic, multi, multi1, hopllm, vector。
展示逻辑和计分逻辑与 pipeline/evaluation/benchmark.py 完全一致。

详细参数说明和使用示例见 docs/search.md。
"""

import argparse
import asyncio
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from pipeline.evaluation.metrics import RetrievalRecall
from pipeline.evaluation.utils import DatasetLoader, MLflowTracker, MLflowConfig, get_local_ip
from pipeline.evaluation.utils import LLMTokenTracker, enable_llm_tracking
from pipeline.utils import get_logger
from pipeline.core.config import get_settings

logger = get_logger("scripts.run_search_benchmark")

# 压制 pipeline 内部的详细日志，只保留 WARNING 以上
logging.getLogger("pipeline").setLevel(logging.WARNING)
# 单独放开 hopllm / multi / multi1 的检索流程日志（实际 logger 前缀是 pipeline.）
logging.getLogger("pipeline.search.multi").setLevel(logging.INFO)
logging.getLogger("pipeline.search.hopllm").setLevel(logging.INFO)
# 放开 LLM 重试日志，方便观察重试次数和等待时间
logging.getLogger("pipeline.ai.llm").setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：与 benchmark.py 保持完全一致
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_source_info(dataset_name: str, model_name: str) -> Dict[str, Any]:
    """
    从 pipeline/evaluation/source/SAG/{model_name}/{dataset_name}/{timestamp}/
    加载指定模型下最新时间戳文件夹的 source_info.json。

    Args:
        dataset_name: 数据集名称
        model_name: 模型名称（从 .env 的 LLM_MODEL 读取）

    Returns:
        包含 source_config_id, model_name, timestamp 等信息的字典

    Raises:
        FileNotFoundError: 模型目录、数据集目录或 source_info.json 不存在
    """
    current_file = Path(__file__)
    sag_base_dir = current_file.parent.parent / "pipeline" / "evaluation" / "source" / "SAG"

    if not sag_base_dir.exists():
        raise FileNotFoundError(f"SAG directory not found: {sag_base_dir}")

    # 直接定位到指定模型的目录
    model_dir = sag_base_dir / model_name
    if not model_dir.exists():
        available_models = [d.name for d in sag_base_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"模型目录不存在: {model_dir}\n"
            f"可用模型: {available_models}\n"
            f"提示: 请检查 .env 文件中的 LLM_MODEL 配置是否正确"
        )

    dataset_dir = model_dir / dataset_name
    if not dataset_dir.exists():
        available_datasets = [d.name for d in model_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"数据集目录不存在: {dataset_dir}\n"
            f"模型 {model_name} 下可用的数据集: {available_datasets}"
        )

    # 收集该模型该数据集下所有时间戳的 source_info.json
    all_source_info_files = []
    for ts_dir in dataset_dir.iterdir():
        if ts_dir.is_dir():
            source_info_path = ts_dir / "source_info.json"
            if source_info_path.exists():
                all_source_info_files.append(source_info_path)

    if not all_source_info_files:
        raise FileNotFoundError(
            f"在 {dataset_dir} 下未找到任何 source_info.json 文件"
        )

    # 按时间戳排序（目录名格式：YYYYMMDD_HHMMSS），选择最新的
    latest_source_info_path = max(all_source_info_files, key=lambda p: p.parent.name)
    logger.info(f"Loading source info from: {latest_source_info_path}")

    with open(latest_source_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return {
        "source_config_id": info.get("source_config_id"),
        "dataset_name": info.get("dataset_name"),
        "timestamp": info.get("timestamp"),
        "source_name": info.get("source_name"),
        "file_path": str(latest_source_info_path),
    }


def calculate_precision_f1_at_k(
    gold_docs: List[List[str]],
    retrieved_docs: List[List[str]],
    k_list: List[int],
) -> Dict[str, float]:
    """
    计算 Precision@K 和 F1@K。
    与 benchmark.py 的 Evaluate._calculate_precision_f1_at_k 完全一致。
    """
    k_list = sorted(set(k_list))
    pooled_precision = {k: 0.0 for k in k_list}
    pooled_f1 = {k: 0.0 for k in k_list}

    num_examples = len(gold_docs)
    if num_examples == 0:
        result: Dict[str, float] = {}
        for k in k_list:
            result[f"Precision@{k}"] = 0.0
            result[f"F1@{k}"] = 0.0
        return result

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        gold_set = set(example_gold_docs)
        for k in k_list:
            top_k_docs = example_retrieved_docs[:k]
            relevant_retrieved = set(top_k_docs) & gold_set

            precision = len(relevant_retrieved) / len(top_k_docs) if top_k_docs else 0.0
            recall = len(relevant_retrieved) / len(gold_set) if gold_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  ) if (precision + recall) > 0 else 0.0

            pooled_precision[k] += precision
            pooled_f1[k] += f1

    pooled_results: Dict[str, float] = {}
    for k in k_list:
        pooled_results[f"Precision@{k}"] = round(pooled_precision[k] / num_examples, 4)
        pooled_results[f"F1@{k}"] = round(pooled_f1[k] / num_examples, 4)
    return pooled_results


# ─────────────────────────────────────────────────────────────────────────────
# 批次回调：与 benchmark.py 的 _handle_bench_callback 完全一致
# ─────────────────────────────────────────────────────────────────────────────

async def handle_bench_callback(
    current_idx: int,
    total: int,
    results_so_far: List[Dict],          # 已完成的 search_result 列表
    gold_docs_for_recall: List[List[str]],
    bench_size: int,
    mlflow_tracker: Optional[Any],
    bench_logger,
) -> None:
    """
    处理 bench 回调逻辑，与 benchmark.py._handle_bench_callback 完全一致。
    """
    batch_index = (current_idx + bench_size - 1) // bench_size
    total_batches = (total + bench_size - 1) // bench_size

    bench_logger.info(f"\n{'='*80}")
    bench_logger.info(
        f"📝 Bench 进度: 批次 {batch_index}/{total_batches} ({current_idx}/{total} 问题)")
    bench_logger.info(f"{'='*80}")

    current_gold_docs = gold_docs_for_recall[:current_idx]

    full_count = partial_count = zero_count = 0
    retrieved_docs_list = []

    for i, result in enumerate(results_so_far):
        sections = result.get("sections", [])
        retrieved_docs_list.append(sections)
        gold_docs = current_gold_docs[i]
        matched = [doc for doc in sections if doc in gold_docs]

        if len(matched) == 0:
            zero_count += 1
        elif len(matched) < len(gold_docs):
            partial_count += 1
        else:
            full_count += 1

    bench_logger.info(f"\n📊 累积召回情况统计 ({current_idx} 个问题):")
    bench_logger.info("=" * 50)
    bench_logger.info(
        f"✅ 全部召回: {full_count} 个 ({full_count/current_idx*100:.1f}%)")
    bench_logger.info(
        f"⚠️  部分召回: {partial_count} 个 ({partial_count/current_idx*100:.1f}%)")
    bench_logger.info(
        f"❌ 零召回: {zero_count} 个 ({zero_count/current_idx*100:.1f}%)")
    bench_logger.info("=" * 50)

    # Recall@K
    recall_metric = RetrievalRecall()
    pooled_recall, _ = recall_metric.calculate_metric_scores(
        gold_docs=current_gold_docs,
        retrieved_docs=retrieved_docs_list,
        k_list=[1, 2, 5, 10]
    )

    # Precision@K / F1@K
    pooled_precision_f1 = calculate_precision_f1_at_k(
        gold_docs=current_gold_docs,
        retrieved_docs=retrieved_docs_list,
        k_list=[2, 5, 10]
    )

    bench_logger.info(f"\n✅ 累积Recall@K:")
    for metric, score in pooled_recall.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score*100:.2f}%)")

    bench_logger.info(f"\n✅ 累积Precision/F1@K:")
    for metric, score in pooled_precision_f1.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score*100:.2f}%)")

    # MLflow 上报
    if mlflow_tracker:
        pooled_results = {**pooled_recall, **pooled_precision_f1}
        mlflow_tracker.log_evaluation_metrics(
            full_count, partial_count, zero_count, current_idx, batch_index
        )
        mlflow_tracker.log_recall_metrics(pooled_results, batch_index)


# ─────────────────────────────────────────────────────────────────────────────
# 核心搜索逻辑（保留三种策略）
# ─────────────────────────────────────────────────────────────────────────────

async def run_batch_search(
    questions: List[str],
    source_config_id: str,
    strategy: str,
    top_k: int,
    max_concurrency: int,
    bench_size: int,
    gold_docs_for_recall: List[List[str]],
    mlflow_tracker: Optional[Any],
    bench_logger,
) -> List[Dict]:
    """
    批量搜索，内置 bench_size 回调，通过 pipelineEngine 统一执行搜索。

    返回格式与 benchmark.py 的 search_results 完全一致：
    [
        {
            'question_index': int,
            'question': str,
            'sections': List[str],       # "title\ncontent" 格式
        },
        ...
    ]
    """
    from pipeline import pipelineEngine
    from pipeline.engine.config import TaskConfig
    from pipeline.modules.search.config import (
        SearchBaseConfig, RerankConfig, RerankStrategy, ReturnType,
        MultiConfig, AtomicConfig, VectorConfig,
    )

    # 策略配置映射：直接从字符串映射到 (枚举, 配置工厂)
    strategy_config_map = {
        "atomic": (RerankStrategy.ATOMIC, lambda: AtomicConfig(max_sections=top_k)),
        "multi":  (RerankStrategy.MULTI,  lambda: MultiConfig(strategy="multi", max_sections=top_k)),
        "multi1": (RerankStrategy.MULTI,  lambda: MultiConfig(strategy="multi1", max_sections=top_k)),
        "hopllm": (RerankStrategy.MULTI,  lambda: MultiConfig(strategy="hopllm", max_sections=top_k)),
        "vector": (RerankStrategy.VECTOR, lambda: VectorConfig(top_k=top_k)),
    }

    rerank_strategy, config_factory = strategy_config_map[strategy]
    strategy_config = config_factory()

    total = len(questions)
    results_so_far: List[Dict] = []
    search_results: List[Dict] = []

    # 信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrency)

    async def search_one(idx: int, question: str) -> Dict:
        async with semaphore:
            try:
                # 通过引擎执行搜索，不直接访问底层 searcher
                engine = pipelineEngine(
                    task_config=TaskConfig(
                        task_name=f"search_{idx}",
                        source_config_id=source_config_id,
                    ),
                    auto_setup_logging=False,
                )
                await engine.search_async(
                    SearchBaseConfig(
                        query=question,
                        return_type=ReturnType.PARAGRAPH,
                        rerank=RerankConfig(strategy=rerank_strategy),
                        strategy_config=strategy_config,
                    )
                )
                engine_result = engine.get_result()
                raw_sections = (
                    engine_result.search_result.data_full
                    if engine_result and engine_result.search_result
                    else []
                )

                def _normalize_section(s) -> str:
                    if isinstance(s, str):
                        return s
                    heading = s.get("heading", "") or s.get("title", "") or ""
                    content = s.get("content", "") or ""
                    # 去掉 content 开头的 markdown 标题行（"# ..." 格式）
                    lines = content.split("\n")
                    if lines and lines[0].strip().lstrip("#").strip() == heading.strip():
                        content = "\n".join(lines[1:]).lstrip("\n")
                    return f"{heading}\n{content}"

                sections = [_normalize_section(s) for s in raw_sections]
                sections = sections[:top_k]
                return {
                    "question_index": idx + 1,
                    "question": question,
                    "sections": sections,
                }
            except Exception as e:
                logger.warning(f"问题 {idx+1} 搜索失败: {e}")
                return {
                    "question_index": idx + 1,
                    "question": question,
                    "sections": [],
                }

    # 顺序提交，保持结果顺序，并在批次边界触发回调
    tasks = [search_one(i, q) for i, q in enumerate(questions)]

    # 按 bench_size 分批执行（保证回调时序与 benchmark.py 一致）
    for batch_start in range(0, total, bench_size):
        batch_tasks = tasks[batch_start: batch_start + bench_size]
        batch_results = await asyncio.gather(*batch_tasks)

        for r in batch_results:
            search_results.append(r)
            results_so_far.append(r)

        current_idx = len(results_so_far)
        await handle_bench_callback(
            current_idx=current_idx,
            total=total,
            results_so_far=results_so_far,
            gold_docs_for_recall=gold_docs_for_recall,
            bench_size=bench_size,
            mlflow_tracker=mlflow_tracker,
            bench_logger=bench_logger,
        )

    return search_results


# ─────────────────────────────────────────────────────────────────────────────
# 最终结果打印：与 benchmark.py 主函数的汇总输出完全一致
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(
    dataset_name: str,
    strategy: str,
    questions: List[str],
    search_results: List[Dict],
    gold_docs_for_recall: List[List[str]],
    k_values: List[int],
    search_time: float,
    bench_logger,
) -> Dict[str, Any]:
    """
    打印最终汇总结果，格式与 benchmark.py 完全一致。
    返回包含所有指标的字典（供保存 JSON）。
    """
    total = len(search_results)

    # ── 召回统计 ──────────────────────────────────────────────
    full_count = partial_count = zero_count = 0
    retrieved_docs_list = []

    for i, result in enumerate(search_results):
        sections = result.get("sections", [])
        retrieved_docs_list.append(sections)
        if i < len(gold_docs_for_recall):
            gold_docs = gold_docs_for_recall[i]
            matched = [doc for doc in sections if doc in gold_docs]
            if len(matched) == 0:
                zero_count += 1
            elif len(matched) < len(gold_docs):
                partial_count += 1
            else:
                full_count += 1

    # ── Recall@K ──────────────────────────────────────────────
    recall_metric = RetrievalRecall()
    pooled_recall, _ = recall_metric.calculate_metric_scores(
        gold_docs=gold_docs_for_recall[:total],
        retrieved_docs=retrieved_docs_list,
        k_list=k_values
    )

    # ── Precision@K / F1@K ────────────────────────────────────
    precision_f1_k_list = [k for k in k_values if k >= 2]
    if not precision_f1_k_list:
        precision_f1_k_list = k_values
    pooled_precision_f1 = calculate_precision_f1_at_k(
        gold_docs=gold_docs_for_recall[:total],
        retrieved_docs=retrieved_docs_list,
        k_list=precision_f1_k_list
    )

    # ── 打印（与 benchmark.py _handle_bench_callback 格式完全一致）──
    bench_logger.info(f"\n{'='*80}")
    bench_logger.info(f"📝 最终评估结果: 数据集={dataset_name}, 策略={strategy}, 共 {total} 个问题")
    bench_logger.info(f"{'='*80}")

    bench_logger.info(f"\n📊 最终召回情况统计 ({total} 个问题):")
    bench_logger.info("=" * 50)
    bench_logger.info(
        f"✅ 全部召回: {full_count} 个 ({full_count/total*100:.1f}%)")
    bench_logger.info(
        f"⚠️  部分召回: {partial_count} 个 ({partial_count/total*100:.1f}%)")
    bench_logger.info(
        f"❌ 零召回: {zero_count} 个 ({zero_count/total*100:.1f}%)")
    bench_logger.info("=" * 50)

    bench_logger.info(f"\n✅ 最终Recall@K:")
    for metric, score in pooled_recall.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score*100:.2f}%)")

    bench_logger.info(f"\n✅ 最终Precision/F1@K:")
    for metric, score in pooled_precision_f1.items():
        bench_logger.info(f"  {metric}: {score:.4f} ({score*100:.2f}%)")

    bench_logger.info(f"\n阶段耗时统计:")
    bench_logger.info("=" * 50)
    bench_logger.info(f"  SEARCH阶段: {search_time:.1f} 秒")
    bench_logger.info(f"  总计: {search_time:.1f} 秒")
    bench_logger.info("=" * 50)

    total_successful = sum(1 for r in search_results if r.get("sections"))
    bench_logger.info(
        f"\n检索统计: {total_successful}/{total} 个问题检索成功")
    bench_logger.info(f"{'='*80}\n")

    return {
        "recall": pooled_recall,
        "precision_f1": pooled_precision_f1,
        "statistics": {
            "total_questions": total,
            "full_recall_count": full_count,
            "partial_recall_count": partial_count,
            "zero_recall_count": zero_count,
            "successful_searches": total_successful,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="搜索 + Benchmark 评估（展示/计分逻辑与 benchmark.py 完全一致，保留三策略）"
    )

    # 必填
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="数据集名称 (musique, hotpotqa, test_hotpotqa 等)")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["atomic", "multi", "multi1", "hopllm", "vector"],
                        help="搜索策略")

    # 检索参数
    parser.add_argument("--top-k", type=int, default=10,
                        help="返回前K个结果（默认：10）")
    parser.add_argument("--k-values", type=str, default="1,2,5,10",
                        help="评估的K值列表，逗号分隔（默认：1,2,5,10）")
    parser.add_argument("--max-concurrency", type=int, default=1,
                        help="搜索并发数（默认：1）")
    parser.add_argument("--limit", type=int, nargs='+', default=None, metavar='N',
                        help="限制处理范围：--limit N 只处理前N条；--limit S E 处理第S到第E条（0-based，含两端）")
    parser.add_argument("--bench-size", type=int, default=5,
                        help="每 N 个问题打印一次累积统计（默认：5）")

    # 输出
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认：output/<数据集>/<策略>/<时间戳>/）")

    # 数据源
    parser.add_argument("--source-config-id", type=str, default=None,
                        help="直接指定 source_config_id，跳过自动文件查找")

    # MLflow
    parser.add_argument("--use-mlflow", action="store_true",
                        help="启用 MLflow 跟踪")
    parser.add_argument("--mlflow-url", type=str, default=None,
                        help="MLflow Tracking Server 地址（默认使用 .env 中的 MLFLOW_URL）")
    parser.add_argument("--mlflow-experiment", type=str, default="lgxbenchmark",
                        help="MLflow 实验名称（默认：lgxbenchmark）")
    args = parser.parse_args()

    # 读取 .env 中的 LLM_MODEL 配置
    settings = get_settings()
    llm_model = settings.llm_model
    logger.info(f"📌 当前 LLM 模型: {llm_model}")

    # 解析 K 值
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # 确定输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "output" / args.dataset_name / args.strategy / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 将所有日志同时写入 output_dir/run.log ─────────────────
    log_file = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    # 挂到根 logger，确保所有子 logger 的日志都能落盘
    logging.getLogger().addHandler(file_handler)
    logger.info(f"📄 日志文件: {log_file}")

    # bench_logger：使用 logger（与 benchmark.py 一致用 logging 而非 print）
    bench_logger = logging.getLogger("scripts.run_search_benchmark")

    # ── 打印启动信息 ──────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("🔍 检索信息概览")
    logger.info("=" * 80)
    logger.info(f"📊 数据集名称: {args.dataset_name}")
    logger.info(f"🔧 检索策略: {args.strategy}")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  K值列表: {k_values}")
    logger.info(f"  并发数: {args.max_concurrency}")
    logger.info(f"  Bench size: {args.bench_size}")

    # ── 解析 --limit 参数 ──────────────────────────────────────
    limit_start: Optional[int] = None
    limit_end: Optional[int] = None   # 切片用的 stop（exclusive）
    if args.limit:
        if len(args.limit) == 1:
            limit_end = args.limit[0]
            logger.info(f"  限制条数: 前 {limit_end} 条（索引 0~{limit_end-1}）")
        elif len(args.limit) == 2:
            limit_start, limit_end_incl = args.limit[0], args.limit[1]
            if limit_start > limit_end_incl:
                logger.error(f"--limit 起始索引 {limit_start} > 结束索引 {limit_end_incl}，请检查参数")
                sys.exit(1)
            limit_end = limit_end_incl + 1   # 转为 exclusive
            logger.info(f"  限制范围: 第 {limit_start} 条 ~ 第 {limit_end_incl} 条（共 {limit_end - limit_start} 条）")
        else:
            logger.error("--limit 最多接受两个整数")
            sys.exit(1)

    logger.info(f"  输出目录: {output_dir}")
    logger.info("\n" + "=" * 80)

    # ── 加载数据集 ────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset_loader = DatasetLoader(args.dataset_name)
    dataset_info = dataset_loader.load_all()

    questions: List[str] = dataset_info["questions"]
    if args.limit:
        questions = questions[limit_start:limit_end]

    gold_docs_for_recall = dataset_loader.get_gold_docs_for_recall(limit=None)
    if gold_docs_for_recall and args.limit:
        gold_docs_for_recall = gold_docs_for_recall[limit_start:limit_end]

    logger.info(
        f"Successfully loaded dataset with {dataset_info['total_questions']} questions"
    )
    logger.info(f"❓ 问题总数: {len(questions)}")
    logger.info(f"📋 数据范围: [0:{len(questions)}]，共 {len(questions)} 个问题:")
    logger.info("\n" + "=" * 80)

    # ── 获取 source_config_id ─────────────────────────────────
    if args.source_config_id:
        source_config_id = args.source_config_id
        source_timestamp = "manual"
        logger.info(f"📦 使用数据源（手动指定）: {source_config_id}")
    else:
        try:
            source_info = load_latest_source_info(args.dataset_name, llm_model)
            source_config_id = source_info["source_config_id"]
            source_timestamp = source_info.get("timestamp", "unknown")

            # 增强日志输出，明确显示使用的数据源信息
            logger.info(f"📦 使用数据源: {source_config_id}")
            logger.info(f"   模型: {source_info.get('model_name', 'unknown')}")
            logger.info(f"   时间戳: {source_timestamp}")
            logger.info(f"   文件路径: {source_info.get('file_path', 'unknown')}")
        except Exception as e:
            logger.error(f"无法获取 source_config_id: {e}")
            sys.exit(1)

    # ── 初始化 MLflow ─────────────────────────────────────────
    mlflow_tracker = None
    if args.use_mlflow:
        try:
            # 如果未指定 --mlflow-url，则从 .env 读取 MLFLOW_URL
            mlflow_url = args.mlflow_url or settings.mlflow_url
            logger.info(f"📊 MLflow Tracking Server: {mlflow_url}")

            mlflow_config = MLflowConfig(
                uri=mlflow_url,
                experiment=f"{get_local_ip()}_{args.mlflow_experiment}",
                dataset_name=args.dataset_name,
                bench_size=args.bench_size,
                enable_qa=False,
            )
            mlflow_tracker = MLflowTracker(mlflow_config, questions)
            mlflow_tracker.start()

            # ── 记录运行命令 + 脚本参数 ──────────────────────────
            import mlflow, shlex
            cmd_parts = ["uv run python scripts/run_search_benchmark.py",
                f"--dataset-name {args.dataset_name}",
                f"--strategy {args.strategy}",
                f"--top-k {args.top_k}",
                f"--k-values \"{args.k_values}\"",
                f"--max-concurrency {args.max_concurrency}",
                f"--bench-size {args.bench_size}",
            ]
            if args.limit:
                cmd_parts.append(f"--limit {' '.join(str(x) for x in args.limit)}")
            if args.output_dir:
                cmd_parts.append(f"--output-dir {args.output_dir}")
            if args.source_config_id:
                cmd_parts.append(f"--source-config-id {args.source_config_id}")
            if args.use_mlflow:
                cmd_parts += [
                    "--use-mlflow",
                    f"--mlflow-url {args.mlflow_url}",
                    f"--mlflow-experiment {args.mlflow_experiment}",
                ]
            mlflow.log_param("run_command", " \\\n    ".join(cmd_parts))

            # ── 记录策略实际生效的默认配置 ────────────────────────
            from pipeline.modules.search.config import MultiConfig, AtomicConfig, VectorConfig
            if args.strategy == "multi":
                cfg = MultiConfig(strategy="multi", max_sections=args.top_k)
                strategy_defaults = {
                    "strategy": "multi",
                    "entity_top_k": cfg.entity_top_k,
                    "multi_top_k": cfg.multi_top_k,
                    "key_similarity_threshold": cfg.key_similarity_threshold,
                    "similarity_threshold": cfg.similarity_threshold,
                    "max_hops": cfg.max_hops,
                    "max_events": cfg.max_events,
                    "rerank_top_k": cfg.rerank_top_k,
                    "max_sections": cfg.max_sections,
                }
            elif args.strategy == "multi1":
                cfg = MultiConfig(strategy="multi1", max_sections=args.top_k)
                strategy_defaults = {
                    "strategy": "multi1",
                    "entity_top_k": cfg.entity_top_k,
                    "multi_top_k": cfg.multi_top_k,
                    "key_similarity_threshold": cfg.key_similarity_threshold,
                    "similarity_threshold": cfg.similarity_threshold,
                    "max_events_a": cfg.max_events_a,
                    "max_events_b": cfg.max_events_b,
                    "max_hop_retries": cfg.max_hop_retries,
                    "rerank_top_k": cfg.rerank_top_k,
                    "max_sections": cfg.max_sections,
                }
            elif args.strategy == "hopllm":
                cfg = MultiConfig(strategy="hopllm", max_sections=args.top_k)
                strategy_defaults = {
                    "strategy": "hopllm",
                    "entity_top_k": cfg.entity_top_k,
                    "multi_top_k": cfg.multi_top_k,
                    "key_similarity_threshold": cfg.key_similarity_threshold,
                    "similarity_threshold": cfg.similarity_threshold,
                    "max_events_a": cfg.max_events_a,
                    "max_events_b": cfg.max_events_b,
                    "max_hop_retries": cfg.max_hop_retries,
                    "rerank_top_k": cfg.rerank_top_k,
                    "max_sections": cfg.max_sections,
                }
            elif args.strategy == "atomic":
                cfg = AtomicConfig(max_sections=args.top_k)
                strategy_defaults = {
                    "strategy": "atomic",
                    "entity_top_k": cfg.entity_top_k,
                    "atomic_top_k": cfg.atomic_top_k,
                    "key_similarity_threshold": cfg.key_similarity_threshold,
                    "similarity_threshold": cfg.similarity_threshold,
                    "max_hops": cfg.max_hops,
                    "max_events": cfg.max_events,
                    "rerank_top_k": cfg.rerank_top_k,
                    "max_sections": cfg.max_sections,
                }
            else:  # vector
                cfg = VectorConfig(top_k=args.top_k)
                strategy_defaults = {
                    "strategy": "vector",
                    "top_k": cfg.top_k,
                    "title_weight": cfg.title_weight,
                    "content_weight": cfg.content_weight,
                    "similarity_threshold": cfg.similarity_threshold,
                }
            mlflow.log_params({f"cfg_{k}": v for k, v in strategy_defaults.items()})

            # ── 记录模型和数据源信息 ──────────────────────────────
            mlflow.log_param("llm_model", llm_model)
            mlflow.log_param("source_config_id", source_config_id)
            mlflow.log_param("source_timestamp", source_timestamp)

            logger.info("✅ MLflow 追踪器初始化完成")
        except Exception as e:
            logger.warning(f"MLflow 初始化失败: {e}")
            mlflow_tracker = None

    # ── 启用 LLM token 追踪 ───────────────────────────────────
    token_tracker = LLMTokenTracker()
    enable_llm_tracking(token_tracker)
    logger.info("✅ LLM token 追踪已启用")

    # ── 执行搜索 ──────────────────────────────────────────────
    logger.info(f"\n🚀 启动检索 (策略: {args.strategy})...")
    logger.info("=" * 80)

    search_start = time.perf_counter()
    search_results = await run_batch_search(
        questions=questions,
        source_config_id=source_config_id,
        strategy=args.strategy,
        top_k=args.top_k,
        max_concurrency=args.max_concurrency,
        bench_size=args.bench_size,
        gold_docs_for_recall=gold_docs_for_recall,
        mlflow_tracker=mlflow_tracker,
        bench_logger=bench_logger,
    )
    search_time = time.perf_counter() - search_start

    # ── 保存搜索原始结果 ──────────────────────────────────────
    search_output = output_dir / "search_results.json"
    serializable = [
        {
            "question_index": r["question_index"],
            "question": r["question"],
            "retrieved_docs": r["sections"],
        }
        for r in search_results
    ]
    with open(search_output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"\n💾 搜索结果已保存: {search_output}")

    # ── 打印最终汇总（与 benchmark.py 完全一致）──────────────
    metrics = print_final_summary(
        dataset_name=args.dataset_name,
        strategy=args.strategy,
        questions=questions,
        search_results=search_results,
        gold_docs_for_recall=gold_docs_for_recall,
        k_values=k_values,
        search_time=search_time,
        bench_logger=bench_logger,
    )

    # ── 保存 benchmark 结果 ───────────────────────────────────
    token_summary = token_tracker.get_summary()
    benchmark_output = output_dir / "benchmark_results.json"
    with open(benchmark_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": {**metrics["recall"], **metrics["precision_f1"]},
                "statistics": metrics["statistics"],
                "llm_token_usage": token_summary,
                "metadata": {
                    "dataset_name": args.dataset_name,
                    "strategy": args.strategy,
                    "top_k": args.top_k,
                    "k_values": k_values,
                    "total_questions": len(search_results),
                    "search_time_seconds": round(search_time, 2),
                    "timestamp": timestamp,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"💾 评估结果已保存: {benchmark_output}")

    # ── MLflow 最终指标上报 ────────────────────────────────────
    if mlflow_tracker:
        try:
            import mlflow
            all_metrics = {**metrics["recall"], **metrics["precision_f1"]}

            # 计算最终的 batch_index，与中间记录保持一致
            final_batch_index = (len(search_results) + args.bench_size - 1) // args.bench_size

            mlflow_tracker.log_recall_metrics(all_metrics, step=final_batch_index)
            stats = metrics["statistics"]
            mlflow_tracker.log_evaluation_metrics(
                stats["full_recall_count"],
                stats["partial_recall_count"],
                stats["zero_recall_count"],
                len(search_results),
                step=final_batch_index,
            )
            mlflow.log_metric("search_time_seconds", search_time)
            mlflow.log_param("output_dir", str(output_dir))
            logger.info("✓ MLflow metrics/params 记录完成")
        except Exception as e:
            logger.warning(f"MLflow 记录失败: {e}")
        finally:
            mlflow_tracker.end()

    logger.info(f"\n✅ 完成！总耗时: {search_time:.1f} 秒")


if __name__ == "__main__":
    # 配置日志（与 benchmark.py 完全一致）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
