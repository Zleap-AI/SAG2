"""
Benchmark 评估器

用于评估搜索结果的质量，支持多种评估指标：
- Precision@K: 前K个结果中相关文档的比例
- Recall@K: 前K个结果中召回的相关文档占所有相关文档的比例
- NDCG@K: 归一化折损累积增益，考虑排序位置
- MRR: 平均倒数排名，第一个相关结果的排名倒数

使用示例：
    from pipeline.evaluation import Evaluator

    evaluator = Evaluator()
    results = evaluator.evaluate(
        predictions=[
            {"query_id": "q1", "retrieved_chunks": ["chunk1", "chunk2", "chunk3"]},
            {"query_id": "q2", "retrieved_chunks": ["chunk4", "chunk5"]},
        ],
        ground_truth={
            "q1": ["chunk1", "chunk3", "chunk5"],
            "q2": ["chunk4", "chunk6"],
        },
        k_values=[1, 3, 5, 10]
    )
    print(results)  # {"precision@1": 0.5, "recall@5": 0.6, "ndcg@10": 0.7, "mrr": 0.75}
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pipeline.utils import get_logger

logger = get_logger("evaluation.evaluator")


class Evaluator:
    """搜索结果评估器"""

    def __init__(self):
        pass

    @staticmethod
    def _precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        计算 Precision@K

        Args:
            retrieved: 检索结果列表（按排序）
            relevant: 相关文档集合
            k: 截断位置

        Returns:
            Precision@K 分数 [0, 1]
        """
        if k <= 0 or not retrieved:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_count = sum(1 for doc in retrieved_at_k if doc in relevant)
        return relevant_count / k

    @staticmethod
    def _recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        计算 Recall@K

        Args:
            retrieved: 检索结果列表（按排序）
            relevant: 相关文档集合
            k: 截断位置

        Returns:
            Recall@K 分数 [0, 1]
        """
        if not relevant or k <= 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_count = sum(1 for doc in retrieved_at_k if doc in relevant)
        return relevant_count / len(relevant)

    @staticmethod
    def _dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        计算 DCG@K (Discounted Cumulative Gain)

        Args:
            retrieved: 检索结果列表（按排序）
            relevant: 相关文档集合
            k: 截断位置

        Returns:
            DCG@K 分数
        """
        if k <= 0 or not retrieved:
            return 0.0

        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                # rel = 1 if relevant, 0 otherwise
                # DCG = sum(rel_i / log2(i+2))  # i+2 because i starts from 0
                dcg += 1.0 / math.log2(i + 2)
        return dcg

    @staticmethod
    def _idcg_at_k(relevant: Set[str], k: int) -> float:
        """
        计算 IDCG@K (Ideal DCG)

        Args:
            relevant: 相关文档集合
            k: 截断位置

        Returns:
            IDCG@K 分数
        """
        if not relevant or k <= 0:
            return 0.0

        # 理想情况：所有相关文档都在前面
        ideal_k = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
        return idcg

    def _ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        计算 NDCG@K (Normalized DCG)

        Args:
            retrieved: 检索结果列表（按排序）
            relevant: 相关文档集合
            k: 截断位置

        Returns:
            NDCG@K 分数 [0, 1]
        """
        dcg = self._dcg_at_k(retrieved, relevant, k)
        idcg = self._idcg_at_k(relevant, k)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _mrr(retrieved: List[str], relevant: Set[str]) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)

        Args:
            retrieved: 检索结果列表（按排序）
            relevant: 相关文档集合

        Returns:
            RR 分数（单个查询的倒数排名）
        """
        if not retrieved or not relevant:
            return 0.0

        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_single_query(
        self,
        retrieved: List[str],
        relevant: List[str],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        评估单个查询的结果

        Args:
            retrieved: 检索结果列表（按排序）
            relevant: 相关文档列表
            k_values: 评估的K值列表（默认 [1, 3, 5, 10]）

        Returns:
            评估指标字典
        """
        k_values = k_values or [1, 3, 5, 10]
        relevant_set = set(relevant)

        metrics = {}

        # 计算各个 K 值的指标
        for k in k_values:
            metrics[f"precision@{k}"] = self._precision_at_k(retrieved, relevant_set, k)
            metrics[f"recall@{k}"] = self._recall_at_k(retrieved, relevant_set, k)
            metrics[f"ndcg@{k}"] = self._ndcg_at_k(retrieved, relevant_set, k)

        # MRR 不依赖 K
        metrics["mrr"] = self._mrr(retrieved, relevant_set)

        return metrics

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: Dict[str, List[str]],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        评估多个查询的结果（宏平均）

        Args:
            predictions: 预测结果列表
                [
                    {"query_id": "q1", "retrieved_chunks": ["chunk1", "chunk2", ...]},
                    {"query_id": "q2", "retrieved_chunks": ["chunk3", ...]},
                ]
            ground_truth: 真实相关文档
                {
                    "q1": ["chunk1", "chunk3", "chunk5"],
                    "q2": ["chunk4", "chunk6"],
                }
            k_values: 评估的K值列表（默认 [1, 3, 5, 10]）

        Returns:
            平均评估指标字典
        """
        k_values = k_values or [1, 3, 5, 10]

        if not predictions:
            logger.warning("预测结果为空，返回零分")
            return {f"precision@{k}": 0.0 for k in k_values} | \
                   {f"recall@{k}": 0.0 for k in k_values} | \
                   {f"ndcg@{k}": 0.0 for k in k_values} | \
                   {"mrr": 0.0}

        # 累积各个查询的指标
        all_metrics: Dict[str, List[float]] = {}

        for pred in predictions:
            query_id = pred.get("query_id", "")
            retrieved = pred.get("retrieved_chunks", [])

            if query_id not in ground_truth:
                logger.warning(f"查询 {query_id} 不在 ground_truth 中，跳过")
                continue

            relevant = ground_truth[query_id]
            query_metrics = self.evaluate_single_query(retrieved, relevant, k_values)

            for metric_name, value in query_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # 计算宏平均
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[metric_name] = sum(values) / len(values) if values else 0.0

        return avg_metrics

    def save_results(
        self,
        metrics: Dict[str, float],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        保存评估结果到文件

        Args:
            metrics: 评估指标字典
            output_path: 输出文件路径（JSON 格式）
            metadata: 额外元数据（如 strategy, dataset, timestamp 等）
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metrics": metrics,
            "metadata": metadata or {},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ 评估结果已保存: {output_file}")


__all__ = ["Evaluator"]
