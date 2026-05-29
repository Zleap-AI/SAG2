"""
搜索器 - 统一入口

提供三种搜索方式：
1. VECTOR  - 纯向量搜索（跳过 Recall/Expand，直接向量检索段落）
2. ATOMIC  - 原子事项检索（三元组 + LLM 精选）
3. MULTI   - 多元事项检索（多实体 + 多跳扩展 + LLM 精选）
            通过 MultiConfig.strategy 参数支持三种子策略：
            - "multi": 固定跳数扩展
            - "multi1": 双阶段扩跳（全量种子）
            - "hopllm": 双阶段扩跳（粗排种子）
"""

import time
from typing import Dict, Any, Optional

from pipeline.core.prompt.manager import PromptManager
from pipeline.exceptions import SearchError
from pipeline.modules.search.config import SearchConfig, RerankStrategy, ReturnType, MultiConfig
from pipeline.modules.search.vector import VectorSearcher
from pipeline.modules.search.atomic import AtomicSearcher
from pipeline.modules.search.multi import MultiSearcher
from pipeline.utils import get_logger

logger = get_logger("search.searcher")


class SAGSearcher:
    """
    SAG搜索器 - 提供三种搜索方式

    1. VECTOR 模式：
       - 跳过 Recall/Expand，直接向量搜索段落
       - 仅支持 PARAGRAPH 返回类型

    2. ATOMIC 模式：
       - 原子事项检索（三元组 + LLM 精选）
       - 返回段落列表

    3. MULTI 模式：
       - 多元事项检索（多实体 + 多跳扩展 + LLM 精选）
       - 通过 MultiConfig.strategy 参数支持三种子策略：
         * "multi": 固定跳数扩展
         * "multi1": 双阶段扩跳（全量种子）
         * "hopllm": 双阶段扩跳（粗排种子）
       - 返回段落列表

    返回结果格式：
    {
        "sections": List[Dict],        # 段落列表
        "clues": List[Dict],           # 线索列表（支持前端图谱）
        "stats": Dict,                 # 统计信息
        "query": Dict                  # 查询信息
    }
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        model_config: Optional[Dict] = None,
    ):
        """
        初始化搜索器

        Args:
            prompt_manager: 提示词管理器
            model_config: LLM配置字典（可选）
        """
        self.prompt_manager = prompt_manager
        self.model_config = model_config
        self.logger = get_logger("search.sag")

        # 初始化三种搜索器
        self._vector_searcher = VectorSearcher()
        self._atomic_searcher = AtomicSearcher()
        self._multi_searcher = MultiSearcher()

        self.logger.info("SAG搜索器初始化完成")

    async def search(self, config: SearchConfig) -> Dict[str, Any]:
        """
        执行搜索

        Args:
            config: 搜索配置

        Returns:
            {
                "sections": List[Dict],        # 段落列表
                "clues": List[Dict],           # 线索列表
                "stats": Dict,                 # 统计信息
                "query": Dict                  # 查询信息
            }
        """
        try:
            total_start = time.perf_counter()

            # 打印配置参数
            self.logger.info("=" * 100)
            self.logger.info("📋 SAG搜索配置参数详情:")
            self.logger.info("=" * 100)
            self.logger.info("🔹 基础参数:")
            self.logger.info(f"  query: '{config.query}'")
            self.logger.info(f"  strategy: {config.rerank.strategy}")
            self.logger.info(f"  source_config_ids: {config.source_config_ids[:5] if config.source_config_ids else []}")
            self.logger.info(f"  return_type: {config.return_type}")
            self.logger.info("=" * 100)

            self.logger.info(
                f"🔍 开始搜索：query='{config.query}', strategy={config.rerank.strategy}"
            )

            # 根据策略选择搜索方式
            strategy = config.rerank.strategy

            # VECTOR 模式：纯向量搜索
            if strategy == RerankStrategy.VECTOR:
                if config.return_type != ReturnType.PARAGRAPH:
                    raise SearchError(
                        f"VECTOR 策略仅支持 PARAGRAPH 模式，当前为 {config.return_type}"
                    )

                self.logger.info("=" * 60)
                self.logger.info("【VECTOR 模式】跳过 Recall/Expand，直接向量搜索")
                self.logger.info("=" * 60)

                vector_start = time.perf_counter()
                rerank_result = await self._vector_searcher.search_chunks_for_rerank(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=config.strategy_config if config.strategy_config is not None else config,
                )
                vector_time = time.perf_counter() - vector_start
                total_time = time.perf_counter() - total_start

                response = {
                    "sections": rerank_result.get("sections", []),
                    "clues": [],
                    "stats": {
                        "vector": {
                            "sections_count": len(rerank_result.get("sections", [])),
                        },
                        "timing": {
                            "vector": vector_time,
                            "total": total_time,
                        },
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    f"✅ VECTOR 搜索完成：返回 {len(response['sections'])} 个段落，总耗时={total_time:.3f}s"
                )
                return response

            # ATOMIC 模式：原子事项检索
            elif strategy == RerankStrategy.ATOMIC:
                self.logger.info("=" * 60)
                self.logger.info("【ATOMIC 模式】原子事项检索")
                self.logger.info("=" * 60)

                atomic_start = time.perf_counter()
                rerank_result = await self._atomic_searcher.search_for_rerank(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=config.strategy_config if config.strategy_config is not None else config,
                )
                atomic_time = time.perf_counter() - atomic_start
                total_time = time.perf_counter() - total_start

                response = {
                    "sections": rerank_result.get("sections", []),
                    "clues": [],
                    "stats": {
                        "atomic": {
                            "sections_count": len(rerank_result.get("sections", [])),
                        },
                        "timing": {
                            "atomic": atomic_time,
                            "total": total_time,
                        },
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    f"✅ ATOMIC 搜索完成：返回 {len(response['sections'])} 个段落，总耗时={total_time:.3f}s"
                )
                return response

            # MULTI 模式：多元事项检索
            elif strategy == RerankStrategy.MULTI:
                self.logger.info("=" * 60)
                self.logger.info("【MULTI 模式】多元事项检索")
                self.logger.info("=" * 60)

                multi_start = time.perf_counter()
                rerank_result = await self._multi_searcher.search_for_rerank(
                    query=config.query,
                    source_config_ids=config.get_source_config_ids(),
                    config=config.strategy_config if config.strategy_config is not None else config,
                )
                multi_time = time.perf_counter() - multi_start
                total_time = time.perf_counter() - total_start

                response = {
                    "sections": rerank_result.get("sections", []),
                    "clues": [],
                    "stats": {
                        "multi": {
                            "sections_count": len(rerank_result.get("sections", [])),
                        },
                        "timing": {
                            "multi": multi_time,
                            "total": total_time,
                        },
                    },
                    "query": {
                        "original": config.original_query or config.query,
                        "current": config.query,
                        "rewritten": False,
                    },
                }

                self.logger.info(
                    f"✅ MULTI 搜索完成：返回 {len(response['sections'])} 个段落，总耗时={total_time:.3f}s"
                )
                return response

            else:
                raise SearchError(f"不支持的搜索策略: {strategy}")

        except Exception as e:
            self.logger.error(f"❌ 搜索失败: {e}", exc_info=True)
            raise SearchError(f"搜索失败: {e}") from e


# 向后兼容别名
EventSearcher = SAGSearcher

__all__ = [
    "SAGSearcher",
    "EventSearcher",
]
