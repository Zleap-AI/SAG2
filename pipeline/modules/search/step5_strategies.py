"""
Step5 扩展策略

提供三种多跳扩展策略，通过策略模式与 UnifiedMultiSearcher 解耦：

- MultiStep5Strategy:   单阶段固定跳数扩展（对应原 multi.py）
- Multi1Step5Strategy:  双阶段扩展，阶段B以 hop1 全量事项实体为种子（对应原 multi1.py）
- HopLLMStep5Strategy:  双阶段扩展，阶段B以粗排后事项实体为种子（对应原 hopllm.py）

核心差异（阶段B种子选择）：
- Multi1: seed = hop1 所有事项实体（广度优先，覆盖面广）
- HopLLM: seed = eventset 经 Step6 粗排后 top-N 事项实体（质量优先，精准度高）
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from pipeline.db import EventEntity, SourceEvent, get_session_factory

if TYPE_CHECKING:
    from pipeline.modules.search.multi import MultiSearcher
    from pipeline.modules.search.config import MultiConfig

logger = logging.getLogger("search.step5_strategies")


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------

class Step5Strategy(ABC):
    """Step5 多跳扩展策略抽象基类"""

    @abstractmethod
    async def expand(
        self,
        searcher: "MultiSearcher",
        event_entities: Dict[str, List[str]],
        source_config_ids: Optional[List[str]],
        config: Any,
        query: str = "",
    ) -> Dict[str, Any]:
        """
        执行扩展策略。

        Args:
            searcher:         MultiSearcher 实例（共享 _entity_ids / _relation_ids）
            event_entities:   Step4 返回的 {event_id: [entity_id, ...]}（hop0 事件的实体）
            source_config_ids: 信息源 ID 列表
            config:           策略对应的配置对象（MultiConfig）
            query:            原始查询文本（HopLLM 阶段B种子粗排时需要）

        Returns:
            {
                "eventset_details":  {event_id: {"title": str, "content": str}},  # hop0+hop1
                "eventset_entities": {event_id: [entity_id, ...]},                 # hop0+hop1
                "eventset1_details": {event_id: {"title": str, "content": str}},  # hop2+（可能为空）
                "eventset1_entities":{event_id: [entity_id, ...]},                 # hop2+（可能为空）
            }
        """

    # ------------------------------------------------------------------
    # 通用工具方法（供子类复用）
    # ------------------------------------------------------------------

    async def _query_new_event_ids(
        self,
        entity_ids: List[str],
        exclude_ids: set,
        source_config_ids: Optional[List[str]],
    ) -> List[str]:
        """
        根据 entity_ids 查询关联的新事项 ID（不在 exclude_ids 中）。

        Args:
            entity_ids:        实体 ID 列表
            exclude_ids:       已存在的事项 ID 集合（去重用）
            source_config_ids: 信息源 ID 过滤

        Returns:
            新事项 ID 列表
        """
        new_event_ids: List[str] = []
        session_factory = get_session_factory()
        async with session_factory() as session:
            stmt = select(EventEntity.event_id).where(
                EventEntity.entity_id.in_(entity_ids)
            ).distinct()
            if source_config_ids:
                stmt = stmt.join(
                    SourceEvent, SourceEvent.id == EventEntity.event_id
                ).where(
                    SourceEvent.source_config_id.in_(source_config_ids)
                )
            result = await session.execute(stmt)
            for row in result.fetchall():
                if row[0] not in exclude_ids:
                    new_event_ids.append(row[0])
        return new_event_ids


# ---------------------------------------------------------------------------
# 策略一：MultiStep5Strategy（固定跳数扩展）
# ---------------------------------------------------------------------------

class MultiStep5Strategy(Step5Strategy):
    """
    multi 策略：单阶段固定跳数扩展。

    逻辑：
      hop=0: entity_set = step2 实体，relation_set = step3 合并事件
      hop=N: 上一跳 events 的新实体 → 新 events（不在 relation_set 中）
             更新两个 set，直到无新实体/事项或达到 max_hops
    """

    async def expand(
        self,
        searcher: "MultiSearcher",
        event_entities: Dict[str, List[str]],
        source_config_ids: Optional[List[str]],
        config: Any,
        query: str = "",
    ) -> Dict[str, Any]:
        max_hops = getattr(config, "max_hops", 1)

        all_details: Dict[str, Dict[str, str]] = {}
        all_entities: Dict[str, List[str]] = {}

        # hop=0: 初始化 relation_set（entity_set 已由 step2 填充）
        searcher._relation_ids.update(event_entities.keys())

        if max_hops == 0:
            return {
                "eventset_details": all_details,
                "eventset_entities": all_entities,
                "eventset1_details": {},
                "eventset1_entities": {},
            }

        prev_hop_entities = event_entities

        for hop in range(max_hops):
            pre_events = len(searcher._relation_ids)
            pre_entities = len(searcher._entity_ids)

            new_entity_ids = searcher.get_new_entity_ids(prev_hop_entities)
            if not new_entity_ids:
                logger.info(
                    f"[Step5-Multi] hop={hop+1}/{max_hops} "
                    f"无新实体 (tracked_entities={len(searcher._entity_ids)})，停止"
                )
                break

            searcher._entity_ids.update(new_entity_ids)
            logger.info(
                f"[Step5-Multi] hop={hop+1}/{max_hops} "
                f"entities: {pre_entities} -> +{len(new_entity_ids)} new, total={len(searcher._entity_ids)}"
            )

            new_event_ids = await self._query_new_event_ids(
                new_entity_ids, searcher._relation_ids, source_config_ids
            )

            if not new_event_ids:
                logger.info(
                    f"[Step5-Multi] hop={hop+1}/{max_hops} "
                    f"无新事项 (tracked_events={len(searcher._relation_ids)})，停止"
                )
                break

            hop_details, hop_entities = await searcher.step4_fetch_event_details(new_event_ids)
            searcher._relation_ids.update(new_event_ids)
            all_details.update(hop_details)
            all_entities.update(hop_entities)
            prev_hop_entities = hop_entities

            logger.info(
                f"[Step5-Multi] hop={hop+1}/{max_hops} done: "
                f"events {pre_events} -> {len(searcher._relation_ids)} (+{len(new_event_ids)}), "
                f"entities {pre_entities} -> {len(searcher._entity_ids)}"
            )

        return {
            "eventset_details": all_details,
            "eventset_entities": all_entities,
            "eventset1_details": {},
            "eventset1_entities": {},
        }


# ---------------------------------------------------------------------------
# 策略二：Multi1Step5Strategy（双阶段 - 全量种子）
# ---------------------------------------------------------------------------

class Multi1Step5Strategy(Step5Strategy):
    """
    multi1 策略：双阶段扩展，阶段B以 hop1 全量事项实体为种子。

    阶段A（固定1跳）：生成 eventset（hop0 + hop1，去重）
    阶段B（动态扩跳）：从 hop1 所有事项实体继续扩跳，生成 eventset1（hop2+，与 eventset 无交集）
                       直到 len(eventset1) >= max_events_b 或达到 max_hop_retries

    不足 max_events_b 且耗尽重试时抛出 RuntimeError。
    """

    async def expand(
        self,
        searcher: "MultiSearcher",
        event_entities: Dict[str, List[str]],
        source_config_ids: Optional[List[str]],
        config: Any,
        query: str = "",
    ) -> Dict[str, Any]:
        max_events_b = getattr(config, "max_events_b", 0)
        max_hop_retries = getattr(config, "max_hop_retries", 3)

        # ==== 阶段A：固定1跳 ====
        eventset_details, eventset_entities = await self._expand_phase_a(
            searcher, event_entities, source_config_ids
        )

        # ==== 阶段B：以 hop1 全量事项实体为种子动态扩跳 ====
        # 【关键】使用 hop1 所有事项的实体（广度优先，无筛选）
        eventset1_details, eventset1_entities = await self._expand_phase_b(
            searcher=searcher,
            seed_event_entities=eventset_entities,  # hop1 全量事项实体
            source_config_ids=source_config_ids,
            max_events_b=max_events_b,
            max_hop_retries=max_hop_retries,
            raise_on_limit=False,  # 超出上限时只 warning，不抛异常，以当前结果继续
        )

        return {
            "eventset_details": eventset_details,
            "eventset_entities": eventset_entities,
            "eventset1_details": eventset1_details,
            "eventset1_entities": eventset1_entities,
        }

    async def _expand_phase_a(
        self,
        searcher: "UnifiedMultiSearcher",
        event_entities: Dict[str, List[str]],
        source_config_ids: Optional[List[str]],
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
        """阶段A：固定1跳，生成 eventset（hop0 + hop1）"""
        # hop=0: 初始化 relation_set
        searcher._relation_ids.update(event_entities.keys())

        eventset_details: Dict[str, Dict[str, str]] = {}
        eventset_entities: Dict[str, List[str]] = {}

        pre_events_a = len(searcher._relation_ids)
        pre_entities_a = len(searcher._entity_ids)

        new_entity_ids_hop1 = searcher.get_new_entity_ids(event_entities)

        if not new_entity_ids_hop1:
            logger.info(
                f"[Step5-阶段A] hop=1 无新实体 (tracked_entities={len(searcher._entity_ids)})，"
                "阶段A提前结束，eventset 仅含 hop0"
            )
            return eventset_details, eventset_entities

        searcher._entity_ids.update(new_entity_ids_hop1)

        hop1_event_ids = await self._query_new_event_ids(
            new_entity_ids_hop1, searcher._relation_ids, source_config_ids
        )

        if not hop1_event_ids:
            logger.info(
                f"[Step5-阶段A] hop=1 无新事项 (tracked_events={len(searcher._relation_ids)})，"
                "阶段A提前结束，eventset 仅含 hop0"
            )
            return eventset_details, eventset_entities

        hop1_details, hop1_entities = await searcher.step4_fetch_event_details(hop1_event_ids)
        searcher._relation_ids.update(hop1_event_ids)
        eventset_details.update(hop1_details)
        eventset_entities.update(hop1_entities)

        logger.info(
            f"[Step5-阶段A] hop=1 done: "
            f"events {pre_events_a} -> {len(searcher._relation_ids)} (+{len(hop1_event_ids)}), "
            f"entities {pre_entities_a} -> {len(searcher._entity_ids)}"
        )
        return eventset_details, eventset_entities

    async def _expand_phase_b(
        self,
        searcher: "UnifiedMultiSearcher",
        seed_event_entities: Dict[str, List[str]],
        source_config_ids: Optional[List[str]],
        max_events_b: int,
        max_hop_retries: int,
        raise_on_limit: bool = False,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
        """
        阶段B：以指定事项实体为种子，动态扩跳生成 eventset1（hop2+）。

        与 eventset（self._relation_ids）保证无交集。

        Args:
            seed_event_entities: 扩跳起点实体字典
            raise_on_limit:      （保留参数，当前两种策略均不抛异常）True/False 均只 warning 继续
        """
        eventset1_details: Dict[str, Dict[str, str]] = {}
        eventset1_entities: Dict[str, List[str]] = {}

        if not seed_event_entities:
            logger.info("[Step5-阶段B] 种子为空，跳过阶段B，eventset1 为空")
            return eventset1_details, eventset1_entities

        if max_events_b == 0:
            logger.info("[Step5-阶段B] max_events_b=0，跳过阶段B，eventset1 为空")
            return eventset1_details, eventset1_entities

        retry = 0
        hop_num = 2
        cur_hop_entities = seed_event_entities

        logger.info(
            f"[Step5-阶段B] 开始动态扩跳，目标 eventset1 >= {max_events_b}，"
            f"最大重试 {max_hop_retries} 跳"
        )

        while retry < max_hop_retries:
            pre_entities_b = len(searcher._entity_ids)

            logger.info(
                f"[Step5-阶段B] ── hop={hop_num}（第 {retry+1}/{max_hop_retries} 次重试）开始 "
                f"| eventset1 当前={len(eventset1_details)} / 目标={max_events_b} "
                f"| 上跳携带事项数={len(cur_hop_entities)}"
            )

            new_entity_ids = searcher.get_new_entity_ids(cur_hop_entities)

            if not new_entity_ids:
                logger.info(
                    f"[Step5-阶段B] hop={hop_num} 无新实体（已跟踪实体={len(searcher._entity_ids)}），"
                    f"自然收敛，终止扩跳（共完成 {retry} 跳，eventset1={len(eventset1_details)} 条）"
                )
                break

            logger.info(
                f"[Step5-阶段B] hop={hop_num} 发现新实体 +{len(new_entity_ids)} 个 "
                f"（实体总量 {pre_entities_b} → {pre_entities_b + len(new_entity_ids)}）"
            )
            searcher._entity_ids.update(new_entity_ids)

            # 排除已在 eventset（_relation_ids）及 eventset1 中的事项
            exclude = searcher._relation_ids | set(eventset1_details.keys())
            new_event_ids = await self._query_new_event_ids(
                new_entity_ids, exclude, source_config_ids
            )

            if not new_event_ids:
                logger.info(
                    f"[Step5-阶段B] hop={hop_num} 新实体未关联任何新事项，"
                    f"自然收敛，终止扩跳（共完成 {retry} 跳，eventset1={len(eventset1_details)} 条）"
                )
                break

            hop_details, hop_entities = await searcher.step4_fetch_event_details(new_event_ids)
            # 注意：阶段B的事项不加入 _relation_ids（保持 eventset 边界清晰）
            eventset1_details.update(hop_details)
            eventset1_entities.update(hop_entities)
            cur_hop_entities = hop_entities

            logger.info(
                f"[Step5-阶段B] hop={hop_num} 完成 "
                f"| 本跳新增事项 +{len(new_event_ids)} 条 "
                f"| eventset1 累计 {len(eventset1_details)} 条 "
                f"| 目标 {max_events_b}，"
                f"{'✓ 已满足' if len(eventset1_details) >= max_events_b else f'还差 {max_events_b - len(eventset1_details)} 条'}"
            )

            if len(eventset1_details) >= max_events_b:
                logger.info(
                    f"[Step5-阶段B] eventset1={len(eventset1_details)} >= max_events_b={max_events_b}，"
                    f"目标达成，共扩跳 {retry+1} 跳（hop2~hop{hop_num}），终止"
                )
                break

            retry += 1
            hop_num += 1

        else:
            # while 正常耗尽（未 break），说明达到重试上限仍不足，只 warning 继续
            msg = (
                f"[Step5-阶段B] 已执行 {max_hop_retries} 次扩跳（hop2~hop{hop_num}），"
                f"eventset1 仍不足 max_events_b（当前={len(eventset1_details)}，目标={max_events_b}）"
            )
            logger.warning(msg + "，达到上限，以当前结果继续后续步骤")

        logger.info(
            f"[Step5-阶段B] 结束 | eventset1 最终={len(eventset1_details)} 条"
        )
        return eventset1_details, eventset1_entities


# ---------------------------------------------------------------------------
# 策略三：HopLLMStep5Strategy（双阶段 - 粗排种子）
# ---------------------------------------------------------------------------

class HopLLMStep5Strategy(Step5Strategy):
    """
    hopllm 策略：双阶段扩展，阶段B以粗排后事项实体为种子。

    与 Multi1 的区别：
    - Multi1: seed = hop1 所有事项实体（广度优先）
    - HopLLM: 先对 eventset 做 Step6 粗排，seed = 粗排 top 事项的实体（质量优先）

    阶段A（固定1跳）：生成 eventset（hop0 + hop1）
    中间步骤：对 eventset 粗排，取 top 事项的实体作为阶段B的种子
    阶段B（动态扩跳）：从精选种子实体扩跳，生成 eventset1（hop2+）

    不足 max_events_b 且耗尽重试时只发出 warning，不抛 RuntimeError。
    """

    async def expand(
        self,
        searcher: "MultiSearcher",
        event_entities: Dict[str, List[str]],
        source_config_ids: Optional[List[str]],
        config: Any,
        query: str = "",
    ) -> Dict[str, Any]:
        max_events_a = getattr(config, "max_events_a", 100)
        max_events_b = getattr(config, "max_events_b", 0)
        max_hop_retries = getattr(config, "max_hop_retries", 3)

        # ==== 阶段A：固定1跳 ====
        # 复用 Multi1Step5Strategy 的阶段A逻辑（代码相同）
        _multi1_strategy = Multi1Step5Strategy()
        eventset_details, eventset_entities = await _multi1_strategy._expand_phase_a(
            searcher, event_entities, source_config_ids
        )

        # eventset = hop0（step3初始）+ hop1（阶段A）
        # 在这里调用者会把 event_details 和 eventset_details 合并后传入 step6
        # 我们需要让 searcher 的调用者能获取到完整的 eventset
        # 注意：eventset_details 仅含 hop1 新增部分，hop0 在 event_details 中

        # ==== 中间步骤：对 eventset 粗排，选出 top 事项作为阶段B的种子 ====
        # 需要 caller 传入完整的 eventset（hop0+hop1），通过 _relation_ids 获取
        all_eventset_ids = list(searcher._relation_ids)

        logger.info(
            f"[Step5-HopLLM中间粗排] 对 eventset（{len(all_eventset_ids)} 条）粗排，"
            f"选出 top-{max_events_a} 事项作为阶段B种子..."
        )
        ranked_es = await searcher.step6_coarse_rank(
            query=query,
            event_ids=all_eventset_ids,
            source_config_ids=source_config_ids,
            max_events=max_events_a,
        )
        ranked_event_ids = [item["event_id"] for item in ranked_es]

        # 查询粗排事项的实体（需要重新从DB查，因为 hop0 的实体不在 eventset_entities 中）
        logger.info(
            f"[Step5-HopLLM中间粗排] 查询粗排后 {len(ranked_event_ids)} 个事项的实体..."
        )
        _, seed_entities_for_b = await searcher.step4_fetch_event_details(ranked_event_ids)

        logger.info(
            f"[Step5-HopLLM中间粗排] 获得种子实体关联 "
            f"{sum(len(v) for v in seed_entities_for_b.values())} 个"
        )

        # ==== 阶段B：从粗排精选种子实体扩跳 ====
        # 【关键差异】使用粗排后的精选实体，而非 hop1 全量实体
        eventset1_details, eventset1_entities = await _multi1_strategy._expand_phase_b(
            searcher=searcher,
            seed_event_entities=seed_entities_for_b,  # 精选种子
            source_config_ids=source_config_ids,
            max_events_b=max_events_b,
            max_hop_retries=max_hop_retries,
            raise_on_limit=False,  # hopllm 超出上限时只 warning，不抛异常
        )

        return {
            "eventset_details": eventset_details,
            "eventset_entities": eventset_entities,
            "eventset1_details": eventset1_details,
            "eventset1_entities": eventset1_entities,
        }


__all__ = [
    "Step5Strategy",
    "MultiStep5Strategy",
    "Multi1Step5Strategy",
    "HopLLMStep5Strategy",
]
