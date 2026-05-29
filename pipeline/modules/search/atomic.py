"""
原子事项检索器

基于三元组（主体-关系-客体）的向量检索器。
检索恰好包含两个实体的原子事项，支持 title + content 混合搜索。

使用示例：
    from pipeline.modules.search import AtomicSearcher, AtomicConfig

    config = AtomicConfig(
        atomic_top_k=20,
        similarity_threshold=0.4
    )

    searcher = AtomicSearcher()
    results = await searcher.search(
        query="海尔集团人单合一模式",
        source_config_ids=["source_1", "source_2"],
        config=config
    )
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from pipeline.core.ai.factory import create_llm_client
from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.core.storage.elasticsearch import get_es_client
from pipeline.core.storage.repositories.entity_repository import EntityVectorRepository
from pipeline.core.storage.repositories.event_repository import EventVectorRepository
from pipeline.core.storage.repositories.source_chunk_repository import SourceChunkRepository
from pipeline.db import EventEntity, SourceChunk, SourceEvent, get_session_factory
from pipeline.modules.load.processor import DocumentProcessor
from pipeline.modules.search.config import AtomicConfig
from pipeline.utils import get_logger
from pipeline.utils.token_counter import TokenCounter

logger = get_logger("search.atomic")

# NER 提示词（参考 HippoRAG 风格）
_NER_SYSTEM_PROMPT = "You're a very effective entity extraction system."

_NER_ONE_SHOT_INPUT = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?
"""

_NER_ONE_SHOT_OUTPUT = """{"named_entities": ["First for Women", "Arthur's Magazine"]}"""

_NER_TEMPLATE = "Question: {}"

# Rerank 提示词（参考 HippoRAG 风格）
_RERANK_SYSTEM_PROMPT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly {top_k} relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of {top_k} relation lines, most useful first)."""

_RERANK_EXAMPLE_1_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
When did Lothair Ii's mother die?

Relationship descriptions:
[53] bertha married to theobald of arles
[54] bertha married to adalbert ii of tuscany
[42] lothair ii son of ermengarde of tours
[43] lothair ii married to teutberga
[41] lothair ii son of emperor lothair i
[60] lothair ii husband of waldrada
[67] waldrada was mistress of lothair ii
"""

_RERANK_EXAMPLE_1_OUTPUT = """{"thought_process": "2-hop question: First find Lothair II's mother (relation [42]: Ermengarde of Tours), \
then find death date. [41] gives father for family context.", \
"useful_relations": ["[42] lothair ii son of ermengarde of tours", "[41] lothair ii son of emperor lothair i", \
"[43] lothair ii married to teutberga", "[60] lothair ii husband of waldrada", "[67] waldrada was mistress of lothair ii"]}"""

_RERANK_EXAMPLE_2_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
What country is the composer of "Erta Eterna" from?

Relationship descriptions:
[12] terra eterna composed by paulo flores
[15] paulo flores born in angola
[18] paulo flores genre is semba
[22] angola located in africa
[25] semba originated in angola
[30] paulo flores nationality angolan
"""

_RERANK_EXAMPLE_2_OUTPUT = """{"thought_process": "2-hop question: First find composer of Terra Eterna ([12]: Paulo Flores), \
then find his country ([15] born in Angola or [30] nationality Angolan).", \
"useful_relations": ["[12] terra eterna composed by paulo flores", "[15] paulo flores born in angola", \
"[30] paulo flores nationality angolan", "[22] angola located in africa", "[25] semba originated in angola"]}"""

_RERANK_EXAMPLE_3_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
Who is the director of the film that won the award also won by "The Hurt Locker"?

Relationship descriptions:
[5] the hurt locker won academy award best picture
[8] the hurt locker directed by kathryn bigelow
[12] moonlight won academy award best picture
[15] moonlight directed by barry jenkins
[20] la la land won golden globe best musical
[25] barry jenkins born in miami
"""

_RERANK_EXAMPLE_3_OUTPUT = """{"thought_process": "3-hop question: (1) Find award won by The Hurt Locker ([5]: Academy Award Best Picture), \
(2) Find another film with same award ([12]: Moonlight), (3) Find director ([15]: Barry Jenkins).", \
"useful_relations": ["[5] the hurt locker won academy award best picture", "[12] moonlight won academy award best picture", \
"[15] moonlight directed by barry jenkins", "[8] the hurt locker directed by kathryn bigelow", "[25] barry jenkins born in miami"]}"""

_RERANK_TEMPLATE = """Question:
{question}

Relationship descriptions:
{relations}
"""


class AtomicSearcher:
    """
    原子事项检索器

    检索原子化三元组事项（每个事项恰好包含 2 个实体）。
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self._llm_client = None
        self._processor = None
        self._entity_repo = None
        self._entity_ids: set = set()
        self._relation_ids: set = set()
        self.token_counter = token_counter or TokenCounter()

    async def _get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = await create_llm_client(scenario="search")
        return self._llm_client

    def _get_entity_repo(self) -> EntityVectorRepository:
        if self._entity_repo is None:
            self._entity_repo = EntityVectorRepository(get_es_client())
        return self._entity_repo

    async def _get_processor(self) -> DocumentProcessor:
        if self._processor is None:
            llm_client = await self._get_llm_client()
            self._processor = DocumentProcessor(llm_client=llm_client)
        return self._processor

    async def step1_extract_entities(self, query: str) -> List[str]:
        """
        Step1: 从 query 中提取命名实体

        Args:
            query: 用户查询文本

        Returns:
            实体名称列表，如 ["海尔集团", "人单合一"]
        """
        llm_client = await self._get_llm_client()

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=_NER_SYSTEM_PROMPT),
            LLMMessage(role=LLMRole.USER, content=_NER_ONE_SHOT_INPUT),
            LLMMessage(role=LLMRole.ASSISTANT, content=_NER_ONE_SHOT_OUTPUT),
            LLMMessage(role=LLMRole.USER, content=_NER_TEMPLATE.format(query)),
        ]

        response = await llm_client.chat_with_schema(
            messages,
            response_schema={
                "type": "object",
                "properties": {
                    "named_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["named_entities"],
            },
        )

        # 记录 token 消耗
        if hasattr(response, "usage"):
            usage = response.usage
            self.token_counter.add_record(
                scenario="atomic_ner",
                model=getattr(response, "model", "unknown"),
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                metadata={"query": query},
            )

        entities = response.get("named_entities", response.get("entities", []))
        entities = [str(e).strip() for e in entities if e]

        logger.info(f"[Step1-实体提取] query='{query}' -> entities={entities}")
        return entities

    async def step2_retrieve_entities(
        self,
        query_entities: List[str],
        source_config_ids: List[str],
        entity_top_k: Optional[int] = None,
        key_similarity_threshold: Optional[float] = None,
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Step2: 根据 query 提取的实体名称，从 ES 向量检索相似实体

        每个查询实体最多找到 entity_top_k 个，相似度分数必须 >= key_similarity_threshold。

        Args:
            query_entities: Step1 提取的实体名称列表
            source_config_ids: 信息源 ID 列表
            entity_top_k: 每个查询实体检索的最大数量（默认取 AtomicConfig）
            key_similarity_threshold: 实体最低相似度阈值（默认取 AtomicConfig）

        Returns:
            (entity_ids, entity_names, scores) 去重后的三元组
        """
        if not query_entities:
            return [], [], []

        config = AtomicConfig()
        top_k = entity_top_k or config.entity_top_k
        threshold = key_similarity_threshold if key_similarity_threshold is not None else config.key_similarity_threshold

        processor = await self._get_processor()
        repo = self._get_entity_repo()

        # 批量生成 query 实体的向量
        embeddings = [await processor.generate_embedding(name) for name in query_entities]

        # 逐个实体做 kNN 搜索，聚合去重
        entity_ids: List[str] = []
        entity_names: List[str] = []
        scores: List[float] = []
        seen: set = set()

        for vec in embeddings:
            results = await repo.search_similar(
                query_vector=vec,
                k=top_k,
                source_config_ids=source_config_ids,
            )
            for hit in results:
                score = hit.get("_score", 0.0)
                if score < threshold:
                    continue
                eid = hit.get("entity_id", "")
                if eid and eid not in seen:
                    seen.add(eid)
                    entity_ids.append(eid)
                    entity_names.append(hit.get("name", ""))
                    scores.append(score)
                    self._entity_ids.add(eid)

        logger.info(
            f"[Step2-实体检索] query_entities={query_entities} -> "
            f"retrieved {len(entity_ids)} entities, "
            f"top_scores={scores[:5] if scores else []}"
        )
        return entity_ids, entity_names, scores

    async def step3_retrieve_events(
        self,
        query: str,
        source_config_ids: List[str],
        entity_ids: Optional[List[str]] = None,
        atomic_top_k: int = 20,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step3: 双通道召回 + 去重合并

        通道1 (entity→event): entity_ids → EventEntity（不限数量）
        通道2 (query→event): query embedding → content_vector kNN（上限 atomic_top_k）

        两个通道结果按 event_id 去重合并，仅返回 event_id 和 score。

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            entity_ids: Step2 检索到的实体 ID（可选，用于通道1）
            atomic_top_k: 通道2 query→event 最大数量
            similarity_threshold: 通道2 向量最低相似度阈值

        Returns:
            [{"event_id": str, "score": float}, ...]
        """
        config = AtomicConfig()
        threshold = similarity_threshold if similarity_threshold is not None else config.similarity_threshold

        merged: Dict[str, float] = {}

        # --- 通道1: entity → event（DB 查询，不限数量）---
        if entity_ids:
            session_factory = get_session_factory()

            async with session_factory() as session:
                stmt = select(EventEntity.event_id).where(
                    EventEntity.entity_id.in_(entity_ids)
                )
                if source_config_ids:
                    stmt = stmt.join(
                        SourceEvent, SourceEvent.id == EventEntity.event_id
                    ).where(
                        SourceEvent.source_config_id.in_(source_config_ids)
                    )
                result = await session.execute(stmt)
                for row in result.fetchall():
                    merged[row[0]] = 0.0

        # --- 通道2: query → event（ES 向量，上限 atomic_top_k）---
        processor = await self._get_processor()
        query_vector = await processor.generate_embedding(query)

        event_repo = EventVectorRepository(get_es_client())

        es_results = await event_repo.search_similar_by_title(
            query_vector=query_vector,
            k=atomic_top_k * 3,
            source_config_ids=source_config_ids,
        )

        db_count = 0
        es_new_count = 0
        es_count = 0

        for hit in es_results:
            if es_count >= atomic_top_k:
                break

            score = hit.get("_score", 0.0)
            if score < threshold:
                continue

            eid = hit.get("event_id", "")
            if not eid:
                continue

            if eid not in merged:
                es_new_count += 1
            merged[eid] = score
            es_count += 1

        db_count = len(merged) - es_new_count
        items = [{"event_id": eid, "score": score} for eid, score in merged.items()]

        logger.info(
            f"[Step3-双通道召回] query='{query}' -> "
            f"entity→event={db_count}, query→event={es_new_count}, "
            f"merged={len(items)}"
        )
        return items

    async def step4_fetch_event_details(
        self,
        event_ids: List[str],
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
        """
        Step4: 查询事项详情和关联实体（只查不更新 Set）

        Args:
            event_ids: 事项 ID 列表

        Returns:
            (event_details, event_entities):
            - event_details: {event_id: {"title": str, "content": str}}
            - event_entities: {event_id: [entity_id, entity_id, ...]}
        """
        if not event_ids:
            return {}, {}

        session_factory = get_session_factory()
        event_details: Dict[str, Dict[str, str]] = {}
        event_entities: Dict[str, List[str]] = {}

        async with session_factory() as session:
            events_stmt = select(SourceEvent).where(
                SourceEvent.id.in_(event_ids)
            )
            result = await session.execute(events_stmt)
            for event in result.scalars().all():
                event_details[event.id] = {
                    "title": event.title or "",
                    "content": event.content or "",
                }

            ee_stmt = select(EventEntity.event_id, EventEntity.entity_id).where(
                EventEntity.event_id.in_(event_ids)
            )
            result = await session.execute(ee_stmt)
            for row in result.fetchall():
                eid, kid = row[0], row[1]
                if eid not in event_details:
                    continue
                if eid not in event_entities:
                    event_entities[eid] = []
                event_entities[eid].append(kid)

        logger.info(
            f"[Step4-事项详情] input_event_ids={len(event_ids)}, "
            f"found_events={len(event_details)}, "
            f"event_entity_relations={sum(len(v) for v in event_entities.values())}"
        )
        return event_details, event_entities

    def get_new_entity_ids(self, event_entities: Dict[str, List[str]]) -> List[str]:
        """
        从 event_entities 中找出未在 self._entity_ids 中出现过的实体 ID

        用于扩展时发现新实体，决定是否需要进一步检索。

        Args:
            event_entities: {event_id: [entity_id, ...]}

        Returns:
            新的实体 ID 列表（去重）
        """
        all_ids = set()
        for entity_ids in event_entities.values():
            all_ids.update(entity_ids)
        new_ids = all_ids - self._entity_ids
        logger.info(
            f"[去重] total={len(all_ids)}, "
            f"already_tracked={len(all_ids) - len(new_ids)}, "
            f"new={len(new_ids)}"
        )
        return list(new_ids)

    async def step5_expand(
        self,
        event_entities: Dict[str, List[str]],
        source_config_ids: Optional[List[str]] = None,
        max_hops: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
        """
        Step5: 多跳扩展

        逻辑：
          hop=0: entity_set = step2 实体, relation_set = step3 合并事件
          hop=N: prev_hop_entities → 新 keys (不在 entity_set)
                 新 keys → 新 relations (不在 relation_set)
                 更新两个 set, prev_hop_entities = 本跳新事件的 entities

        Args:
            event_entities: Step4 返回的 {event_id: [entity_id, ...]}
            source_config_ids: 信息源 ID 列表（可选）
            max_hops: 最大跳数（默认取 AtomicConfig）

        Returns:
            (all_details, all_entities) 所有扩展轮次累积的详情和实体字典
        """
        config = AtomicConfig()
        max_hops = max_hops if max_hops is not None else config.max_hops

        all_details: Dict[str, Dict[str, str]] = {}
        all_entities: Dict[str, List[str]] = {}

        # hop=0: 初始化 relation_set（entity_set 已由 step2 填充）
        self._relation_ids.update(event_entities.keys())

        if max_hops == 0:
            return all_details, all_entities

        # 上一跳的 event_entities，用于每轮发现新 keys
        prev_hop_entities = event_entities

        for hop in range(max_hops):
            pre_events = len(self._relation_ids)
            pre_entities = len(self._entity_ids)

            # 1. 从上一跳 events 找新 keys（不在 entity_set 中）
            new_entity_ids = self.get_new_entity_ids(prev_hop_entities)

            if not new_entity_ids:
                logger.info(
                    f"[Step5-扩展] hop={hop+1}/{max_hops} "
                    f"无新实体 (tracked_entities={len(self._entity_ids)})，停止"
                )
                break

            # 2. 新 keys 加入 entity_set
            self._entity_ids.update(new_entity_ids)

            logger.info(
                f"[Step5-扩展] hop={hop+1}/{max_hops} "
                f"entities: {pre_entities} -> +{len(new_entity_ids)} new, total={len(self._entity_ids)}"
            )

            # 3. 新 keys → DB 查新 relations（不在 relation_set 中）
            new_event_ids: List[str] = []
            session_factory = get_session_factory()
            async with session_factory() as session:
                stmt = select(EventEntity.event_id).where(
                    EventEntity.entity_id.in_(new_entity_ids)
                ).distinct()
                if source_config_ids:
                    stmt = stmt.join(
                        SourceEvent, SourceEvent.id == EventEntity.event_id
                    ).where(
                        SourceEvent.source_config_id.in_(source_config_ids)
                    )
                result = await session.execute(stmt)
                for row in result.fetchall():
                    if row[0] not in self._relation_ids:
                        new_event_ids.append(row[0])

            if not new_event_ids:
                logger.info(
                    f"[Step5-扩展] hop={hop+1}/{max_hops} "
                    f"无新事项 (tracked_events={len(self._relation_ids)})，停止"
                )
                break

            # 4. 查新事项详情
            hop_details, hop_entities = await self.step4_fetch_event_details(new_event_ids)

            # 5. 新 relations 加入 relation_set
            self._relation_ids.update(new_event_ids)

            all_details.update(hop_details)
            all_entities.update(hop_entities)

            # 6. 保存本跳结果，供下一跳使用
            prev_hop_entities = hop_entities

            logger.info(
                f"[Step5-扩展] hop={hop+1}/{max_hops} done: "
                f"events {pre_events} -> {len(self._relation_ids)} (+{len(new_event_ids)}), "
                f"entities {pre_entities} -> {len(self._entity_ids)}"
            )

        return all_details, all_entities


    async def step6_coarse_rank(
        self,
        query: str,
        event_ids: List[str],
        source_config_ids: Optional[List[str]] = None,
        max_events: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step6: 粗排序

        用 query 向量在 ES 中做一次 kNN 搜索，通过 event_ids 过滤，
        返回最多 max_events 条按相似度降序的结果。

        Args:
            query: 查询文本
            event_ids: 需要排序的事项 ID 列表
            source_config_ids: 信息源 ID 列表（可选）
            max_events: 最大返回数量（默认取 AtomicConfig）

        Returns:
            [{"event_id": str, "score": float}, ...] 按相似度降序
        """
        if not event_ids:
            return []

        config = AtomicConfig()
        max_events = max_events or config.max_events

        processor = await self._get_processor()
        query_vector = await processor.generate_embedding(query)

        event_repo = EventVectorRepository(get_es_client())
        results = await event_repo.search_similar_by_title(
            query_vector=query_vector,
            k=max_events,
            source_config_ids=source_config_ids,
            event_ids=event_ids,
        )

        scored = []
        for hit in results:
            eid = hit.get("event_id", "")
            score = hit.get("_score", 0.0)
            if eid:
                scored.append({"event_id": eid, "score": score})

        top_score_str = f"{scored[0]['score']:.4f}" if scored else "0"
        logger.info(
            f"[Step6-粗排序] input={len(event_ids)}, "
            f"returned={len(scored)}, "
            f"top_score={top_score_str}"
        )
        return scored

    @staticmethod
    def _correct_rerank_line(
        predict_line: str,
        relation_texts: List[str],
        relation_ids: List[str],
    ) -> Optional[str]:
        """LLM 返回的 id 无效时，用文本内容匹配纠错"""
        text = predict_line[predict_line.find("]") + 1:].strip()
        for line_text, id_ in zip(relation_texts, relation_ids):
            if line_text.strip() == text:
                return id_
        return None

    def _parse_rerank_response(
        self,
        useful_relations: List[str],
        valid_ids: set,
        relation_ids: List[str],
        relation_texts: List[str],
    ) -> List[str]:
        """解析 LLM 返回的 useful_relations，提取 [id] 并纠错"""
        selected: List[str] = []
        for line in useful_relations:
            if "[" not in line or "]" not in line:
                continue
            rel_id = line[line.find("[") + 1: line.find("]")].strip()
            if rel_id in valid_ids and rel_id not in selected:
                selected.append(rel_id)
            elif rel_id not in valid_ids:
                corrected = self._correct_rerank_line(line, relation_texts, relation_ids)
                if corrected and corrected not in selected:
                    selected.append(corrected)
        return selected

    async def step7_llm_rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Step7: LLM 精选最相关的原子事项

        将候选事项格式化为 [id] title content，通过 few-shot prompt 让 LLM
        挑选 top_k 条最相关的事项，解析响应并映射回原始数据。

        Args:
            query: 查询文本
            items: 候选事项 [{event_id, title, content, score}]
            top_k: 精选返回数量

        Returns:
            筛选后的事项列表，保持 LLM 选择的顺序
        """
        if not items:
            return []

        top_k = min(top_k, len(items))

        # 1. 构建 idx → event_id 映射 + 格式化 relation 文本
        idx_to_event_id: Dict[str, str] = {}
        relation_lines: List[str] = []
        relation_texts: List[str] = []

        for i, item in enumerate(items):
            idx = str(i)
            idx_to_event_id[idx] = item["event_id"]
            text = item.get("title", "").strip()
            relation_lines.append(f"[{i}] {text}")
            relation_texts.append(text)

        relations_str = "\n".join(relation_lines)
        valid_ids = set(idx_to_event_id.keys())

        # 2. 构建 messages：SYSTEM + 3 组 few-shot + 最终 prompt
        system_prompt = _RERANK_SYSTEM_PROMPT.format(top_k=top_k)
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
            # few-shot 1
            LLMMessage(role=LLMRole.USER, content=_RERANK_EXAMPLE_1_INPUT),
            LLMMessage(role=LLMRole.ASSISTANT, content=_RERANK_EXAMPLE_1_OUTPUT),
            # few-shot 2
            LLMMessage(role=LLMRole.USER, content=_RERANK_EXAMPLE_2_INPUT),
            LLMMessage(role=LLMRole.ASSISTANT, content=_RERANK_EXAMPLE_2_OUTPUT),
            # few-shot 3
            LLMMessage(role=LLMRole.USER, content=_RERANK_EXAMPLE_3_INPUT),
            LLMMessage(role=LLMRole.ASSISTANT, content=_RERANK_EXAMPLE_3_OUTPUT),
            # 实际查询
            LLMMessage(
                role=LLMRole.USER,
                content=_RERANK_TEMPLATE.format(question=query, relations=relations_str),
            ),
        ]

        # 3. 调用 LLM
        llm_client = await self._get_llm_client()
        response = await llm_client.chat_with_schema(
            messages,
            response_schema={
                "type": "object",
                "properties": {
                    "thought_process": {"type": "string"},
                    "useful_relations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["thought_process", "useful_relations"],
            },
        )

        # 记录 token 消耗
        if hasattr(response, "usage"):
            usage = response.usage
            self.token_counter.add_record(
                scenario="atomic_rerank",
                model=getattr(response, "model", "unknown"),
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                metadata={"query": query, "candidates": len(items)},
            )

        # 4. 解析 + 纠错
        useful_relations = response.get("useful_relations", [])
        selected_indices = self._parse_rerank_response(
            useful_relations, valid_ids, list(idx_to_event_id.keys()), relation_texts,
        )

        # 5. 映射回原始数据
        results = []
        event_id_to_item = {item["event_id"]: item for item in items}
        for idx in selected_indices[:top_k]:
            event_id = idx_to_event_id.get(idx)
            if event_id and event_id in event_id_to_item:
                results.append(event_id_to_item[event_id])

        logger.info(
            f"[Step7-LLM精选] query='{query}', candidates={len(items)}, "
            f"selected={len(results)}, top_k={top_k}"
        )
        return results

    async def step8_fetch_chunks(
        self,
        event_ids: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """
        Step8: 根据 event_id 查找关联的 chunk

        source_event.chunk_id → source_chunk 查详情

        Args:
            event_ids: 事项 ID 列表

        Returns:
            {event_id: {"chunk_id": str, "heading": str, "content": str}}
        """
        if not event_ids:
            return {}

        session_factory = get_session_factory()
        event_chunk_map: Dict[str, str] = {}
        result_map: Dict[str, Dict[str, str]] = {}

        async with session_factory() as session:
            # 1. 查 event → chunk_id
            stmt = select(SourceEvent.id, SourceEvent.chunk_id).where(
                SourceEvent.id.in_(event_ids)
            )
            result = await session.execute(stmt)
            chunk_ids: set = set()
            for row in result.fetchall():
                eid, chunk_id = row[0], row[1]
                if chunk_id:
                    event_chunk_map[eid] = chunk_id
                    chunk_ids.add(chunk_id)

            if not chunk_ids:
                return {}

            # 2. 查 chunk 详情
            chunk_stmt = select(SourceChunk).where(
                SourceChunk.id.in_(chunk_ids)
            )
            result = await session.execute(chunk_stmt)
            chunk_map: Dict[str, Dict[str, str]] = {}
            for chunk in result.scalars().all():
                chunk_map[chunk.id] = {
                    "chunk_id": chunk.id,
                    "source_id": chunk.source_id or "",
                    "source_config_id": chunk.source_config_id or "",
                    "heading": chunk.heading or "",
                    "content": chunk.content or "",
                    "rank": chunk.rank,
                }

            # 3. 按 event_id 映射
            for eid, chunk_id in event_chunk_map.items():
                if chunk_id in chunk_map:
                    result_map[eid] = chunk_map[chunk_id]

        logger.info(
            f"[Step8-Chunk查找] events={len(event_ids)} -> "
            f"chunk_ids={len(chunk_ids)}, matched={len(result_map)}"
        )
        return result_map

    async def search(
        self,
        query: str,
        source_config_ids: List[str],
        config: Optional[AtomicConfig] = None,
    ) -> Dict[str, Any]:
        """
        搜索原子事项

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            config: AtomicConfig 配置

        Returns:
            {
                "items": [
                    {
                        "event_id": str,
                        "title": str,
                        "content": str,
                        "score": float,
                        "chunk": {"chunk_id": str, "heading": str, "content": str} or None,
                    }
                ],
                "_timings": {"total": float}
            }
        """
        config = config or AtomicConfig()
        self._entity_ids = set()
        self._relation_ids = set()
        start_time = time.perf_counter()

        logger.info(f"[原子事项检索] query='{query}', atomic_top_k={config.atomic_top_k}")

        # Step1: 提取实体
        query_entities = await self.step1_extract_entities(query)

        # Step2: ES 向量检索实体
        entity_ids, entity_names, entity_scores = await self.step2_retrieve_entities(
            query_entities=query_entities,
            source_config_ids=source_config_ids,
            entity_top_k=config.entity_top_k,
            key_similarity_threshold=config.key_similarity_threshold,
        )

        # Step3: 双通道召回事项
        event_items = await self.step3_retrieve_events(
            query=query,
            source_config_ids=source_config_ids,
            entity_ids=entity_ids,
            atomic_top_k=config.atomic_top_k,
            similarity_threshold=config.similarity_threshold,
        )

        event_ids = [item["event_id"] for item in event_items]

        if not event_ids:
            total_time = time.perf_counter() - start_time
            return {
                "items": [],
                "_timings": {"total": total_time},
            }

        # Step4: 查询事项详情
        event_details, event_entities = await self.step4_fetch_event_details(event_ids)

        # Step5: 多跳扩展
        expand_details, expand_entities = await self.step5_expand(
            event_entities=event_entities,
            source_config_ids=source_config_ids,
            max_hops=config.max_hops,
        )

        # 合并所有事项详情（初始 + 扩展），只保存一次
        all_details = {**event_details, **expand_details}

        # Step6: 粗排序
        ranked = await self.step6_coarse_rank(
            query=query,
            event_ids=list(all_details.keys()),
            source_config_ids=source_config_ids,
            max_events=config.max_events,
        )

        # 组装候选列表
        candidates = []
        for item in ranked:
            eid = item["event_id"]
            detail = all_details.get(eid, {})
            candidates.append({
                "event_id": eid,
                "title": detail.get("title", ""),
                "content": detail.get("content", ""),
                "score": item["score"],
            })

        # Step7: LLM 精选
        items = await self.step7_llm_rerank(
            query=query,
            items=candidates,
            top_k=config.rerank_top_k,
        )

        # Step8: 查找关联 chunk
        filtered_event_ids = [item["event_id"] for item in items]
        chunk_map = await self.step8_fetch_chunks(filtered_event_ids)

        for item in items:
            item["chunk"] = chunk_map.get(item["event_id"])

        total_time = time.perf_counter() - start_time
        return {
            "items": items,
            "_timings": {"total": total_time},
        }

    async def search_for_rerank(
        self,
        query: str,
        source_config_ids: List[str],
        query_vector: Optional[List[float]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        原子事项检索（rerank 兼容接口）

        返回格式与 VectorSearcher.search_chunks_for_rerank 一致，
        方便接入统一的 rerank 流程。

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            query_vector: 可选的预计算向量（暂未使用）
            config: AtomicConfig 或 SearchConfig 对象

        Returns:
            {"sections": [...], "_timings": {...}}
        """
        atomic_config = config if isinstance(config, AtomicConfig) else AtomicConfig()

        result = await self.search(query, source_config_ids, atomic_config)

        seen_chunk_ids: set = set()
        sections = []
        for i, item in enumerate(result.get("items", [])):
            chunk = item.get("chunk")
            if not chunk:
                continue
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            sections.append({
                "chunk_id": chunk_id,
                "source_id": chunk["source_id"],
                "source_config_id": chunk["source_config_id"],
                "heading": chunk["heading"],
                "content": chunk["content"],
                "rank": chunk.get("rank", i),
                "score": item["score"],
                "weight": item["score"],
            })

        # Native 补充：去重后不足 max_sections 时，用 query→chunk 填充
        target = atomic_config.max_sections
        if len(sections) < target:
            atomic_count = len(sections)
            supplement = await self.search_chunks(
                query=query,
                source_config_ids=source_config_ids,
                config=atomic_config,
            )
            native_added = 0
            for sec in supplement.get("sections", []):
                if sec["chunk_id"] in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(sec["chunk_id"])
                sections.append(sec)
                native_added += 1
                if len(sections) >= target:
                    break
            logger.info(
                f"[Native补充] atomic={atomic_count}, native=+{native_added}, "
                f"total={len(sections)}"
            )

        return {
            "sections": sections[:target],
            "_timings": result.get("_timings", {}),
        }

    async def search_chunks(
        self,
        query: str,
        source_config_ids: List[str],
        config: Optional[AtomicConfig] = None,
    ) -> Dict[str, Any]:
        """
        Query→Chunk 直接向量检索

        跳过实体提取和多跳扩展，直接用 query 向量检索 chunk。
        用于简单场景或作为 Atomic 管线的补充通道。

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            config: AtomicConfig 配置

        Returns:
            {"sections": [...], "_timings": {"total": float}}
        """
        config = config or AtomicConfig()
        start_time = time.perf_counter()

        processor = await self._get_processor()
        query_vector = await processor.generate_embedding(query)

        chunk_repo = SourceChunkRepository(get_es_client())
        es_results = await chunk_repo.search_similar_by_content(
            query_vector=query_vector,
            k=config.max_sections*2,
            source_config_ids=source_config_ids,
        )

        sections = []
        for result in es_results:
            score = result.get("_score", 0.0)
            sections.append({
                "chunk_id": result.get("chunk_id"),
                "source_id": result.get("source_id"),
                "source_config_id": result.get("source_config_id"),
                "heading": result.get("heading"),
                "content": result.get("content"),
                "rank": result.get("rank"),
                "score": score,
                "weight": score,
            })

        sections = sorted(sections, key=lambda x: x["score"], reverse=True)[:config.max_sections]
        total_time = time.perf_counter() - start_time

        logger.info(
            f"[Query→Chunk] query='{query}', "
            f"returned={len(sections)}, total_time={total_time:.3f}s"
        )

        return {
            "sections": sections,
            "_timings": {"total": total_time},
        }


__all__ = ["AtomicSearcher"]
