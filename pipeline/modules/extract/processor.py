"""
事项处理器 - LLM提取

职责：
- 构建提示词和输入（从 YAML 读取）
- 调用 LLM（带重试）
- 历史事项召回（作为背景信息）
- 输出校验
"""

import copy
import json
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select

from pipeline.core.ai.base import BaseLLMClient
from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.core.prompt.manager import PromptManager
from pipeline.core.storage.elasticsearch import get_es_client
from pipeline.core.storage.repositories.event_repository import EventVectorRepository
from pipeline.db import get_session_factory
from pipeline.db.models import Article, EntityType as DBEntityType, SourceEvent
from pipeline.exceptions import ExtractError
from pipeline.modules.extract.config import ExtractConfig
from pipeline.utils import get_logger

logger = get_logger("extract.processor")


class EventProcessor:
    """事项处理器 - 负责LLM调用"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_manager: PromptManager,
        config: ExtractConfig,
    ):
        """
        初始化事项处理器

        Args:
            llm_client: LLM客户端
            prompt_manager: 提示词管理器
            config: 提取配置
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.config = config
        self.session_factory = get_session_factory()

        # 状态
        self.entity_types: List[DBEntityType] = []

        # 历史事项召回相关（延迟初始化）
        self._es_client = None
        self._event_repo = None
        self._embedding_client = None

    async def initialize(self, entity_types: List[DBEntityType]):
        """
        初始化（设置实体类型）

        Args:
            entity_types: 实体类型列表
        """
        self.entity_types = entity_types

    async def process(
        self,
        items: List,
        metadata: Dict,
        source_type: str,
    ) -> Dict:
        """
        处理提取（调用LLM）

        流程：
        1. 召回历史事项（作为背景）
        2. 构建系统提示词
        3. 构建输入 JSON
        4. 调用 LLM
        5. 校验输出

        Args:
            items: ArticleSection 或 ChatMessage 列表
            metadata: 元数据 {document_title, chunk_title, previous_context}
            source_type: "ARTICLE" 或 "CHAT"

        Returns:
            LLM返回的原始结果Dict
        """
        if not items:
            logger.info("items 为空，跳过提取")
            return {"type": "response", "data": {"items": [], "meta": {}}}

        try:
            # 1. 召回历史事项
            related_events = await self._recall_related_events(items, source_type)

            # 2. 构建系统提示词
            system_prompt = self._build_system_prompt()
            logger.info(f"系统提示词长度: {len(system_prompt)} 字符")

            # 3. 构建输入 JSON
            user_input = self._build_input(items, metadata, source_type, related_events)
            logger.info(
                f"输入: {len(items)} items, type={source_type}, 历史事项={len(related_events)}"
            )

            # 4. 构建消息（Few-shot 格式：system + 示例 user + 示例 assistant + 实际 user）
            messages = self._build_messages(system_prompt, user_input)

            # 5. 调用 LLM
            schema = self._build_schema()
            result = await self._call_llm_with_retry(messages, schema)

            # 6. 校验输出
            self._validate_output(result)

            # 记录元数据（meta 在 data 内部）
            meta = result.get("data", {}).get("meta", {})
            logger.info(
                f"LLM返回: reason={meta.get('reason', '')}, "
                f"confidence={meta.get('confidence', 0)}"
            )

            return result

        except Exception as e:
            logger.error(f"提取失败: {e}", exc_info=True)
            raise ExtractError(f"提取失败: {e}") from e

    async def get_source_created_time(self, items: List, source_type: str) -> Optional[datetime]:
        """
        获取源创建时间

        Args:
            items: 输入 items
            source_type: "ARTICLE"

        Returns:
            创建时间，如果不存在则返回 None
        """
        if source_type != "ARTICLE" or not items:
            return None

        article_id = getattr(items[0], "article_id", None)
        if not article_id:
            return None

        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(Article.created_time).where(Article.id == article_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.info(f"获取文章创建时间失败: {e}")
            return None

    def _build_system_prompt(self) -> str:
        """构建系统提示词（从 YAML 读取，不含示例）"""
        tz = ZoneInfo(self.config.timezone)
        time_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M")

        config = self.prompt_manager.get_template_config("extract", test_mode=self.config.test_mode)
        template = config.get("template", "")
        strict_requirements_template = config.get("strict_requirements", "")

        custom_background = self._format_custom(self.config.custom_background)

        # 如果启用严格过滤，从 YAML 读取规则并追加
        custom_requirements = self.config.custom_requirements
        if self.config.enable_strict_filtering and strict_requirements_template:
            formatted_strict = self._format_custom(strict_requirements_template)
            if custom_requirements:
                custom_requirements = custom_requirements + "\n" + formatted_strict
            else:
                custom_requirements = formatted_strict

        custom_requirements = self._format_custom(custom_requirements)

        try:
            return template.format(
                time=time_str,
                timezone=self.config.timezone,
                custom_background=custom_background,
                custom_requirements=custom_requirements,
            )
        except KeyError:
            return self.prompt_manager.render(
                "extract",
                time=time_str,
                timezone=self.config.timezone,
                custom_background=custom_background,
                custom_requirements=custom_requirements,
            )

    def _format_custom(self, text: str) -> str:
        """格式化自定义文本"""
        if not text:
            return ""
        lines = text.strip().split("\n")
        return "\n" + "\n".join(f"    {line}" for line in lines)

    def _build_messages(self, system_prompt: str, user_input: Dict) -> List[LLMMessage]:
        """
        构建消息列表（Few-shot 格式）

        消息结构：
        1. system: 系统提示词（不含示例）
        2. user: 示例输入 JSON
        3. assistant: 示例输出 JSON
        4. user: 当前实际输入 JSON
        """
        # 获取示例
        config = self.prompt_manager.get_template_config("extract", test_mode=self.config.test_mode)
        examples = config.get("examples", {})
        example_input = examples.get("input", "")
        example_output = examples.get("output", "")

        messages = [
            # 1. 系统提示词
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
        ]

        # 2-3. Few-shot 示例（如果有）
        if example_input and example_output:
            messages.append(LLMMessage(role=LLMRole.USER, content=example_input.strip()))
            messages.append(LLMMessage(role=LLMRole.ASSISTANT, content=example_output.strip()))
            logger.debug("添加 Few-shot 示例消息")

        # 4. 当前实际输入
        messages.append(
            LLMMessage(role=LLMRole.USER, content=json.dumps(user_input, ensure_ascii=False))
        )

        logger.info(f"构建消息列表: {len(messages)} 条消息")
        return messages

    def _build_input(
        self,
        items: List,
        metadata: Dict,
        source_type: str,
        related_events: List[Dict],
    ) -> Dict:
        """
        构建输入 JSON（新结构：data 只含 items，meta 含所有元数据）

        结构：
        - type: "request"
        - data: { items: [...] }
        - meta: { source_type, source_title, source_summary, previous_context, entity_types, related_events }
        """
        is_article = source_type == "ARTICLE"

        # 构建 items 数组
        items_data = []
        for i, item in enumerate(items, 1):
            item_data = {"id": i, "content": item.content}
            items_data.append(item_data)

        # 构建 entity_types 对象数组（包含 type 和 description）
        entity_types_data = []
        for et in self.entity_types:
            entity_types_data.append(
                {"type": et.type, "description": et.description or f"{et.name}实体"}
            )

        # 构建输入结构（meta 在 data 内部）
        input_meta = {
            "source_type": "article" if is_article else "chat",
            "source_title": metadata.get("document_title", ""),
            "source_summary": metadata.get("document_summary", ""),
            "entity_types": entity_types_data,
        }

        # 可选字段：previous_context
        previous_context = metadata.get("previous_context", "")
        if previous_context:
            input_meta["previous_context"] = previous_context

        # 可选字段：related_events
        if related_events:
            input_meta["related_events"] = related_events

        return {
            "type": "request",
            "data": {
                "items": items_data,
                "meta": input_meta,
            },
        }

    def _build_schema(self) -> Dict:
        """构建输出 Schema"""
        config = self.prompt_manager.get_template_config("extract", test_mode=self.config.test_mode)
        output_schema = config.get("output_schema", {})
        definitions = config.get("definitions", {})

        schema = copy.deepcopy({**output_schema, "definitions": definitions})

        # 动态注入实体类型枚举
        valid_types = [et.type for et in self.entity_types]
        if valid_types:
            try:
                if "definitions" in schema and "entity" in schema["definitions"]:
                    entity_def = schema["definitions"]["entity"]
                    if "properties" in entity_def and "type" in entity_def["properties"]:
                        entity_def["properties"]["type"]["enum"] = valid_types
                        logger.info(f"注入实体类型枚举: {len(valid_types)} 个类型")
            except Exception as e:
                logger.info(f"注入实体类型枚举失败: {e}，继续使用默认 schema")

        return schema

    async def _call_llm_with_retry(self, messages: List[LLMMessage], schema: Dict) -> Dict:
        """调用 LLM（带重试）"""
        try:
            logger.info("调用 LLM（带重试机制）")

            result = await self.llm_client.chat_with_schema(
                messages, response_schema=schema
            )

            logger.info("LLM 调用成功")
            return result

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise ExtractError(f"LLM 调用失败: {e}") from e

    def _validate_output(self, result: Dict):
        """
        校验输出格式（增强验证）

        严格验证：结构错误会抛出异常
        宽松验证：内容质量问题只记录警告，不中断任务
        """
        # === 严格验证：结构错误 ===
        if result.get("type") != "response":
            raise ValueError(f"输出 type 必须为 'response'，实际: {result.get('type')}")

        if "data" not in result:
            raise ValueError("输出缺少 'data' 字段")

        if "items" not in result.get("data", {}):
            raise ValueError("输出 data 缺少 'items' 字段")

        if "meta" not in result.get("data", {}):
            raise ValueError("输出 data 缺少 'meta' 字段")

        # === 宽松验证：内容质量（记录警告，不中断） ===
        items = result.get("data", {}).get("items", [])
        valid_types = {et.type for et in self.entity_types}

        empty_refs_count = 0
        empty_title_count = 0
        empty_content_count = 0
        invalid_entity_types = set()

        def validate_item(item: Dict, path: str = ""):
            """递归验证事项（包括 children）"""
            nonlocal empty_refs_count, empty_title_count, empty_content_count

            item_path = f"{path}.{item.get('title', '?')}" if path else item.get("title", "?")

            # 验证 references（只统计，不单独记录）
            refs = item.get("references", [])
            if not refs or len(refs) == 0:
                empty_refs_count += 1

            # 验证 title（只统计，不单独记录）
            if not item.get("title", "").strip():
                empty_title_count += 1

            # 验证 content（只统计，不单独记录）
            if not item.get("content", "").strip():
                empty_content_count += 1

            # 验证实体类型
            entities = item.get("entities", [])
            for entity in entities:
                entity_type = entity.get("type")
                if entity_type and entity_type not in valid_types:
                    invalid_entity_types.add(entity_type)

            # 递归验证 children
            children = item.get("children", [])
            for child in children:
                validate_item(child, item_path)

        # 验证所有事项
        for item in items:
            validate_item(item)

        # 汇总警告
        if empty_refs_count > 0:
            logger.warning(f"输出验证: {empty_refs_count} 个事项的 references 为空")
        if empty_title_count > 0:
            logger.warning(f"输出验证: {empty_title_count} 个事项的 title 为空")
        if empty_content_count > 0:
            logger.warning(f"输出验证: {empty_content_count} 个事项的 content 为空")
        if invalid_entity_types:
            logger.warning(
                f"输出验证: 发现无效实体类型 {invalid_entity_types}，"
                f"允许的类型: {sorted(valid_types)}"
            )

    async def _recall_related_events(self, items: List, _source_type: str = None) -> List[Dict]:
        """
        召回历史事项（用于分类和实体命名参考）

        Args:
            items: ArticleSection 或 ChatMessage 列表
            _source_type: 保留用于未来扩展

        Returns:
            历史事项列表 [{title, category, entities: [{type, name}]}]
        """
        if not self.config.enable_related_events:
            return []

        try:
            await self._ensure_recall_deps()

            # 构建查询文本
            content_text = " ".join([item.content for item in items])

            # 生成向量（使用 embedding 专用长度限制，避免超过模型 token 上限）
            max_len = self.config.embedding_max_length
            content_vector = await self._embedding_client.generate_embedding(content_text[:max_len])

            # 从向量库召回
            results = await self._event_repo.search_similar_by_content(
                query_vector=content_vector,
                k=self.config.related_events_top_k,
                source_config_id=self.config.source_config_id,
            )

            # 过滤低相似度结果
            results = [
                r for r in results if r.get("_score", 0) >= self.config.related_events_threshold
            ]

            if not results:
                return []

            # 从数据库加载事项详情
            event_ids = [r["event_id"] for r in results]
            related_events = []

            async with self.session_factory() as session:
                from pipeline.db.models import EventEntity
                from sqlalchemy.orm import selectinload

                stmt = (
                    select(SourceEvent)
                    .options(
                        selectinload(SourceEvent.event_associations).selectinload(
                            EventEntity.entity
                        )
                    )
                    .where(SourceEvent.id.in_(event_ids))
                )
                db_events = (await session.execute(stmt)).scalars().all()

                for event in db_events:
                    entities = [
                        {"type": ee.entity.type, "name": ee.entity.name}
                        for ee in event.event_associations
                        if ee.entity
                    ]
                    related_events.append(
                        {
                            "title": event.title,
                            "category": event.category or "",
                            "entities": entities[:10],
                        }
                    )

            logger.info(f"召回 {len(related_events)} 个相关历史事项作为参考")
            return related_events

        except Exception as e:
            logger.info(f"历史事项召回失败: {e}")
            return []

    async def _ensure_recall_deps(self):
        """确保召回依赖可用"""
        if self._event_repo is None:
            from pipeline.modules.load.processor import DocumentProcessor

            self._es_client = get_es_client()
            self._event_repo = EventVectorRepository(self._es_client)
            self._embedding_client = DocumentProcessor(llm_client=self.llm_client)
