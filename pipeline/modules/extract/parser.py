"""
解析器模块 - 完整解析层

职责：
- 事项解析：LLM结果 -> SourceEvent
- 实体解析：创建/查找 Entity
- 关系解析：创建 EventEntity 关联
- 值类型推断：解析实体值类型
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError

from pipeline.db import Entity, EventEntity, SourceEvent, get_session_factory
from pipeline.db.models import EntityType as DBEntityType
from pipeline.modules.extract.config import ExtractConfig
from pipeline.utils import get_logger, get_utc_now

logger = get_logger("extract.parser")


@dataclass
class ParseContext:
    """解析上下文"""

    source_config_id: str
    source_type: str  # "ARTICLE" or "CHAT"
    source_id: str
    chunk_id: str
    source_created_time: Optional[datetime] = None


class ResultParser:
    """
    结果解析器 - 协调整个解析流程

    流程：
    1. 解析事项：LLM返回的Dict -> SourceEvent列表
    2. 解析实体：创建/查找 Entity，建立 EventEntity 关联
    """

    def __init__(self, config: ExtractConfig):
        """
        初始化解析器

        Args:
            config: 提取配置
        """
        self.config = config
        self.session_factory = get_session_factory()
        self.value_parser = EntityValueParser()

    def parse_events(
        self,
        raw_items: List[Dict],
        items: List,
        context: ParseContext,
    ) -> List[SourceEvent]:
        """
        解析事项（LLM结果 -> SourceEvent列表）

        Args:
            raw_items: LLM返回的事项数据列表
            items: 原始输入 items（用于引用解析）
            context: 解析上下文

        Returns:
            SourceEvent 列表（扁平化，包含所有层级）
        """
        # 构建 index -> item 映射（1-based）
        index_map = {i + 1: item for i, item in enumerate(items)}
        all_item_ids = [item.id for item in items]

        # 递归解析（扁平化存储）
        all_events = []
        for item_data in raw_items:
            events = self._parse_item_recursive(
                item_data,
                index_map,
                all_item_ids,
                context,
                parent_id=None,
                level=0,
            )
            all_events.extend(events)

        return all_events

    def _parse_item_recursive(
        self,
        item_data: Dict,
        index_map: Dict,
        all_item_ids: List[str],
        context: ParseContext,
        parent_id: Optional[str],
        level: int = 0,
    ) -> List[SourceEvent]:
        """
        递归解析单个事项（返回扁平列表）

        Args:
            item_data: 事项数据
            index_map: {序号: item} 映射
            all_item_ids: 所有 item 的 UUID 列表
            context: 解析上下文
            parent_id: 父事项ID
            level: 层级深度（0=L0顶层，1=L1，2=L2）

        Returns:
            [parent, child1, child2, ...] 扁平列表
        """
        # 过滤无效事项（LLM标记）
        if not item_data.get("is_valid", True):
            reason = item_data.get("reason", "")
            logger.info(f"过滤无效事项: {item_data.get('title', '')} - {reason}")
            return []

        # 解析 references
        ref_indices = item_data.get("references", [])
        valid_refs = self._parse_references(ref_indices, index_map, all_item_ids)

        # 提取时间
        start_time, end_time = self._extract_times(
            context.source_type, valid_refs, index_map, context.source_created_time
        )

        # 转换实体格式
        raw_entities = self._parse_raw_entities(item_data.get("entities", []))

        # 提取字段（空字符串转为 None）
        category = item_data.get("category") or None
        priority = item_data.get("priority") or None
        status = item_data.get("status") or None
        keywords = item_data.get("keywords") or None

        # 创建事项
        event_id = str(uuid.uuid4())
        event = SourceEvent(
            id=event_id,
            source_config_id=context.source_config_id,
            source_type=context.source_type,
            type=context.source_type,
            source_id=context.source_id or "",
            article_id=context.source_id if context.source_type == "ARTICLE" else None,
            chunk_id=context.chunk_id,
            parent_id=parent_id,
            rank=0,  # rank 统一在 extractor 中分配
            level=level,
            title=item_data.get("title", ""),
            summary=item_data.get("summary", ""),
            content=item_data.get("content", ""),
            category=category,
            keywords=keywords,
            priority=priority,
            status=status,
            start_time=start_time or get_utc_now().replace(tzinfo=None),
            end_time=end_time,
            references=valid_refs,
            extra_data={
                "raw_entities": {"entities": raw_entities},
                "raw_data": item_data,  # 保存原始数据用于质量过滤
            },
        )

        result = [event]

        # 递归处理子事项
        children = item_data.get("children", [])
        if children:
            logger.info(f"事项 '{event.title}' 包含 {len(children)} 个子事项")
            for child_data in children:
                child_events = self._parse_item_recursive(
                    child_data,
                    index_map,
                    all_item_ids,
                    context,
                    parent_id=event_id,
                    level=level + 1,
                )
                result.extend(child_events)

        return result

    def _parse_references(
        self,
        ref_indices: List,
        index_map: Dict,
        all_item_ids: List[str],
    ) -> List[str]:
        """
        解析 references（index -> UUID）

        Args:
            ref_indices: LLM 返回的引用序号列表
            index_map: {序号: item} 映射
            all_item_ids: 所有 item 的 UUID 列表（兜底用）

        Returns:
            UUID 列表
        """
        valid_refs = []
        invalid_indices = []

        for idx in ref_indices:
            # 兼容字符串数字
            if isinstance(idx, str):
                if idx.isdigit():
                    idx = int(idx)
                else:
                    invalid_indices.append((idx, "非数字字符串"))
                    continue

            if isinstance(idx, int) and idx in index_map:
                valid_refs.append(index_map[idx].id)
            elif isinstance(idx, int):
                invalid_indices.append((idx, f"越界(有效范围1-{len(index_map)})"))

        # 记录无效索引
        if invalid_indices:
            logger.warning(f"包含无效引用: {invalid_indices}")

        # 兜底：无有效引用时使用全部 items
        if not valid_refs:
            if ref_indices:
                logger.warning(f"所有引用无效 (共{len(ref_indices)}个)，使用全部片段作为兜底")
            else:
                logger.warning("引用为空，使用全部片段作为兜底")
            valid_refs = all_item_ids

        return valid_refs

    def _extract_times(
        self,
        source_type: str,
        valid_refs: List[str],
        index_map: Dict,
        source_created_time: Optional[datetime],
    ) -> tuple:
        """
        提取时间

        - ARTICLE: 使用文档创建时间

        Args:
            source_type: "ARTICLE"
            valid_refs: 有效的引用 UUID 列表
            index_map: {序号: item} 映射
            source_created_time: 源创建时间

        Returns:
            (start_time, end_time)
        """
        if source_type == "ARTICLE":
            if source_created_time:
                return source_created_time, source_created_time
            return None, None

        return None, None

    def _parse_raw_entities(self, entities_list: List) -> List[Dict]:
        """
        解析实体格式（统一为列表格式）

        Args:
            entities_list: 实体列表

        Returns:
            标准化的实体列表
        """
        result = []

        if isinstance(entities_list, list):
            for entity in entities_list:
                if isinstance(entity, dict):
                    result.append(
                        {
                            "type": entity.get("type", ""),
                            "name": entity.get("name", ""),
                            "description": entity.get("description", ""),
                        }
                    )
        return result

    async def process_entity_associations(
        self,
        events: List[SourceEvent],
        entity_types: List[DBEntityType],
    ) -> List[SourceEvent]:
        """
        处理实体关联（完整流程）

        流程：
        1. 实体去重和创建（使用缓存避免重复查询）
        2. 合并描述（同一实体的多个描述合并）
        3. 创建 EventEntity 关联

        Args:
            events: 事项列表（extra_data["raw_entities"]["entities"] 包含实体数据）
            entity_types: 实体类型列表

        Returns:
            处理后的事项列表（已设置 event_associations）
        """
        if not events:
            return events

        entity_cache = {}

        for event in events:
            raw_entities = event.extra_data.get("raw_entities", {}).get("entities", [])
            if not raw_entities:
                event.event_associations = []
                continue

            entity_map = {}

            # 收集实体（去重和创建）
            for entity_data in raw_entities:
                cache_key = self._build_cache_key(entity_data)

                if cache_key in entity_cache:
                    entity = entity_cache[cache_key]
                else:
                    entity = await self._get_or_create_entity(entity_data, entity_types)
                    if entity:
                        entity_cache[cache_key] = entity

                if entity is None:
                    continue

                # 收集描述（去重）
                if entity.id not in entity_map:
                    entity_map[entity.id] = {"name": entity.name, "descriptions": []}

                description = entity_data.get("description", "").strip()
                if description and description not in entity_map[entity.id]["descriptions"]:
                    entity_map[entity.id]["descriptions"].append(description)

            # 创建 EventEntity 关联
            event.event_associations = []
            for entity_id, info in entity_map.items():
                final_description = self._merge_descriptions(info["descriptions"])
                assoc = EventEntity(
                    id=str(uuid.uuid4()),
                    event_id=event.id,
                    entity_id=entity_id,
                    description=final_description,
                )
                event.event_associations.append(assoc)

        return events

    def _build_cache_key(self, entity_data: Dict) -> tuple:
        """构建实体缓存键"""
        return (
            entity_data.get("type", ""),
            entity_data.get("name", "").strip().lower(),
        )

    def _merge_descriptions(self, descriptions: List[str]) -> str:
        """合并描述列表"""
        return "、".join(descriptions) if descriptions else ""

    async def _get_or_create_entity(
        self,
        entity_data: Dict,
        entity_types: List[DBEntityType],
    ) -> Optional[Entity]:
        """
        查找或创建实体（并发安全）

        支持重试机制处理锁等待超时和死锁

        Args:
            entity_data: 实体数据字典 {type, name, description, ...}
            entity_types: 实体类型列表

        Returns:
            Entity 对象，如果实体类型无效或名称无效则返回 None
        """
        entity_name = entity_data.get("name", "").strip()
        if len(entity_name) <= 1:
            return None

        normalized_name = entity_name.lower()
        max_retries = 3
        base_delay = 0.1

        for attempt in range(max_retries):
            try:
                return await self._get_or_create_entity_inner(
                    entity_data, normalized_name, entity_types
                )
            except OperationalError as e:
                error_str = str(e)
                is_lock_timeout = "1205" in error_str or "Lock wait timeout" in error_str
                is_deadlock = "1213" in error_str or "Deadlock" in error_str
                is_lost_connection = "2013" in error_str or "Lost connection" in error_str

                if (is_lock_timeout or is_deadlock or is_lost_connection) and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(f"实体创建异常({e.orig if hasattr(e, 'orig') else e})，重试 {attempt + 1}/{max_retries}: {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise

        return None

    async def _get_or_create_entity_inner(
        self,
        entity_data: Dict,
        normalized_name: str,
        entity_types: List[DBEntityType],
    ) -> Optional[Entity]:
        """实际执行查找或创建"""
        async with self.session_factory() as session:
            # 查找已存在
            existing = await session.execute(
                select(Entity)
                .where(Entity.source_config_id == self.config.source_config_id)
                .where(Entity.type == entity_data["type"])
                .where(Entity.normalized_name == normalized_name)
            )
            entity = existing.scalar_one_or_none()
            if entity:
                return entity

            # 查找实体类型
            entity_type = next((et for et in entity_types if et.type == entity_data["type"]), None)
            if not entity_type:
                logger.warning(
                    f"跳过无效实体类型: type={entity_data['type']}, "
                    f"name={entity_data.get('name', 'N/A')}"
                )
                return None

            # 创建新实体
            entity = Entity(
                id=str(uuid.uuid4()),
                source_config_id=self.config.source_config_id,
                entity_type_id=entity_type.id,
                type=entity_data["type"],
                name=entity_data["name"],
                normalized_name=normalized_name,
                description=None,
            )

            # 解析类型化值
            typed_fields = self._parse_entity_value(entity_data, entity_type)
            if typed_fields:
                source = typed_fields.pop("_source", "code")
                entity.value_type = typed_fields.get("value_type")
                entity.value_raw = typed_fields.get("value_raw")
                entity.int_value = typed_fields.get("int_value")
                entity.float_value = typed_fields.get("float_value")
                entity.datetime_value = typed_fields.get("datetime_value")
                entity.bool_value = typed_fields.get("bool_value")
                entity.enum_value = typed_fields.get("enum_value")
                entity.value_unit = typed_fields.get("value_unit")
                entity.value_confidence = typed_fields.get("value_confidence")

                # 记录非text类型的解析结果
                if entity.value_type and entity.value_type != "text":
                    unit_str = f", unit={entity.value_unit}" if entity.value_unit else ""
                    logger.debug(
                        f"实体值解析: {entity_data['name'][:15]} → "
                        f"{entity.value_type}({source}){unit_str}"
                    )

            try:
                session.add(entity)
                await session.commit()
                await session.refresh(entity)
                return entity
            except IntegrityError:
                # 并发冲突：重新查询
                await session.rollback()
                logger.debug(f"实体并发冲突，重新查询: {entity_data['name']}")

                retry = await session.execute(
                    select(Entity)
                    .where(Entity.source_config_id == self.config.source_config_id)
                    .where(Entity.type == entity_data["type"])
                    .where(Entity.normalized_name == normalized_name)
                )
                return retry.scalar_one_or_none()

    def _parse_entity_value(self, entity_data: Dict, entity_type: DBEntityType) -> Dict[str, Any]:
        """
        解析实体值类型（LLM优先 + 代码兜底）

        优先级：
        1. LLM 返回的 value_type（如果有效）
        2. 代码兜底解析（使用 EntityValueParser）

        Args:
            entity_data: 实体数据字典，可能包含 value_type, value, unit
            entity_type: 实体类型对象，包含 value_constraints

        Returns:
            类型化字段字典，包含 value_type, value_raw, int_value 等
        """
        name = entity_data["name"]
        llm_type = entity_data.get("value_type")
        llm_value = entity_data.get("value")
        llm_unit = entity_data.get("unit")

        valid_types = ("text", "int", "float", "datetime", "bool", "enum")
        value_constraints = getattr(entity_type, "value_constraints", None)

        # LLM 有效 → 使用 LLM 返回的类型和值
        if llm_type and llm_type in valid_types:
            value_raw = self._build_value_raw(name, llm_value, llm_unit, llm_type)

            fields = {
                "value_type": llm_type,
                "value_raw": value_raw,
                "int_value": None,
                "float_value": None,
                "datetime_value": None,
                "bool_value": None,
                "enum_value": None,
                "value_unit": llm_unit,
                "value_confidence": Decimal("0.90"),
            }

            # 使用 EntityValueParser 解析 LLM 返回的值
            if llm_value:
                try:
                    parsed = self.value_parser.parse(
                        llm_value,
                        entity_type=entity_data["type"],
                        value_constraints=value_constraints,
                    )
                    if parsed:
                        if parsed["type"] == "int":
                            fields["int_value"] = parsed["value"]
                        elif parsed["type"] == "float":
                            fields["float_value"] = Decimal(str(parsed["value"]))
                        elif parsed["type"] == "datetime":
                            fields["datetime_value"] = parsed["value"]
                        elif parsed["type"] == "bool":
                            fields["bool_value"] = parsed["value"]
                        elif parsed["type"] == "enum":
                            fields["enum_value"] = parsed["value"]
                        if parsed.get("unit") and not llm_unit:
                            fields["value_unit"] = parsed["unit"]
                except Exception as e:
                    logger.debug(f"LLM 值解析失败: {llm_value}, error={e}")

            fields["_source"] = "llm"
            return fields

        # 代码兜底：使用 EntityValueParser 从 name 中解析
        result = self.value_parser.parse_to_typed_fields(
            name, entity_type=entity_data["type"], value_constraints=value_constraints
        )
        result["_source"] = "code"
        return result

    def _build_value_raw(self, name: str, value: str, unit: str, value_type: str) -> str:
        """构建完整的 value_raw"""
        name = name or ""

        if value_type == "text" or not value:
            return name

        if value_type == "datetime":
            if re.search(r"(\d{4}年|\d{1,2}月|\d{1,2}日|\d{1,2}[:：点]|\d{4}-)", name):
                return name
            return name + value

        if re.search(r"\d", name):
            if unit and unit not in name:
                return name + unit
            return name

        result = name + value
        if unit and unit not in result:
            result += unit

        return result


class EntityValueParser:
    """
    实体值类型解析器

    功能：将实体名称文本解析为类型化值
    支持类型：int、float、datetime、bool、enum、text
    """

    # 中文数字映射
    CN_NUM_MAP = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "百": 100,
        "千": 1000,
        "万": 10000,
        "亿": 100000000,
        "兆": 1000000000000,
    }

    # 单位倍数映射
    UNIT_MULTIPLIER = {
        "元": 1,
        "美元": 1,
        "USD": 1,
        "$": 1,
        "万": 10000,
        "万元": 10000,
        "亿": 100000000,
        "亿元": 100000000,
        "克": 0.001,
        "g": 0.001,
        "kg": 1,
        "公斤": 1,
        "千克": 1,
        "吨": 1000,
        "米": 1,
        "m": 1,
        "公里": 1000,
        "km": 1000,
        "厘米": 0.01,
        "cm": 0.01,
        "秒": 1,
        "s": 1,
        "分钟": 60,
        "小时": 3600,
        "天": 86400,
    }

    # 布尔值映射
    BOOL_TRUE = ["是", "对", "真", "yes", "true", "已", "有", "启用", "开启"]
    BOOL_FALSE = ["否", "错", "假", "no", "false", "未", "无", "禁用", "关闭"]

    def parse(
        self,
        text: str,
        entity_type: Optional[str] = None,
        entity_type_category: Optional[str] = None,
        value_constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        解析实体值

        Args:
            text: 原始文本
            entity_type: 实体类型
            entity_type_category: 属性类型类别
            value_constraints: 值约束

        Returns:
            解析结果字典 {type, raw, value, unit, confidence} 或 None
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # 严格模式：如果配置了 value_constraints.type，强制按该类型解析
        if value_constraints and "type" in value_constraints:
            constraint_type = value_constraints["type"]
            result = None

            if constraint_type == "int":
                result = self._parse_number(text, entity_type, value_constraints, force_int=True)
            elif constraint_type == "float":
                result = self._parse_number(text, entity_type, value_constraints, force_float=True)
            elif constraint_type == "enum":
                result = self._parse_enum(text, entity_type, value_constraints)
            elif constraint_type == "datetime":
                result = self._parse_compact_datetime(text) or self._parse_datetime(
                    text, entity_type, value_constraints
                )
            elif constraint_type == "bool":
                result = self._parse_bool(text, entity_type, value_constraints)
            elif constraint_type == "text":
                result = self._parse_text(text)

            if result:
                result["raw"] = text
            return result

        # 兼容模式：按优先级尝试各种类型解析
        time_keywords = ["time", "date", "时间", "日期", "datetime"]
        is_time_type = (entity_type_category and entity_type_category.lower() in time_keywords) or (
            entity_type
            and any(kw in entity_type.lower() for kw in ["时间", "日期", "time", "date"])
        )

        if is_time_type:
            compact_result = self._parse_compact_datetime(text)
            if compact_result:
                compact_result["raw"] = text
                return compact_result

        parsers = [
            self._parse_datetime,
            self._parse_number,
            self._parse_enum,
            self._parse_bool,
        ]

        for parser in parsers:
            result = parser(text, entity_type, value_constraints)
            if result:
                result["raw"] = text
                return result

        return {"type": "text", "raw": text, "value": text, "unit": None, "confidence": 1.0}

    def _parse_number(
        self,
        text: str,
        _entity_type: Optional[str] = None,
        _value_constraints: Optional[Dict[str, Any]] = None,
        force_int: bool = False,
        force_float: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """解析数值类型"""
        configured_unit = _value_constraints.get("unit") if _value_constraints else None
        if configured_unit:
            unit_match_result = self._try_parse_with_unit(
                text, configured_unit, force_int, force_float
            )
            if unit_match_result:
                return unit_match_result

        pattern = r"^([\d,.]+(?:e[+-]?\d+)?)\s*([a-zA-Z\u4e00-\u9fa5]*?)$"
        match = re.match(pattern, text, re.IGNORECASE)

        if match:
            number_str = match.group(1).replace(",", "")
            unit = match.group(2).strip() or None

            try:
                if force_int:
                    if "." in number_str or "e" in number_str.lower():
                        return None
                    num = int(number_str)
                    if unit and unit in self.UNIT_MULTIPLIER:
                        num = num * self.UNIT_MULTIPLIER[unit]
                    return {"type": "int", "value": int(num), "unit": unit, "confidence": 0.95}

                if force_float:
                    if "e" in number_str.lower():
                        num = float(number_str)
                    elif "." in number_str:
                        num = float(number_str)
                    else:
                        num = int(number_str)
                    if unit and unit in self.UNIT_MULTIPLIER:
                        num = num * self.UNIT_MULTIPLIER[unit]
                    return {"type": "float", "value": float(num), "unit": unit, "confidence": 0.95}

                if "e" in number_str.lower():
                    num = float(number_str)
                elif "." in number_str:
                    num = float(number_str)
                else:
                    num = int(number_str)

                if unit and unit in self.UNIT_MULTIPLIER:
                    num = num * self.UNIT_MULTIPLIER[unit]

                if isinstance(num, int):
                    value_type = "int"
                    value = num
                elif isinstance(num, float) and num.is_integer():
                    value_type = "int"
                    value = int(num)
                else:
                    value_type = "float"
                    value = float(num)

                return {"type": value_type, "value": value, "unit": unit, "confidence": 0.95}
            except ValueError:
                pass

        cn_result = self._parse_chinese_number(text)
        if cn_result:
            if force_int and cn_result["type"] != "int":
                return None
            if force_float and cn_result["type"] != "float":
                cn_result["type"] = "float"
                cn_result["value"] = float(cn_result["value"])
            return cn_result

        return None

    def _parse_chinese_number(self, text: str) -> Optional[Dict[str, Any]]:
        """解析中文数字"""
        if len(text) > 6:
            return None

        pattern = r"^([一二三四五六七八九十百千万亿兆]+)$"
        match = re.match(pattern, text)

        if not match:
            return None

        cn_text = match.group(1)
        try:
            value = self._simple_chinese_to_num(cn_text)
            if value is not None:
                return {"type": "int", "value": value, "unit": None, "confidence": 0.85}
        except Exception as e:
            logging.debug(f"中文数字解析失败: {text}, error={e}")

        return None

    def _try_parse_with_unit(
        self,
        text: str,
        configured_unit: str,
        force_int: bool = False,
        force_float: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """智能单位匹配解析"""
        quantifiers = ["个", "件", "条", "项", "批", "次", "笔", "单", "组"]

        for quantifier in quantifiers + [""]:
            pattern = rf"^([\d,.]+(?:e[+-]?\d+)?){quantifier}{re.escape(configured_unit)}$"
            match = re.match(pattern, text, re.IGNORECASE)

            if match:
                number_str = match.group(1).replace(",", "")
                try:
                    if force_int:
                        if "." in number_str or "e" in number_str.lower():
                            continue
                        num = int(number_str)
                        return {
                            "type": "int",
                            "value": num,
                            "unit": configured_unit,
                            "confidence": 0.95,
                        }
                    elif force_float:
                        num = float(number_str)
                        return {
                            "type": "float",
                            "value": num,
                            "unit": configured_unit,
                            "confidence": 0.95,
                        }
                    else:
                        if "." in number_str or "e" in number_str.lower():
                            num = float(number_str)
                            return {
                                "type": "float",
                                "value": num,
                                "unit": configured_unit,
                                "confidence": 0.95,
                            }
                        else:
                            num = int(number_str)
                            return {
                                "type": "int",
                                "value": num,
                                "unit": configured_unit,
                                "confidence": 0.95,
                            }
                except ValueError:
                    continue

            # 尝试中文数字
            cn_pattern = (
                rf"^([一二三四五六七八九十百千万亿兆]+){quantifier}{re.escape(configured_unit)}$"
            )
            cn_match = re.match(cn_pattern, text)

            if cn_match:
                cn_text = cn_match.group(1)
                try:
                    value = self._simple_chinese_to_num(cn_text)
                    if value is not None:
                        if force_float:
                            return {
                                "type": "float",
                                "value": float(value),
                                "unit": configured_unit,
                                "confidence": 0.90,
                            }
                        else:
                            return {
                                "type": "int",
                                "value": value,
                                "unit": configured_unit,
                                "confidence": 0.90,
                            }
                except Exception as e:
                    logging.debug(f"中文数字单位匹配失败: {text}, error={e}")
                    continue

        return None

    def _simple_chinese_to_num(self, cn_text: str) -> Optional[int]:
        """简化的中文数字转换"""
        total = 0
        unit = 1

        for char in reversed(cn_text):
            num = self.CN_NUM_MAP.get(char)
            if num is None:
                return None

            if num >= 10:
                if num > unit:
                    unit = num
                else:
                    unit *= num
            else:
                total += num * unit

        return total if total > 0 else None

    def _parse_datetime(
        self,
        text: str,
        _entity_type: Optional[str] = None,
        _value_constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """解析时间类型"""
        text_clean = text.replace("年", "-").replace("月", "-").replace("日", "")
        text_clean = re.sub(r"-+", "-", text_clean)
        text_clean = text_clean.strip("-")

        # 完整日期时间
        pattern_datetime = r"(\d{4})-(\d{1,2})-(\d{1,2})[\sT]+(\d{1,2}):(\d{1,2}):(\d{1,2})"
        match = re.search(pattern_datetime, text_clean)
        if match:
            try:
                dt = datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                    int(match.group(5)),
                    int(match.group(6)),
                )
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.98}
            except ValueError:
                pass

        # 日期时间（到分钟）
        pattern_datetime_min = r"(\d{4})-(\d{1,2})-(\d{1,2})[\sT]+(\d{1,2}):(\d{1,2})"
        match = re.search(pattern_datetime_min, text_clean)
        if match:
            try:
                dt = datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                    int(match.group(5)),
                )
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.95}
            except ValueError:
                pass

        # ISO 日期
        pattern_iso = r"(\d{4})-(\d{1,2})-(\d{1,2})"
        match = re.search(pattern_iso, text_clean)
        if match:
            try:
                dt = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.98}
            except ValueError:
                pass

        # YYYY-MM
        pattern_month = r"(\d{4})-(\d{1,2})(?:[^\d]|$)"
        match = re.search(pattern_month, text_clean)
        if match:
            try:
                dt = datetime(int(match.group(1)), int(match.group(2)), 1)
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.90}
            except ValueError:
                pass

        # 仅年份
        money_patterns = [
            r"元",
            r"美元",
            r"USD",
            r"\$",
            r"¥",
            r"万元",
            r"亿元",
            r"块",
            r"角",
            r"分",
            r"块钱",
            r"毛",
            r"EUR",
            r"£",
            r"RMB",
            r"CNY",
            r"HKD",
            r"/月",
            r"/年",
            r"/天",
            r"/周",
            r"每月",
            r"每年",
            r"起",
            r"左右",
            r"约",
        ]
        has_money_indicator = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in money_patterns
        )

        pattern_year = r"^(\d{4})(?:[^\d]|$)"
        match = re.match(pattern_year, text_clean)
        if match and not has_money_indicator:
            year = int(match.group(1))
            if 1970 <= year <= 2099:
                try:
                    dt = datetime(year, 1, 1)
                    return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.85}
                except ValueError:
                    pass

        return None

    def _parse_compact_datetime(self, text: str) -> Optional[Dict[str, Any]]:
        """解析紧凑日期格式"""
        # YYYYMMDDHHmmss（14位）
        if re.match(r"^(\d{14})$", text):
            try:
                year = int(text[0:4])
                month = int(text[4:6])
                day = int(text[6:8])
                hour = int(text[8:10])
                minute = int(text[10:12])
                second = int(text[12:14])
                dt = datetime(year, month, day, hour, minute, second)
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.95}
            except ValueError:
                return None

        # YYYYMMDDHHmm（12位）
        if re.match(r"^(\d{12})$", text):
            try:
                year = int(text[0:4])
                month = int(text[4:6])
                day = int(text[6:8])
                hour = int(text[8:10])
                minute = int(text[10:12])
                dt = datetime(year, month, day, hour, minute)
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.92}
            except ValueError:
                return None

        # YYYYMMDD（8位）
        if re.match(r"^(\d{8})$", text):
            try:
                year = int(text[0:4])
                month = int(text[4:6])
                day = int(text[6:8])
                dt = datetime(year, month, day)
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.95}
            except ValueError:
                return None

        # YYYYMM（6位）
        if re.match(r"^(\d{6})$", text):
            try:
                year = int(text[0:4])
                month = int(text[4:6])
                dt = datetime(year, month, 1)
                return {"type": "datetime", "value": dt, "unit": None, "confidence": 0.90}
            except ValueError:
                return None

        return None

    def _parse_bool(
        self,
        text: str,
        _entity_type: Optional[str] = None,
        _value_constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """解析布尔类型"""
        if len(text) > 10:
            return None

        text_lower = text.lower().strip()

        if text_lower in self.BOOL_TRUE:
            return {"type": "bool", "value": True, "unit": None, "confidence": 0.95}

        if text_lower in self.BOOL_FALSE:
            return {"type": "bool", "value": False, "unit": None, "confidence": 0.95}

        return None

    def _parse_enum(
        self,
        text: str,
        _entity_type: Optional[str] = None,
        value_constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """解析枚举类型"""
        if not value_constraints or "enum_values" not in value_constraints:
            return None

        enum_values: List[str] = value_constraints["enum_values"]

        # 精确匹配
        if text in enum_values:
            return {"type": "enum", "value": text, "unit": None, "confidence": 1.0}

        # 模糊匹配
        text_lower = text.lower()
        for enum_val in enum_values:
            if enum_val.lower() in text_lower or text_lower in enum_val.lower():
                return {"type": "enum", "value": enum_val, "unit": None, "confidence": 0.80}

        return {"type": "enum", "value": "UNKNOWN", "unit": None, "confidence": 0.0}

    def _parse_text(self, text: str) -> Optional[Dict[str, Any]]:
        """解析为纯文本类型"""
        return {"type": "text", "value": text, "unit": None, "confidence": 1.0}

    def parse_to_typed_fields(
        self,
        text: str,
        entity_type: Optional[str] = None,
        entity_type_category: Optional[str] = None,
        value_constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """解析为类型化字段（直接映射到数据库字段）"""
        result = self.parse(text, entity_type, entity_type_category, value_constraints)

        if not result:
            result = {
                "type": "text",
                "raw": text or "",
                "value": text or "",
                "unit": None,
                "confidence": 1.0,
            }

        typed_fields = {
            "value_type": result["type"],
            "value_raw": result["raw"],
            "int_value": None,
            "float_value": None,
            "datetime_value": None,
            "bool_value": None,
            "enum_value": None,
            "value_unit": result.get("unit"),
            "value_confidence": Decimal(str(result.get("confidence", 1.0))),
        }

        value = result.get("value")
        if result["type"] == "int":
            typed_fields["int_value"] = value
        elif result["type"] == "float":
            typed_fields["float_value"] = Decimal(str(value))
        elif result["type"] == "datetime":
            typed_fields["datetime_value"] = value
        elif result["type"] == "bool":
            typed_fields["bool_value"] = value
        elif result["type"] == "enum":
            typed_fields["enum_value"] = value

        return typed_fields
