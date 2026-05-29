"""
Entity Data Models
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from pipeline.models.base import pipelineBaseModel, MetadataMixin, TimestampMixin


class EntityType(pipelineBaseModel, MetadataMixin, TimestampMixin):
    """Entity type definition model"""

    id: str = Field(..., description="Entity type ID (UUID)")
    scope: str = Field(
        default="global", description="Scope: global/source/article")
    source_config_id: Optional[str] = Field(
        default=None, description="Source config ID (NULL for system default)")
    article_id: Optional[str] = Field(
        default=None, description="Article ID (only when scope=article)")
    type: str = Field(..., min_length=1, max_length=50, description="Type identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Type name")
    is_default: bool = Field(default=False, description="Is system default type")
    description: Optional[str] = Field(default=None, description="Type description")
    weight: float = Field(default=1.0, ge=0.0, le=9.99, description="Default weight")
    similarity_threshold: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Entity similarity threshold (0.000-1.000)"
    )
    is_active: bool = Field(default=True, description="Is active")
    value_format: Optional[str] = Field(
        default=None, description="Value format template (e.g. {number}{unit})")
    value_constraints: Optional[Dict[str, Any]] = Field(
        default=None, description="Value constraints (e.g. enum list, number range)")

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight range"""
        return round(v, 2)

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold and keep 3 decimal places"""
        return round(v, 3)


class Entity(pipelineBaseModel, MetadataMixin, TimestampMixin):
    """Entity model (many-to-many: linked to events via event_entity)"""

    id: str = Field(..., description="Entity ID (UUID)")
    source_config_id: str = Field(..., description="Source config ID")
    entity_type_id: str = Field(..., description="Entity type ID (references entity_type.id)")
    type: str = Field(
        ..., min_length=1, max_length=50, description="Entity type identifier (redundant field for query)"
    )
    name: str = Field(..., min_length=1, max_length=500, description="Entity name")
    normalized_name: str = Field(..., min_length=1,
                                 max_length=500, description="Normalized name")
    description: Optional[str] = Field(default=None, description="Entity description")

    # ========== Typed value fields (for statistical analysis) ==========
    value_type: Optional[str] = Field(
        default=None, description="Value type (int/float/datetime/bool/enum/text)")
    value_raw: Optional[str] = Field(
        default=None, description="Raw extracted text (e.g. '$199')")
    int_value: Optional[int] = Field(default=None, description="Integer value")
    float_value: Optional[Decimal] = Field(default=None, description="Float value")
    datetime_value: Optional[datetime] = Field(
        default=None, description="Datetime value")
    bool_value: Optional[bool] = Field(default=None, description="Boolean value")
    enum_value: Optional[str] = Field(default=None, description="Enum value")
    value_unit: Optional[str] = Field(
        default=None, description="Unit (e.g. 'USD', 'kg')")
    value_confidence: Optional[Decimal] = Field(
        default=None, ge=0.0, le=1.0, description="Parsing confidence")

    def get_typed_value(self) -> Any:
        """Get typed value based on value_type"""
        if self.value_type == "int":
            return self.int_value
        elif self.value_type == "float":
            return self.float_value
        elif self.value_type == "datetime":
            return self.datetime_value
        elif self.value_type == "bool":
            return self.bool_value
        elif self.value_type == "enum":
            return self.enum_value
        return None

    def get_synonyms(self) -> List[str]:
        """Get synonyms"""
        if self.extra_data and "synonyms" in self.extra_data:
            return self.extra_data["synonyms"]
        return []

    def get_weight(self) -> float:
        """Get weight"""
        if self.extra_data and "weight" in self.extra_data:
            return self.extra_data["weight"]
        return 1.0

    def get_confidence(self) -> float:
        """Get confidence"""
        if self.extra_data and "confidence" in self.extra_data:
            return self.extra_data["confidence"]
        return 1.0


class EventEntity(pipelineBaseModel, MetadataMixin, TimestampMixin):
    """Event-Entity association model (many-to-many)"""

    id: str = Field(..., description="Association ID (UUID)")
    event_id: str = Field(..., description="Event ID")
    entity_id: str = Field(..., description="Entity ID")
    weight: float = Field(default=1.0, ge=0.0, le=9.99,
                          description="Entity weight in this event")

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight range"""
        return round(v, 2)

    def get_confidence(self) -> float:
        """Get confidence"""
        if self.extra_data and "confidence" in self.extra_data:
            return self.extra_data["confidence"]
        return 1.0

    def get_context(self) -> Optional[str]:
        """Get context"""
        if self.extra_data and "context" in self.extra_data:
            return self.extra_data["context"]
        return None


class CustomEntityType(pipelineBaseModel):
    """Custom entity type definition"""

    type: str = Field(..., description="Type identifier")
    name: str = Field(..., description="Type name")
    description: str = Field(..., description="Type description for LLM extraction")
    weight: float = Field(default=1.0, ge=0.0, le=9.99, description="Default weight")
    extraction_prompt: Optional[str] = Field(
        default=None, description="Custom extraction prompt template")
    extraction_examples: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Few-shot examples"
    )
    validation_rule: Optional[Dict[str, Any]] = Field(
        default=None, description="Validation rule")
    metadata_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata schema")


# ==============================================================================
# 默认实体类型定义 (基于5W1H框架)
# ==============================================================================
# 
# 设计原则:
# 1. 本体论类型 - 实体"是什么"，而非"用来做什么"
# 2. 互斥性 - 每个实体只属于一个类型
# 3. 完备性 - 覆盖所有可能的实体类型
# 4. 搜索导向 - 支持LLM识别"线索维度"和"目标维度"
#
# 权重说明 (按搜索重要性分层):
# - 高权重 (1.2~1.5): subject(1.5), action(1.3), metric(1.2), person(1.2)
# - 中权重 (1.0~1.1): organization(1.1), product(1.1), group/work/time/location(1.0)
# - 兜底 (0.5): tags - 避免滥用
#
# 总计: 11个类型，覆盖95%+问答场景
# - 时间(WHEN): time
# - 空间(WHERE): location
# - 主体(WHO): person, organization, group
# - 内容(WHAT): subject, work, product
# - 方式(HOW): action, metric
# - 兜底: tags
#
# ==============================================================================

DEFAULT_ENTITY_TYPES = [
    # ==========================================================================
    # 【WHEN - 时间维度】
    # ==========================================================================
    EntityType(
        id="30000000-0000-0000-0000-000000000001",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="time",
        name="时间",
        is_default=True,
        description="事件发生的时间点、时间范围或周期。如：具体日期、时间段、节假日、季节、年代。示例：2024年、第三季度、春节、周一。",
        weight=1.0,
        similarity_threshold=0.900,
    ),
    
    # ==========================================================================
    # 【WHERE - 空间维度】
    # ==========================================================================
    EntityType(
        id="30000000-0000-0000-0000-000000000002",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="location",
        name="地点",
        is_default=True,
        description="地理位置、行政区划、物理空间、建筑。如：国家、城市、区域、地标、场馆、历史上存在的地理位置、公园。示例：北京、纽约、东京、巴黎、罗马共和国。",
        weight=1.0,
        similarity_threshold=0.750,
    ),
    
    # ==========================================================================
    # 【WHO - 主体维度】
    # ==========================================================================
    EntityType(
        id="30000000-0000-0000-0000-000000000003",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="person",
        name="人物",
        is_default=True,
        description="具体的一个自然人，真实或虚构。如：姓名、艺名、笔名、历史人物、虚构角色。示例：爱因斯坦、莎士比亚、乔布斯。",
        weight=1.2,
        similarity_threshold=0.950,
    ),
    EntityType(
        id="30000000-0000-0000-0000-000000000004",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="organization",
        name="组织",
        is_default=True,
        description="机构、组织或团队名称。如：公司、政府机构、学校、教育机构、NGO、平台、体育团队、乐队、军事组织、政党、委员会。示例：谷歌、联合国、哈佛大学。",
        weight=1.1,
        similarity_threshold=0.850,
    ),
    EntityType(
        id="30000000-0000-0000-0000-000000000005",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="group",
        name="群体",
        is_default=True,
        description="基于共同的人口统计学特征、社会身份、职业、角色或行为特征而形成的、非正式或抽象的人群集合/身份标签。如：年龄群体、职业群体、消费群体、社会身份。示例：青少年、医生、用户、投资者。",
        weight=1.0,
        similarity_threshold=0.700,
    ),
    
    # ==========================================================================
    # 【WHAT - 内容维度】(核心搜索维度)
    # ==========================================================================
    EntityType(
        id="30000000-0000-0000-0000-000000000006",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="subject",
        name="话题",
        is_default=True,
        description="被讨论的核心话题、概念、领域或现象。包括：技术趋势、社会现象、专业领域、历史事件、奖项荣誉、赛事活动等。示例：人工智能、气候变化、辛亥革命、诺贝尔奖。",
        weight=1.5,
        similarity_threshold=0.600,
    ),
    EntityType(
        id="30000000-0000-0000-0000-000000000007",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="work",
        name="作品",
        is_default=True,
        description="人类创作的智力成果或文化产品，通常有书名号《》或者斜体英文标识。如：书籍、影视作品、音乐、小说、论文、游戏。示例：《三体》、《星球大战》、《哈姆雷特》。",
        weight=1.0,
        similarity_threshold=0.850,
    ),
    EntityType(
        id="30000000-0000-0000-0000-000000000008",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="product",
        name="产品",
        is_default=True,
        description="可购买或使用的商品、服务，不含技术概念、创作内容或架构。如：硬件产品、软件产品、服务、品牌。示例：iPhone、可口可乐、Windows。",
        weight=1.1,
        similarity_threshold=0.800,
    ),
    
    # ==========================================================================
    # 【HOW - 方式维度】
    # ==========================================================================
    EntityType(
        id="30000000-0000-0000-0000-000000000009",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="action",
        name="行为",
        is_default=True,
        description="主体执行的动作、操作或方法。如：业务动作、用户行为、方法策略、流程步骤，强调动作本身而非结果。示例：发布、收购、合作、投资。",
        weight=1.3,
        similarity_threshold=0.800,
    ),
    EntityType(
        id="30000000-0000-0000-0000-000000000010",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="metric",
        name="指标",
        is_default=True,
        description="可量化的度量，必须包含具体数值，保留原始格式(含%、万、亿等单位)。如：占比、比例、数量、参数量。示例：12%、增长31%、100万、137B。",
        weight=1.2,
        similarity_threshold=0.800,
    ),
    
    # ==========================================================================
    # 【兜底维度】
    # ==========================================================================
    EntityType(
        id="30000000-0000-0000-0000-000000000011",
        scope="global",
        source_config_id=None,
        article_id=None,
        type="tags",
        name="标签",
        is_default=True,
        description="无法归入以上任何类型的实体，仅作兜底使用。",
        weight=0.5,
        similarity_threshold=0.700,
    ),
]
