"""
配置管理模块

使用pydantic-settings管理配置，支持从环境变量和.env文件读取
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    """查找项目根目录（包含.env文件的目录）"""
    current = Path(__file__).resolve()

    # 向上查找包含 .env 的目录
    for parent in [current.parent] + list(current.parents):
        env_file = parent / ".env"
        if env_file.exists():
            return parent

    # 如果找不到，返回当前文件所在的项目根目录（假设在 pipeline/core/config/）
    return current.parent.parent.parent


# LLM 可靠性常量（统一默认值，避免多处硬编码）
DEFAULT_LLM_MAX_RETRIES = 5


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=str(_find_project_root() / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ======================
    # 数据库配置
    # ======================
    mysql_host: str = Field(default="localhost", description="MySQL主机")
    mysql_port: int = Field(default=3306, description="MySQL端口")
    mysql_user: str = Field(default="sag2", description="MySQL用户")
    mysql_password: str = Field(default="sag2", description="MySQL密码")
    mysql_database: str = Field(default="sag2", description="MySQL数据库名")

    # ======================
    # Elasticsearch配置
    # ======================
    es_host: str = Field(default="localhost", description="ES主机")
    es_port: int = Field(default=9201, description="ES端口")
    es_scheme: str = Field(default="http", description="ES协议(http/https)")
    es_username: Optional[str] = Field(default="elastic", description="ES用户名")
    es_password: Optional[str] = Field(
        default=None, description="ES密码", validation_alias="ELASTIC_PASSWORD"
    )

    # ======================
    # LLM配置（使用中转API或OpenAI官方）
    # ======================
    llm_api_key: str = Field(default="", description="LLM API密钥")
    llm_model: str = Field(default="sophnet/Qwen3-30B-A3B-Thinking-2507", description="LLM模型")
    llm_base_url: Optional[str] = Field(
        default=None, description="LLM API基础URL（留空使用OpenAI官方）"
    )
    llm_data_inspection: bool = Field(
        default=False, description="是否启用LLM内容过滤（绿网），默认关闭"
    )

    # 是否启用模型的思考模式（enable_thinking），默认关闭，需在.env中显式启用
    llm_enable_think: bool = Field(
        default=False, description="是否启用模型的思考模式（enable_thinking）"
    )

    # LLM 行为参数：默认值仅在此处定义；全局配置 = 环境变量（若有）否则用此处默认值；可被数据库配置覆盖
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM温度参数")
    llm_max_tokens: int = Field(default=30000, ge=1, description="LLM最大输出token数")
    llm_top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="LLM top_p参数")
    llm_frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    llm_presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")

    # LLM 可靠性参数
    llm_timeout: int = Field(default=300, ge=1, description="LLM超时时间(秒)")
    llm_max_retries: int = Field(default=DEFAULT_LLM_MAX_RETRIES, ge=0, description="LLM最大重试次数")

    # 数据库配置开关
    use_db_config: bool = Field(default=True, description="是否使用数据库配置")

    # ======================
    # Embedding配置（使用中转API或OpenAI官方）
    # ======================
    embedding_api_key: str = Field(
        default="", description="Embedding API密钥（留空使用llm_api_key）"
    )
    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B", description="Embedding模型"
    )
    embedding_dimensions: Optional[int] = Field(
        default=None,
        description="Embedding维度（可选，留空则使用模型默认维度。text-embedding-3-small默认1536，text-embedding-3-large默认3072）",
    )
    embedding_base_url: Optional[str] = Field(
        default=None, description="Embedding API基础URL（留空使用llm_base_url）"
    )

    # ======================
    # LLM 语言配置
    # ======================
    llm_language: str = Field(
        default="zh", description="LLM输出语言(zh/en)，决定加载哪个语言版本的提示词"
    )

    # ======================
    # 应用配置
    # ======================
    server_type: str = Field(
        default="LOCAL", description="服务环境类型（SAAS/LOCAL）"
    )
    benchmark: bool = Field(default=False, description="Benchmark模式，跳过LLM调用")
    debug: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(default="json", description="日志格式")

    # ======================
    # MLflow 配置
    # ======================
    mlflow_port: int = Field(default=5000, description="MLflow Docker容器端口")
    mlflow_url: Optional[str] = Field(
        default="http://localhost:5000", description="MLflow Tracking Server地址"
    )

    # 实体权重配置
    # entity_weights: str = Field(
    #     default="time:0.9,location:1.0,person:1.1,topic:1.5,action:1.2,tags:1.0",
    #     description="实体类型权重",
    # )

    # ======================
    # 性能配置
    # ======================
    db_pool_size: int = Field(default=100, description="数据库连接池大小")
    db_max_overflow: int = Field(default=200, description="数据库连接池最大溢出")
    db_pool_recycle: int = Field(default=3600, description="数据库连接回收时间(秒)")

    # 缓存TTL
    cache_entity_ttl: int = Field(default=86400, description="实体缓存TTL(秒)")
    cache_llm_ttl: int = Field(default=604800, description="LLM缓存TTL(秒)")
    cache_search_ttl: int = Field(default=3600, description="搜索缓存TTL(秒)")

    
    @property
    def mysql_url(self) -> str:
        """MySQL连接URL"""
        from urllib.parse import quote_plus

        # URL编码用户名和密码，避免特殊字符问题
        encoded_user = quote_plus(self.mysql_user)
        encoded_password = quote_plus(self.mysql_password)
        return (
            f"mysql+aiomysql://{encoded_user}:{encoded_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    @property
    def elasticsearch_url(self) -> str:
        """Elasticsearch连接URL"""
        return f"{self.es_scheme}://{self.es_host}:{self.es_port}"

    @property
    def es_url(self) -> str:
        """Elasticsearch连接URL（兼容旧版本）"""
        return self.elasticsearch_url

    @property
    def amqp_url(self) -> str:
        """AMQP 连接 URL（RabbitMQ）"""
        from urllib.parse import quote_plus

        user = quote_plus(self.rabbitmq_username)
        pwd = quote_plus(self.rabbitmq_password)
        return f"amqp://{user}:{pwd}@{self.rabbitmq_host}:{self.rabbitmq_port}/{self.rabbitmq_vhost}"

    # @property
    # def entity_weights_dict(self) -> Dict[str, float]:
    #     """实体权重字典"""
    #     result = {}
    #     for pair in self.entity_weights.split(","):
    #         if ":" in pair:
    #             key, value = pair.split(":")
    #             try:
    #                 result[key.strip()] = float(value.strip())
    #             except ValueError:
    #                 continue
    #     return result

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"日志级别必须是: {', '.join(allowed)}")
        return v.upper()

    @field_validator("llm_language")
    @classmethod
    def validate_llm_language(cls, v: str) -> str:
        """验证LLM语言配置"""
        allowed = ["zh", "en"]
        if v.lower() not in allowed:
            raise ValueError(f"LLM语言必须是: {', '.join(allowed)}")
        return v.lower()

    @field_validator("server_type")
    @classmethod
    def validate_server_type(cls, v: str) -> str:
        """验证服务环境类型"""
        normalized = v.upper()
        allowed = ["SAAS", "LOCAL"]
        if normalized not in allowed:
            raise ValueError(f"SERVER_TYPE 必须是: {', '.join(allowed)}")
        return normalized


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
