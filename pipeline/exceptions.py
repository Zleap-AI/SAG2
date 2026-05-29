"""
pipeline 异常定义

所有自定义异常都继承自 pipelineError 基类
"""


class pipelineError(Exception):
    """pipeline 基础异常类"""

    def __init__(self, message: str, *args: object) -> None:
        self.message = message
        super().__init__(message, *args)


class ConfigError(pipelineError):
    """配置错误异常"""

    pass


class StorageError(pipelineError):
    """存储层异常"""

    pass


class DatabaseError(StorageError):
    """数据库异常"""

    pass


class CacheError(StorageError):
    """缓存异常"""

    pass


class LLMError(pipelineError):
    """LLM调用异常"""

    pass


class LLMTimeoutError(LLMError):
    """LLM调用超时异常"""

    pass


class LLMRateLimitError(LLMError):
    """LLM速率限制异常"""

    pass


class AIError(pipelineError):
    """AI相关异常（包括LLM和Embedding）"""

    pass


class ValidationError(pipelineError):
    """数据验证异常"""

    pass


class LoadError(pipelineError):
    """文档加载异常"""

    pass


class EntityError(pipelineError):
    """实体处理异常"""

    pass


class ExtractError(pipelineError):
    """事项提取异常"""

    pass


class SearchError(pipelineError):
    """检索异常"""

    pass


class PromptError(pipelineError):
    """提示词异常"""

    pass


# ============ 可重试异常 ============


class RetryableError(pipelineError):
    """可重试异常基类（临时性错误，重试可能成功）"""

    pass


class NetworkError(RetryableError):
    """网络错误（连接超时、网络中断等）"""

    pass


class ResourceBusyError(RetryableError):
    """资源繁忙错误（数据库锁、并发冲突等）"""

    pass


class ServiceUnavailableError(RetryableError):
    """服务不可用错误（外部服务暂时不可用）"""

    pass


# ============ 不可重试异常 ============


class NonRetryableError(pipelineError):
    """不可重试异常基类（永久性错误，重试无意义）"""

    pass


class InvalidInputError(NonRetryableError):
    """无效输入错误（数据格式错误、参数非法等）"""

    pass


class ResourceNotFoundError(NonRetryableError):
    """资源不存在错误（文件不存在、记录不存在等）"""

    pass


class PermissionError(NonRetryableError):
    """权限错误（访问被拒绝、认证失败等）"""

    pass
