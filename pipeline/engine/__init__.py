"""
pipeline 引擎模块

提供统一的任务引擎接口
"""

from pipeline.engine.config import (
    ModelConfig,
    OutputConfig,
    TaskConfig,
)
from pipeline.engine.core import pipelineEngine
from pipeline.engine.enums import LogLevel, OutputMode, TaskStage, TaskStatus
from pipeline.engine.models import StageResult, TaskLog, TaskResult
from pipeline.modules.extract.config import ExtractBaseConfig
from pipeline.modules.load.config import (
    DocumentLoadConfig,
    LoadBaseConfig,
)
from pipeline.modules.search.config import SearchBaseConfig

__all__ = [
    # 核心引擎
    "pipelineEngine",
    # 配置类
    "ModelConfig",
    "LoadBaseConfig",
    "DocumentLoadConfig",
    "ExtractBaseConfig",
    "SearchBaseConfig",
    "OutputConfig",
    "TaskConfig",
    # 枚举
    "TaskStatus",
    "TaskStage",
    "LogLevel",
    "OutputMode",
    # 模型
    "TaskResult",
    "TaskLog",
    "StageResult",
]
