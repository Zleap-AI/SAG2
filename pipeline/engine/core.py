"""
pipeline 引擎核心类
"""

import time
import uuid
import inspect
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional, Tuple

from sqlalchemy import select

from pipeline.core.prompt.manager import get_prompt_manager
from pipeline.db import SourceConfig, get_session_factory
from pipeline.engine.config import ModelConfig, OutputConfig, TaskConfig
from pipeline.engine.enums import LogLevel, TaskStage, TaskStatus
from pipeline.engine.models import StageResult, TaskLog, TaskResult
from pipeline.modules.extract.config import ExtractBaseConfig, ExtractConfig
from pipeline.modules.extract.extractor import EventExtractor
from pipeline.modules.load.config import DocumentLoadConfig, LoadBaseConfig, LoadResult
from pipeline.modules.load.loader import DocumentLoader
from pipeline.modules.search.config import SearchBaseConfig, SearchConfig
from pipeline.modules.search.searcher import SAGSearcher
from pipeline.utils import get_logger, setup_logging

logger = get_logger("pipeline.engine")


class ProgressData(NamedTuple):
    completed: int
    total: int


class pipelineEngine:
    """
    pipeline 任务引擎，支持三个独立阶段：Load → Extract → Search。

    用法：
        engine = pipelineEngine(task_config=TaskConfig(...))
        await engine.load_async(DocumentLoadConfig(...))
        await engine.extract_async(ExtractBaseConfig(...))
        await engine.search_async(SearchBaseConfig(...))
        result = engine.get_result()
    """

    def __init__(
        self,
        task_config: Optional[TaskConfig] = None,
        model_config: Optional[ModelConfig] = None,
        source_config_id: Optional[str] = None,
        auto_setup_logging: bool = True,
        progress_callback: Optional[Callable[[str], Optional[Awaitable[None]]]] = None,
        stage_callback: Optional[Callable[[str, Any], Optional[Awaitable[None]]]] = None,
    ):
        if auto_setup_logging:
            setup_logging()

        self.task_config = task_config
        self._model_config_dict = model_config.model_dump() if model_config else None
        self._progress_callback = progress_callback
        self._stage_callback = stage_callback
        self.task_id = str(uuid.uuid4())

        # 组件
        self._prompt_manager = get_prompt_manager()
        self._session_factory = get_session_factory()
        self._loader = DocumentLoader(progress_callback=self._relay_loader_progress)
        self._extractor = EventExtractor(
            prompt_manager=self._prompt_manager,
            model_config=self._model_config_dict,
            on_progress=self._relay_extract_progress,
        )
        self._searcher = SAGSearcher(
            prompt_manager=self._prompt_manager,
            model_config=self._model_config_dict,
        )

        task_name = task_config.task_name if task_config else "pipeline任务"
        self.result = TaskResult(task_id=self.task_id, task_name=task_name, status=TaskStatus.PENDING)

        self._source_config_id: Optional[str] = (
            source_config_id or (task_config.source_config_id if task_config else None)
        )
        self._load_result: Optional[LoadResult] = None
        self._start_time: Optional[float] = None

        self._log(TaskStage.INIT, LogLevel.INFO, f"引擎初始化完成: {self.task_id}")

    # ── 回调 ──────────────────────────────────────────────────────────────────

    async def _emit(self, callback, *args):
        """安全调用同步或异步回调。"""
        if not callback:
            return
        try:
            result = callback(*args)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.warning(f"回调执行失败: {e}")

    async def _relay_loader_progress(self, message: str):
        await self._emit(self._progress_callback, message)

    async def _relay_extract_progress(self, completed: int, total: int):
        await self._emit(self._stage_callback, "extract_progress", ProgressData(completed, total))

    # ── 日志 / 状态 ───────────────────────────────────────────────────────────

    def _log(self, stage: TaskStage, level: LogLevel, message: str, extra: Optional[Dict] = None):
        self.result.logs.append(TaskLog(stage=stage, level=level, message=message, extra=extra))
        if not self.task_config or self.task_config.output.print_logs:
            getattr(logger, level.value)(message, extra=extra)

    # ── 数据源 ────────────────────────────────────────────────────────────────

    async def _ensure_source(self) -> str:
        """确保 SourceConfig 存在，返回 source_config_id。"""
        if not self._source_config_id:
            self._source_config_id = str(uuid.uuid4())
            self._log(TaskStage.INIT, LogLevel.INFO, f"自动创建信息源: {self._source_config_id}")

        async with self._session_factory() as session:
            row = await session.execute(
                select(SourceConfig).where(SourceConfig.id == self._source_config_id)
            )
            if not row.scalar_one_or_none():
                source_name = (
                    self.task_config.source_name
                    if self.task_config and self.task_config.source_name
                    else f"pipeline-{uuid.uuid4().hex[:8]}"
                )
                session.add(SourceConfig(
                    id=self._source_config_id,
                    name=source_name,
                    description=f"由pipeline引擎创建 (Task: {self.task_id})",
                    target_config={"task_id": self.task_id},
                ))
                await session.commit()

        self.result.source_config_id = self._source_config_id
        return self._source_config_id

    # ── 通用阶段执行模板 ──────────────────────────────────────────────────────

    @asynccontextmanager
    async def _stage(self, stage: TaskStage, status: TaskStatus):
        """
        统一管理阶段的状态转换、计时、异常捕获。

        用法：
            async with self._stage(TaskStage.LOAD, TaskStatus.LOADING) as timer:
                ...  # 业务逻辑
                yield timer  # timer() 返回已耗时秒数
        """
        self.result.status = status
        start = time.time()
        self._log(stage, LogLevel.INFO, f"开始阶段: {stage.value}")
        try:
            yield lambda: time.time() - start
        except Exception as e:
            elapsed = time.time() - start
            self._log(stage, LogLevel.ERROR, f"{stage.value} 失败: {e}")
            setattr(self.result, f"{stage.value}_result", StageResult(
                stage=stage, status="failed", error=str(e), duration=elapsed,
            ))
            if self.task_config and self.task_config.fail_fast:
                raise

    # ── Load 阶段 ─────────────────────────────────────────────────────────────

    async def load_async(self, config: LoadBaseConfig) -> None:
        """加载文档（异步）。"""
        if not isinstance(config, DocumentLoadConfig):
            raise TypeError(f"不支持的配置类型: {type(config).__name__}，请使用 DocumentLoadConfig")

        await self._emit(self._stage_callback, "load_start", config)

        async with self._stage(TaskStage.LOAD, TaskStatus.LOADING) as elapsed:
            await self._emit(self._progress_callback, "正在载入")
            source_id = await self._ensure_source()
            config.source_config_id = config.source_config_id or source_id

            if self.task_config and self.task_config.background and hasattr(config, "background"):
                config.background = config.background or self.task_config.background

            load_result = await self._loader.load(config)
            self._load_result = load_result
            self.result.article_id = load_result.source_id

            self.result.load_result = StageResult(
                stage=TaskStage.LOAD,
                status="success",
                data_ids=load_result.chunk_ids,
                stats={
                    "source_id": load_result.source_id,
                    "source_type": load_result.source_type,
                    "chunk_count": load_result.chunk_count,
                    "title": load_result.title,
                    **load_result.extra,
                },
                duration=elapsed(),
            )
            self._log(TaskStage.LOAD, LogLevel.INFO,
                      f"加载完成: {load_result.source_type} source_id={load_result.source_id} chunks={load_result.chunk_count}")

        await self._emit(self._stage_callback, "load_done", self._load_result)

    # ── Extract 阶段 ──────────────────────────────────────────────────────────

    async def extract_async(self, config: Optional[ExtractBaseConfig] = None) -> None:
        """提取事项（异步）。前提：必须先执行 load_async。"""
        if not self._load_result or not self._load_result.chunk_ids:
            self._log(TaskStage.EXTRACT, LogLevel.WARNING, "无可用 chunks，跳过提取阶段")
            return

        async with self._stage(TaskStage.EXTRACT, TaskStatus.EXTRACTING) as elapsed:
            await self._emit(self._progress_callback, "正在提取事项")
            source_id = await self._ensure_source()
            cfg = config or ExtractBaseConfig()

            events = await self._extractor.extract(ExtractConfig(
                source_config_id=source_id,
                chunk_ids=self._load_result.chunk_ids,
                **cfg.model_dump(),
            ))

            chunk_count = len(self._load_result.chunk_ids)
            self.result.extract_result = StageResult(
                stage=TaskStage.EXTRACT,
                status="success",
                data_ids=[e.id for e in events],
                data_full=[{
                    "id": e.id, "title": e.title, "summary": e.summary, "content": e.content,
                    "entities": [{"name": a.entity.name, "type": a.entity.type} for a in e.event_associations],
                } for e in events],
                stats={
                    "event_count": len(events),
                    "chunk_count": chunk_count,
                    "events_per_chunk": round(len(events) / chunk_count, 2) if chunk_count else 0,
                },
                duration=elapsed(),
            )
            self._log(TaskStage.EXTRACT, LogLevel.INFO, f"提取完成: {len(events)} 个事项")

        await self._emit(self._stage_callback, "extract_done", self.result.extract_result)

    # ── Search 阶段 ───────────────────────────────────────────────────────────

    async def search_async(self, config: SearchBaseConfig) -> None:
        """执行搜索（异步）。"""
        if not config.query:
            self._log(TaskStage.SEARCH, LogLevel.WARNING, "未提供查询，跳过搜索阶段")
            return

        async with self._stage(TaskStage.SEARCH, TaskStatus.SEARCHING) as elapsed:
            search_config = await self._build_search_config(config)
            raw = await self._searcher.search(search_config)
            data_ids, data_full, count = self._parse_search_result(raw)

            self.result.search_result = StageResult(
                stage=TaskStage.SEARCH,
                status="success",
                data_ids=data_ids,
                data_full=data_full,
                stats={
                    "matched_count": count,
                    "clues": raw.get("clues", []),
                    "search_stats": raw.get("stats", {}),
                    "query_info": raw.get("query", {}),
                    "nodes": raw.get("nodes", {}),
                },
                duration=elapsed(),
            )
            self._log(TaskStage.SEARCH, LogLevel.INFO, f"搜索完成: {count} 个匹配结果")

        await self._emit(self._stage_callback, "search_done", self.result.search_result)

    async def _build_search_config(self, config: SearchBaseConfig) -> SearchConfig:
        """将 SearchBaseConfig 补全为可执行的 SearchConfig。"""
        if isinstance(config, SearchConfig):
            return config
        source_id = await self._ensure_source()
        return SearchConfig(
            source_config_id=source_id,
            article_id=None,
            strategy_config=config.strategy_config,
            **{k: v for k, v in config.model_dump().items() if k != "strategy_config"},
        )

    @staticmethod
    def _parse_search_result(raw: Dict) -> Tuple[List[str], List[Dict], int]:
        """
        从 SAGSearcher 原始返回中提取 (data_ids, data_full, count)。
        PARAGRAPH 模式返回 sections；EVENT 模式返回 events。
        """
        sections = raw.get("sections", [])
        if sections:
            data_full = [s if isinstance(s, dict) else {"content": s} for s in sections]
            return [str(i) for i in range(len(data_full))], data_full, len(data_full)

        events = raw.get("events", [])
        data_full = [{"id": e.id, "title": e.title, "summary": e.summary, "content": e.content}
                     for e in events]
        return [e.id for e in events], data_full, len(events)

    # ── 便捷属性 ──────────────────────────────────────────────────────────────

    @property
    def chunk_ids(self) -> Optional[List[str]]:
        return self._load_result.chunk_ids if self._load_result else None

    @property
    def source_id(self) -> Optional[str]:
        return self._load_result.source_id if self._load_result else None

    @property
    def chunk_count(self) -> int:
        return len(self._load_result.chunk_ids) if self._load_result and self._load_result.chunk_ids else 0

    @property
    def event_count(self) -> int:
        r = self.result.extract_result
        return len(r.data_ids) if r and r.data_ids else 0

    def get_result(self) -> TaskResult:
        return self.result
