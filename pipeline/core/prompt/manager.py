"""
Prompt模板管理器

支持从YAML文件加载提示词模板，变量替换
支持多语言：默认中文(zh)，可通过 LLM_LANGUAGE 环境变量切换为英文(en)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from pipeline.exceptions import PromptError
from pipeline.utils import get_logger

logger = get_logger("prompt.manager")


class PromptTemplate:
    """提示词模板"""

    def __init__(
        self,
        name: str,
        template: str,
        variables: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.template = template
        self.variables = variables or []
        self.description = description

    def render(self, **kwargs: Any) -> str:
        """
        渲染模板

        Args:
            **kwargs: 变量值

        Returns:
            渲染后的文本

        Raises:
            PromptError: 缺少必需变量
        """
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise PromptError(f"模板'{self.name}'缺少必需变量: {', '.join(missing)}")

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise PromptError(f"模板变量错误: {e}") from e
        except Exception as e:
            raise PromptError(f"模板渲染失败: {e}") from e


class PromptManager:
    """提示词管理器（支持多语言）"""

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        """
        初始化提示词管理器

        Args:
            prompts_dir: 提示词目录路径

        多语言支持：
            - 默认中文(zh)，提示词在 prompts/ 根目录
            - 英文(en)优先从 prompts/en/ 加载，不存在则 fallback 到根目录
            - 通过环境变量 LLM_LANGUAGE 或 Settings.llm_language 配置
        """
        if prompts_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            prompts_dir = project_root / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_data: Dict[str, Dict[str, Any]] = {}

        self.language = self._get_language()

        if self.prompts_dir.exists():
            self.load_templates()
            logger.info(
                "提示词管理器初始化完成",
                extra={
                    "prompts_dir": str(self.prompts_dir),
                    "language": self.language,
                    "count": len(self.templates),
                },
            )
        else:
            logger.warning(f"提示词目录不存在: {self.prompts_dir}")

    def _get_language(self) -> str:
        """获取语言配置（优先级：Settings > 环境变量 > 默认值zh）"""
        try:
            from pipeline.core.config import get_settings
            return get_settings().llm_language
        except Exception:
            lang = os.getenv("LLM_LANGUAGE", "zh").lower()
            if lang not in ("zh", "en"):
                logger.warning(f"不支持的语言 '{lang}'，使用默认 'zh'")
                return "zh"
            return lang

    def load_templates(self) -> None:
        """
        从YAML文件加载所有模板（支持多语言 fallback）

        加载顺序：
        1. 语言目录（如 en/）- 优先
        2. 根目录 - fallback（跳过已加载的同名模板）
        """
        if not self.prompts_dir.exists():
            logger.warning(f"提示词目录不存在: {self.prompts_dir}")
            return

        # 1. 优先加载语言目录（非中文时）
        if self.language != "zh":
            lang_dir = self.prompts_dir / self.language
            if lang_dir.exists():
                for yaml_file in lang_dir.glob("*.yaml"):
                    try:
                        self._load_yaml_file(yaml_file)
                    except Exception as e:
                        logger.error(f"加载提示词文件失败 {yaml_file}: {e}", exc_info=True)

        # 2. 加载根目录（跳过已存在的模板）
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            try:
                self._load_yaml_file(yaml_file, skip_existing=True)
            except Exception as e:
                logger.error(f"加载提示词文件失败 {yaml_file}: {e}", exc_info=True)

    def _load_yaml_file(
        self, yaml_file: Path, skip_existing: bool = False, template_name: Optional[str] = None
    ) -> None:
        """从YAML文件加载模板"""
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            logger.warning(f"无效的YAML格式: {yaml_file}")
            return

        for name, config in data.items():
            if not isinstance(config, dict):
                continue

            final_template_name = template_name if template_name else name

            if skip_existing and final_template_name in self.templates:
                continue

            self.template_data[final_template_name] = config.copy()

            template_text = config.get("template", "") or config.get("system", "")
            variables = config.get("variables", [])
            description = config.get("description", "")

            template = PromptTemplate(
                name=final_template_name,
                template=template_text,
                variables=variables,
                description=description,
            )

            self.templates[final_template_name] = template
            logger.debug(f"加载模板: {final_template_name}")

    def get(self, name: str) -> PromptTemplate:
        """获取模板"""
        if name not in self.templates:
            raise PromptError(f"模板不存在: {name}")
        return self.templates[name]

    def render(self, name: str, **kwargs: Any) -> str:
        """渲染模板"""
        template = self.get(name)
        return template.render(**kwargs)

    def has(self, name: str) -> bool:
        """检查模板是否存在"""
        return name in self.templates

    def list_templates(self) -> List[str]:
        """列出所有模板名称"""
        return list(self.templates.keys())

    def get_template_config(self, name: str, *, test_mode: bool = False) -> Dict[str, Any]:
        """
        获取模板的完整配置数据（从 YAML 文件，支持测试模式）

        Args:
            name: 模板名称
            test_mode: 是否使用测试版本（读取 test_{name} 配置）

        Returns:
            完整的模板配置字典

        Raises:
            PromptError: 模板配置不存在
        """
        if test_mode:
            test_name = f"test_{name}"
            if test_name in self.template_data:
                logger.info(f"使用测试模板: {test_name}")
                name = test_name
            else:
                logger.warning(f"测试模板不存在: {test_name}，使用默认模板: {name}")

        if name not in self.template_data:
            self.load_templates()
            if name not in self.template_data:
                raise PromptError(f"模板配置不存在: {name}")

        return self.template_data[name]


# 全局管理器实例（单例）
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """获取全局提示词管理器"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def reset_prompt_manager() -> None:
    """重置全局提示词管理器"""
    global _prompt_manager
    _prompt_manager = None
