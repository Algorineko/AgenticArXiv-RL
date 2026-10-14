"""Explicit operation configuration, workspace paths and packaged defaults.

Importing this module does not read a .env file or create file logs. Call
initialize_operation at an operation boundary, never while displaying help.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from importlib import resources
import os
from pathlib import Path


_workspace = None
_file_log_sink = None


def get_workspace() -> Path:
    return _workspace if _workspace is not None else Path.cwd()


def workspace_path(value: str | os.PathLike, *, workspace=None) -> Path:
    """Resolve a user path without searching source or installation directories."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(workspace).expanduser().resolve() if workspace is not None else get_workspace()
    return root / path


def resolve_model_source(source: str, *, local_only: bool = False, workspace=None) -> str:
    """Validate local paths without silently replacing them with a Hub model."""
    path = workspace_path(source, workspace=workspace)
    if path.is_dir():
        return str(path.resolve())
    is_hub_id = (
        len(source.split("/")) == 2
        and not source.startswith(("outputs/", "models/", "./", "../", "/", "~"))
    )
    if not local_only and is_hub_id:
        return source
    raise ValueError(f"指定模型路径不存在或不是模型目录: {path}")


def _default_output_path(filename: str) -> str:
    return str(workspace_path(Path("output") / filename))


@dataclass(frozen=True)
class LLMModels:
    agent_model: str = "gemini-3-pro-preview"
    translate_model: str = "tab_flash_lite_preview"


@dataclass(frozen=True)
class Settings:
    antigravity_base_url: str = ""
    antigravity_api_key: str = "no-token-here"
    models: LLMModels = field(default_factory=LLMModels)
    pdf_raw_path: str = field(default_factory=lambda: _default_output_path("pdf_raw"))
    pdf_cache_path: str = field(default_factory=lambda: _default_output_path("pdf_cache.json"))
    figures_path: str = field(default_factory=lambda: _default_output_path("pdf_figures"))
    pdf_translated_path: str = field(default_factory=lambda: _default_output_path("pdf_translated"))
    pdf_translated_log_path: str = field(
        default_factory=lambda: _default_output_path("pdf_translated_log")
    )
    translate_cache_path: str = field(
        default_factory=lambda: _default_output_path("translate_cache.json")
    )
    pdf2zh_bin: str = "pdf2zh"
    pdf2zh_service: str = "bing"
    pdf2zh_threads: int = 4

    @classmethod
    def from_environment(cls) -> Settings:
        """Read environment settings only at an explicit operation boundary."""
        path_defaults = {
            "pdf_raw_path": ("PDF_RAW_PATH", "pdf_raw"),
            "pdf_cache_path": ("PDF_CACHE_PATH", "pdf_cache.json"),
            "figures_path": ("PDF_FIGURES_PATH", "pdf_figures"),
            "pdf_translated_path": ("PDF_TRANSLATED_PATH", "pdf_translated"),
            "pdf_translated_log_path": ("PDF_TRANSLATED_LOG_PATH", "pdf_translated_log"),
            "translate_cache_path": ("TRANSLATE_CACHE_PATH", "translate_cache.json"),
        }
        paths = {
            name: str(workspace_path(os.getenv(variable, str(Path("output") / filename))))
            for name, (variable, filename) in path_defaults.items()
        }
        return cls(
            antigravity_base_url=os.getenv("LLM_BASE_URL", ""),
            antigravity_api_key=os.getenv("LLM_API_KEY", "no-token-here"),
            models=LLMModels(agent_model=os.getenv("MODEL", "gemini-3-pro-preview")),
            pdf2zh_bin=os.getenv("PDF2ZH_BIN", "pdf2zh"),
            pdf2zh_service=os.getenv("PDF2ZH_SERVICE", "bing"),
            pdf2zh_threads=int(os.getenv("PDF2ZH_THREADS", "4")),
            **paths,
        )


settings = Settings()


def initialize_operation(workspace=None, *, load_env: bool = True, logging: bool = False) -> Settings:
    """Select the workspace and optionally read only its .env, keeping env priority.

    Refresh the shared settings object in place so previously imported consumers
    observe the selected operation's paths and environment.
    """
    global _workspace
    _workspace = Path(workspace).expanduser().resolve() if workspace is not None else Path.cwd()
    if load_env:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=_workspace / ".env", override=False)
    configured = Settings.from_environment()
    for item in fields(Settings):
        object.__setattr__(settings, item.name, getattr(configured, item.name))
    if logging:
        setup_logger()
    return settings


def setup_logger():
    """Enable the legacy rotating file log only after an explicit operation."""
    global _file_log_sink
    from loguru import logger

    if _file_log_sink is not None:
        logger.remove(_file_log_sink)
    else:
        logger.remove()
    _file_log_sink = logger.add(
        workspace_path("output/log.txt"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
        mode="a",
    )
    return logger


def default_mainline_config_resource():
    """Return the installed default resource, not a guessed repository path."""
    return resources.files("agentic_arxiv").joinpath("resources").joinpath("mainline.json")


def read_default_mainline_config() -> bytes:
    return default_mainline_config_resource().read_bytes()
