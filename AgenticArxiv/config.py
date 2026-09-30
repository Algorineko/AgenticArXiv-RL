# AgenticArxiv/config.py
from dataclasses import dataclass
import os
from typing import Optional

try:
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
    load_dotenv()
except Exception:
    pass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


@dataclass(frozen=True)
class LLMModels:
    agent_model: str = os.getenv(
        "MODEL", "gemini-3-pro-preview"
    )
    translate_model: str = "tab_flash_lite_preview"


@dataclass(frozen=True)
class Settings:
    # LLM 端点是显式配置项：未设置 LLM_BASE_URL 时保持为空，
    # 由使用方（utils/llm_client.get_env_llm_client）给出明确报错，而不是拼出残缺 URL。
    antigravity_base_url: str = os.getenv("LLM_BASE_URL", "")
    antigravity_api_key: str = os.getenv("LLM_API_KEY", "no-token-here")
    models: LLMModels = LLMModels()

    # --- PDF download/cache ---
    pdf_raw_path: str = os.getenv(
        "PDF_RAW_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "pdf_raw")
    )
    pdf_cache_path: str = os.getenv(
        "PDF_CACHE_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "pdf_cache.json")
    )

    # --- Extracted paper figures (README T4) ---
    figures_path: str = os.getenv(
        "PDF_FIGURES_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "pdf_figures")
    )

    # --- PDF translate/cache ---
    pdf_translated_path: str = os.getenv(
        "PDF_TRANSLATED_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "pdf_translated")
    )
    pdf_translated_log_path: str = os.getenv(
        "PDF_TRANSLATED_LOG_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "pdf_translated_log")
    )
    translate_cache_path: str = os.getenv(
        "TRANSLATE_CACHE_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "translate_cache.json")
    )

    # --- pdf2zh CLI ---
    pdf2zh_bin: str = os.getenv("PDF2ZH_BIN", "pdf2zh")
    pdf2zh_service: str = os.getenv("PDF2ZH_SERVICE", "bing")
    pdf2zh_threads: int = int(os.getenv("PDF2ZH_THREADS", "4"))

    # --- MySQL ---
    # MySQL 属已归档 Web 栈的可选配置：未设置 MYSQL_URI 时保持为空，
    # 取值方（models/db.py）在使用时会收到 mysql_uri 的明确报错；
    # RL 训练路径不依赖 MySQL，不受影响。
    _mysql_uri: Optional[str] = os.getenv("MYSQL_URI")

    @property
    def mysql_uri(self) -> Optional[str]:
        """显式配置的 MySQL 连接串；未配置时报错提示如何设置。"""
        if not self._mysql_uri:
            raise RuntimeError(
                "未配置 MySQL 连接：请设置环境变量 MYSQL_URI"
                "（例：mysql+pymysql://user:pass@127.0.0.1:3306/agentic_arxiv）。"
                "RL 训练路径不依赖 MySQL，可不配置。"
            )
        return self._mysql_uri


settings = Settings()
