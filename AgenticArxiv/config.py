# AgenticArxiv/config.py
from dataclasses import dataclass
import os

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
    antigravity_base_url: str = os.getenv(
        "LLM_BASE_URL", "https://antigravity.byssted.cn"
    )
    antigravity_api_key: str = os.getenv("LLM_API_KEY", "no-token-here")
    models: LLMModels = LLMModels()

    # --- PDF download/cache ---
    pdf_raw_path: str = os.getenv(
        "PDF_RAW_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "pdf_raw")
    )
    pdf_cache_path: str = os.getenv(
        "PDF_CACHE_PATH", os.path.join(DEFAULT_OUTPUT_DIR, "pdf_cache.json")
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
    mysql_uri: str = os.getenv(
        "MYSQL_URI", "mysql+pymysql://root:root@127.0.0.1:3306/agentic_arxiv"
    )


settings = Settings()
