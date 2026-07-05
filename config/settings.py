"""
전역 설정 관리 모듈
Pydantic BaseSettings로 환경변수를 로드하고 검증합니다.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str = "https://skax.ai-talentlab.com"
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
    OPENAI_MODEL: str = "gpt-5.4"
    EMBEDDING_MODEL: str = "text-embedding-3-large"

    # LangSmith (선택)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "power-market-ai"

    # 공공데이터포털
    PUBLIC_DATA_API_KEY: str

    # 경로
    PDF_PATH: str = "./data/pdf"

    # Qdrant 설정
    QDRANT_DB_PATH: str = "./qdrant_db"
    QDRANT_URL: str = ""          # 비어있으면 로컬, 값 있으면 클라우드 자동 전환
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "power_market_rules"
    VECTOR_SIZE: int = 3072       # text-embedding-3-large 차원

    # 앱 설정
    MAX_CONVERSATION_TURNS: int = 5
    RETRIEVAL_TOP_K: int = 5
    CONFIDENCE_THRESHOLD: float = 0.7

    def __getattr__(self, item: str):
        """Backward-compatibility: allow lowercase attribute access by mapping
        names like `qdrant_db_path` -> `QDRANT_DB_PATH` and
        `public_data_api_key` -> `PUBLIC_DATA_API_KEY`.
        """
        upper = item.upper()
        # upper가 실제 선언된 필드일 때만, 그리고 item 자체와 다를 때만 위임한다
        # (같으면 getattr가 다시 __getattr__을 호출해 무한 재귀에 빠짐)
        if upper != item and upper in type(self).model_fields:
            return getattr(self, upper)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")



@lru_cache()
def get_settings() -> Settings:
    return Settings()


def validate_settings() -> dict:
    """
    필수 설정값 검증
    
    Returns:
        dict: 검증 결과 {"valid": bool, "errors": list}
    """
    from pathlib import Path
    import os
    
    settings = get_settings()
    errors = []
    
    # 필수 API 키 검증
    if not settings.AZURE_OPENAI_API_KEY or settings.AZURE_OPENAI_API_KEY == "your_azure_openai_api_key":
        errors.append("AZURE_OPENAI_API_KEY not set in .env")
    
    if not settings.AZURE_OPENAI_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT == "https://your-azure-openai-endpoint":
        errors.append("AZURE_OPENAI_ENDPOINT not set in .env")
    
    if not settings.PUBLIC_DATA_API_KEY or settings.PUBLIC_DATA_API_KEY == "your_public_data_api_key":
        errors.append("PUBLIC_DATA_API_KEY not set in .env")
    
    # 경로 존재 여부 검증
    pdf_path = Path(settings.PDF_PATH)
    if not pdf_path.exists():
        try:
            pdf_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create PDF_PATH: {e}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }
