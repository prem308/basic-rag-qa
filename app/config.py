"""Application configuration using Pydantic's BaseSettings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
       env_file=".env",
       env_file_encoding="utf-8",
       case_sensitive=False,
       extra="ignore",
   )
    
    # OpenAI API settings
    OPENAI_API_KEY: str

    #qdrant cloud configuration
    QDRANT_API_KEY: str
    qdrant_url: str

    #Collection settings
    collection_name: str = "rag_documents"

    #Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    #model configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    #retriever settings
    reteriever_k: int = 4

    #Logging
    log_level: str = "INFO"

    #RAGAS Evaluation setting
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None
    ragas_llm_temperature: float | None = None
    ragas_embedding_model: str | None = None

    #API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    #application Info
    app_name: str = "RAG Q&A System"
    app_version: str = "0.1.0"

    @property
    def openai_api_key(self) -> str:
        return self.OPENAI_API_KEY

    @property
    def qdrant_api_key(self) -> str:
        return self.QDRANT_API_KEY

    @property
    def retrieval_k(self) -> int:
        return self.reteriever_k


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings instance.

    Returns:
        Settings: An instance of the Settings class with loaded configuration.
    """
    return Settings()



