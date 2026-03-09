from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

@lru_cache
def get_embeddings()-> OpenAIEmbeddings:
    """Get cached OPENAI embeddings instance
    Returns:
        OpenAIEmbeddings: An instance of OpenAIEmbeddings
    """

    settings = get_settings()
    logger.info(f"Initializing OpenAIEmbeddings with model: {settings.embedding_model}")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    logger.info("OpenAIEmbeddings instance created and cached")
    return embeddings

class EmbeddingService:
    """Service for generating embeddings"""

    def __init__(self):
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model_name = settings.embedding_model

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query
        Args:
            text (str): The input text to embed
        Returns:
            list[float]: The generated embedding vector
        """
        logger.debug(f"Generating embedding for query: {text}")
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents
        Args:
            texts (list[str]): The list of input texts to embed
        Returns:
            list[list[float]]: A list of embedding vectors corresponding to the input texts
        """
        logger.debug(f"Generating embeddings for {len(texts)} documents")
        return self.embeddings.embed_documents(texts)