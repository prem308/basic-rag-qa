from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

#Embedding dimention for text-embedding-3-small
EMBEDDING_DIMENSION = 1536

@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Get cached QdrantClient instance.

    Returns:
        Configured QdrantClient instance.
    
    """
    logger.info(f"Creating QdrantClient with URL: {settings.qdrant_url}")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    logger.info("QdrantClient created successfully.")
    return client

class VectorStoreService:
    """Service for managing vector store operations with Qdrant."""

    def __init__(self, collection_name: str | None = None):
        """Initialize the VectorStoreService.

        Args:
            collection_name: Optional name of the Qdrant collection to use. If not provided, a unique name will be generated.
        """
        self.collection_name = collection_name or settings.collection_name
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()

        #Ensure the collection exists when the service is initialized
        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        logger.info(f"VectorStoreService initialized with collection: {self.collection_name}")

    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists, creating it if necessary."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists: {collection_info} with {collection_info.points_count} points.")

        except UnexpectedResponse as e:
            logger.info(f"Collection '{self.collection_name}' does not exist. Creating new collection.")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of document to add
            
        Returns:
            List of document IDs that were added to the vector store.
        """
        if not documents:
            logger.warning("No documents provided to add to vector store.")
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store.")
        
        #create unique IDs for each document
        ids = [str(uuid4()) for _ in documents]

        self.vector_store.add_documents(documents, ids=ids)
        logger.info(f"Successfully added {len(documents)} documents to vector store with IDs: {ids}")
        return ids
        
    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """Search for similar documents in the vector store based on a query.

        Args:
            query: The search query string.
            top_k: Optional number of top results to return. If not provided, defaults to 4.

        Returns:
            List of Documents that are most similar to the query.
        """
        if not query:
            logger.warning("Empty query provided for search.")
            return []
        
        top_k = top_k or settings.reteriever_k
        logger.debug(f"Searching for top {top_k} documents similar to query: '{query}'")

        results = self.vector_store.similarity_search(query, k=top_k)
        logger.info(f"Search completed. Found {len(results)} similar documents.")
        return results
    
    def search_with_scores(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """Search for similar documents in the vector store based on a query, returning documents with similarity scores.

        Args:
            query: The search query string.
            top_k: Optional number of top results to return. If not provided, defaults to 4.

        Returns:
            List of tuples containing Documents and their corresponding similarity scores.
        """
        if not query:
            logger.warning("Empty query provided for search with scores.")
            return []
        
        top_k = top_k or settings.reteriever_k
        logger.debug(f"Searching for top {top_k} documents similar to query: '{query}' with scores")

        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        logger.info(f"Search with scores completed. Found {len(results)} similar documents.")
        return results
        
    def get_retriever(self, top_k: int | None = None) -> Any:
        """Get a retriever object for performing similarity searches.

        Args:
            top_k: number of docs to retrieve.
        Returns:
            Langchain retriever object configured to use the vector store.
        """

        top_k = top_k or settings.reteriever_k

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
    
    def delete_collection(self) -> None:
        """Delete the entire Qdrant collection associated with this vector store."""
        logger.warning(f"Deleting collection '{self.collection_name}' from Qdrant.")
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted successfully.")

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the Qdrant collection.

        Returns:
            A dictionary containing collection information such as name, vector count, and configuration.
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            status = getattr(collection_info, "status", None)
            return {
                "name": self.collection_name,
                "vectors_count": getattr(collection_info, "points_count", 0) or 0,
                "indexed_vectors_count": getattr(collection_info, "indexed_vectors_count", 0) or 0,
                "status": getattr(status, "value", status) or "unknown",
            }
        except UnexpectedResponse as e:
            return{
                "name": self.collection_name,
                "vectors_count": 0,
                "indexed_vectors_count": 0,
                "status": "not found",
            }
        
    def health_check(self) -> bool:
        """Perform a health check on the Qdrant client connection.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        try:
            self.client.get_collections()
            logger.info("Qdrant health check successful.")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False