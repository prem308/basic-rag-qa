import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:

    supported_extensions = [".pdf", ".txt", ".csv"]

    def __init__(
            self,
            chunk_size: int | None = None,
            chuk_overlap: int | None = None,
    ):
        setting = get_settings()
        self.chunk_size = chunk_size or setting.chunk_size
        self.chunk_overlap = chuk_overlap or setting.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

        logger.info(
            f"DocumentProcessor initiated with chunk_size={self.chunk_size}, /nchunk_overlap={self.chunk_overlap}"
        )

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """Load a PDF file
        
        Args:
            file_path = Path(file_path)

        Returns:
            List of document objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading PDF: {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages from PDF: {file_path.name}")
        return documents
    
    def load_text(self, file_path: str|Path) -> list[Document]:
        """Load a text file
        
        Args:
            file_path = Path(file_path)

        Returns:
            List of document objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading text file: {file_path.name}")

        loader = TextLoader(str(file_path))
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} documents from text file: {file_path.name}")
        return documents

    def load_csv(self, file_path: str|Path) -> list[Document]:
        """Load a CSV file
        
        Args:
            file_path = Path(file_path)

        Returns:
            List of document objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading CSV file: {file_path.name}")

        loader = CSVLoader(str(file_path))
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} documents from CSV file: {file_path.name}")
        return documents

    def load_file(self, file_path: str|Path) -> list[Document]:
        """Load a file based on its extension
        
        Args:
            file_path = Path(file_path)

        Returns:
            List of document objects
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.supported_extensions:
            logger.error(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}, supported types are: {self.supported_extensions}")
        
        loaders = {
            ".pdf": self.load_pdf,
            ".txt": self.load_text,
            ".csv": self.load_csv,
        }

        return loaders[ext](file_path)
    
    def load_from_upload(
            self,
            file: BinaryIO,
            filename: str,
    ) -> list[Document]:
        """Load a file from an uploaded binary stream(uploaded file via API)
        
        Args:
            file: Binary stream of the uploaded file
            filename: Original filename of the uploaded file
            
        Returns:
            List of document objects
        """
        ext = Path(filename).suffix.lower()

        if ext not in self.supported_extensions:
            logger.error(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}, supported types are: {self.supported_extensions}")
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext
        ) as temp_file:
            
            temp_file.write(file.read())
            temp_path = temp_file.name

            try:
                documents = self.load_file(temp_path)

                for doc in documents:
                    doc.metadata["source"] = filename

                return documents
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks using the text splitter
        
        Args:
            documents: List of document objects to split
            
        Returns:
            List of split document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks with chunk_size={self.chunk_size} and chunk_overlap={self.chunk_overlap}")
        
        chunks = self.text_splitter.split_documents(documents)

        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    
    def process_file(self, file_path:str|Path) -> list[Document]:
        """Process a file by loading and splitting it into chunks
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_file(file_path)
        return self.split_documents(documents)
    
    def process_upload(
            self,
            file: BinaryIO,
            filename: str,
    ) -> list[Document]:
        """Process an uploaded file by loading and splitting it into chunks
        
        Args:
            file: Binary stream of the uploaded file
            filename: Original filename of the uploaded file
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_from_upload(file, filename)
        return self.split_documents(documents)
            