import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Qdrant collection
        self.create_qdrant_collection()

    def create_qdrant_collection(self):
        client = QdrantClient(url=self.qdrant_url)

        try:
            # Check if the collection exists
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "size": 384,  # The size of the embedding vector
                        "distance": Distance.COSINE
                    }
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to check or create Qdrant collection: {e}")
            raise ConnectionError(f"Failed to check or create Qdrant collection: {e}")

    def create_embeddings(self, file_path: str):
        """
        Creates embeddings from a PDF or DOCX file and stores them in Qdrant.
        
        Args:
            file_path (str): Path to the document (PDF or DOCX).
        
        Returns:
            str: Success or failure message.
        """
        if not os.path.exists(file_path):
            logger.error(f"The file {file_path} does not exist.")
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        try:
            # Determine the file type and load accordingly
            file_extension = file_path.split('.')[-1].lower()
            if file_extension == "pdf":
                loader = UnstructuredPDFLoader(file_path)
            elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")
            
            # Load documents
            docs = loader.load()
            if not docs:
                logger.error("No documents were loaded.")
                raise ValueError("No documents were loaded from the file.")
            
            # Split the document text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=250
            )
            splits = text_splitter.split_documents(docs)
            if not splits:
                logger.error("No text chunks were created from the document.")
                raise ValueError("No text chunks were created from the documents.")
            
            # Create embeddings and store them in Qdrant
            Qdrant.from_documents(
                documents=splits,
                embedding=self.embeddings,
                url=self.qdrant_url,
                collection_name=self.collection_name,
                prefer_grpc=False
            )

            logger.info(f"Embeddings successfully created and stored in Qdrant for file: {file_path}")
            return "✅ Vector DB Successfully Created and Stored in Qdrant!"
        
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise Exception(f"Failed to create embeddings: {e}")
