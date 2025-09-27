# rag_builder.py
import streamlit as st
import os
import logging
from pathlib import Path
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetrieverBuilder:
    """
    A class to build and manage RAG retrievers with proper error handling and configuration.
    """
    
    def __init__(
        self, 
        data_dir: str = './data',
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3
    ):
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
    def _validate_data_directory(self) -> bool:
        """Validate that the data directory exists and contains files."""
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist")
            return False
        
        # Check for markdown files
        md_files = list(self.data_dir.rglob("*.md"))
        if not md_files:
            logger.warning(f"No .md files found in {self.data_dir}")
            return False
            
        logger.info(f"Found {len(md_files)} markdown files in {self.data_dir}")
        return True
    
    def _load_documents(self):
        """Load documents from the data directory."""
        try:
            loader = DirectoryLoader(
                str(self.data_dir),
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True,
                silent_errors=True  # Continue loading even if some files fail
            )
            docs = loader.load()
            
            if not docs:
                raise ValueError("No documents were loaded")
                
            logger.info(f"Loaded {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def _split_documents(self, docs):
        """Split documents into chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]  # Better splitting for markdown
            )
            splits = text_splitter.split_documents(docs)
            
            if not splits:
                raise ValueError("No text chunks were created")
                
            logger.info(f"Created {len(splits)} text chunks")
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def _create_embeddings(self):
        """Create embeddings model with error handling."""
        try:
            logger.info(f"Initializing embedding model: {self.embedding_model}")
            
            # Configure model kwargs for better performance
            model_kwargs = {
                'device': 'cpu',  # Use CPU for compatibility
                'trust_remote_code': False
            }
            
            encode_kwargs = {
                'normalize_embeddings': True  # Normalize embeddings for better similarity search
            }
            
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            logger.info("Embedding model loaded successfully")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def _create_vectorstore(self, splits, embeddings):
        """Create vector store from document splits."""
        try:
            logger.info("Creating vector store...")
            
            # Use a persistent directory for the vector store
            persist_directory = "./chroma_db"
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            
            logger.info("Vector store created successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def build_retriever(self):
        """
        Main method to build the retriever with comprehensive error handling.
        """
        try:
            logger.info("Starting RAG retriever build process")
            
            # Validate data directory
            if not self._validate_data_directory():
                raise ValueError("Data directory validation failed")
            
            # Load documents
            docs = self._load_documents()
            
            # Split documents
            splits = self._split_documents(docs)
            
            # Create embeddings
            embeddings = self._create_embeddings()
            
            # Create vector store
            vectorstore = self._create_vectorstore(splits, embeddings)
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
            
            logger.info("RAG retriever built successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to build retriever: {e}")
            st.error(f"Failed to build retriever: {e}")
            return None

@st.cache_resource
def build_retriever(
    data_dir: str = './data',
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 3
) -> Optional[object]:
    """
    Cached function to build a vector store retriever from documents.
    
    Args:
        data_dir: Directory containing the documents
        embedding_model: HuggingFace embedding model name
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        top_k: Number of documents to retrieve
        
    Returns:
        Retriever object or None if build fails
    """
    builder = RAGRetrieverBuilder(
        data_dir=data_dir,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
    )
    
    return builder.build_retriever()

# Additional utility function for testing
def test_retriever(retriever, query: str = "What is tax deduction?"):
    """Test the retriever with a sample query."""
    if retriever is None:
        logger.error("Retriever is None, cannot test")
        return
        
    try:
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i+1}: {doc.page_content[:200]}...")
    except Exception as e:
        logger.error(f"Error testing retriever: {e}")

if __name__ == "__main__":
    # Example usage
    retriever = build_retriever()
    if retriever:
        test_retriever(retriever)