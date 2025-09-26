# rag_builder.py
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
# --- CHANGE 1: Import the new embeddings class ---
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def build_retriever():
    """
    Builds a vector store retriever from documents in the './data' directory.
    """
    print("--- Building Retriever with Local Embeddings ---")
    # 1. Load Documents
    loader = DirectoryLoader(
        './data',
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True
    )
    docs = loader.load()

    # 2. Split Texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # --- CHANGE 2: Use the local SentenceTransformerEmbeddings ---
    # This model runs on your machine. No API key needed.
    # The first time this runs, it will download the model (~227MB). This is a one-time event.
    print("--- Initializing local embedding model ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("--- Embedding model loaded ---")
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 4. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("--- Retriever Built Successfully ---")
    return retriever