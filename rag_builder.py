# rag_builder.py
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Use a Streamlit cache resource to prevent re-initializing the vector store on every interaction
@st.cache_resource
def build_retriever():
    """
    Builds a vector store retriever from documents in the './data' directory.
    """
    print("--- Building Retriever ---")
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

    # 3. Create Embeddings and Vector Store
    # We will use the free Google Generative AI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 4. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("--- Retriever Built Successfully ---")
    return retriever