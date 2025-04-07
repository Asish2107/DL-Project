import os
from glob import glob
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import time

# ======================
# Configuration
# ======================

#ollama serve  # This should be running in the terminal
#ollama pull mistral #Download the model

PDF_FOLDER = "/Users/kasish/Desktop/JGASVEMLKNPR-PROJECT/DL-Project/data"          # Folder containing PDF documents
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free embedding model
LLM_MODEL = "mistral"          # Free LLM (options: mistral, llama2, gemma:2b)
CHUNK_SIZE = 1000              # Text chunk size for processing
CHUNK_OVERLAP = 200            # Overlap between chunks

# ======================
# Custom Embeddings Class
# ======================
class LocalEmbeddings(Embeddings):
    """Free local embeddings using SentenceTransformers"""
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

# ======================
# Document Processing
# ======================
def load_and_process_documents():
    """Load and split PDF documents with error handling"""
    documents = []
    
    # Get all PDF files in the folder
    pdf_files = glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {PDF_FOLDER}")
    
    # Process each PDF
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_path)
            documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Error loading {pdf_path}: {str(e)}")
            continue
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# ======================
# Ollama Connection Setup
# ======================
def verify_ollama_connection(retries=3, delay=5):
    """Ensure Ollama is running before proceeding"""
    for attempt in range(retries):
        try:
            test_llm = Ollama(model=LLM_MODEL)
            test_llm("test")
            return True
        except Exception as e:
            if attempt < retries - 1:
                print(f"🔌 Connection failed, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Could not connect to Ollama. Please ensure:\n"
                    f"1. Ollama is installed (brew install ollama)\n"
                    f"2. Service is running (ollama serve)\n"
                    f"3. Model is downloaded (ollama pull {LLM_MODEL})"
                )

# ======================
# Main RAG System
# ======================
def initialize_rag_system():
    # Verify Ollama connection first
    verify_ollama_connection()
    
    # Load and process documents
    chunks = load_and_process_documents()
    
    # Create embeddings and vector store
    embeddings = LocalEmbeddings()
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )
    
    # Initialize LLM
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.3,#determines how random or deterministic the model has to be
        timeout=300  # Increase for large documents, wait time after which queries response should be timedout.
    )
    
    # Create QA chain
    # Create and return a RetrievalQA chain using a specified LLM and retriever.
    # This setup enables a question-answering system over a document collection.
    return RetrievalQA.from_chain_type(
        # The language model (LLM) to be used for generating answers.
        # This should be an instance of a model supported by LangChain (e.g., OpenAI's GPT).
        llm=llm,
        # The chain type determines how retrieved documents are processed.
        # "stuff" means all documents are stuffed into a single prompt and passed to the LLM.
        # Best for small numbers of short documents.
        chain_type="stuff",
        # This converts the vector store into a retriever object.
        # The retriever is responsible for finding the most relevant documents for a user's query.
        # `search_kwargs={"k": 3}` tells it to retrieve the top 3 most similar documents.
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        # This ensures that the output will include the documents that were retrieved.
        # Helpful for traceability, citations, or debugging purposes.
        return_source_documents=True
    )

# ======================
# Interactive Query Loop
# ======================
if __name__ == "__main__":
    # Initialize system
    print("🚀 Initializing RAG system...")
    qa_system = initialize_rag_system()
    print("✅ System ready!")
    
    # Query interface
    print("\nAsk questions about your documents (type 'exit' to quit)")
    while True:
        try:
            query = input("\n❓ Question: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            result = qa_system({"query": query})
            
            print("\n💡 Answer:")
            print(result["result"])
            
            print("\n📚 Sources:")
            sources = {doc.metadata["source"] for doc in result["source_documents"]}
            for source in sources:
                print(f"- {source}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")