
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Check Gemini API key
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "Gemini API key not found! Check your .env file."
    )

# ChromaDB storage location
CHROMA_PATH = str(BASE_DIR / "chroma_db")


# Initialize embedding model
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )


# STEP 1: Process the research paper
def process_pdf(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    print("Loading PDF...")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    print(f"Loaded {len(documents)} pages.")

    # STEP 2: Split text into chunks
    print("Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    print(f"Created {len(chunks)} chunks.")

    # STEP 3: Generate embeddings
    print("Generating embeddings...")

    embeddings = get_embeddings()

    # STEP 4: Store in ChromaDB
    print("Storing embeddings in ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="muse_research"
    )

    print("PDF successfully indexed!")

    return vectorstore


# STEP 5: Load existing vector database
def load_vectorstore():

    if not os.path.exists(CHROMA_PATH):
        raise ValueError(
            "No vector database found. Upload a PDF first."
        )

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="muse_research"
    )

    return vectorstore


# STEP 6: Retrieve relevant information
def retrieve_documents(query, k=4):

    print("Searching research paper...")

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(
        query=query,
        k=k
    )

    return results
