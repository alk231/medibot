from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI  # Groq's Llama3 model
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # Import for embeddings
from dotenv import load_dotenv

DOC_PATH = "data/"
CHROMADB_PATH = "db"

load_dotenv()


# Load PDFs from directory
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


documents = load_pdf_files(DOC_PATH)
print(f"Total documents loaded: {len(documents)}")


# Split documents into chunks
def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)


chunks = create_chunks(documents)
print(f"Total chunks created: {len(chunks)}")

# Create embeddings using Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and persist Chroma DB with embeddings
db = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory=CHROMADB_PATH
)

# Now you can use the LLM for generating responses based on the embeddings
