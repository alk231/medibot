from langchain.chat_models import ChatOpenAI  # Import for Groq's ChatOpenAI model
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Set up the Groq model via ChatOpenAI
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",  # Groq API base URL
    model="llama3-70b-8192",  # Specify the model from Groq
)  # Automatically fetch the API key from .env

# Define the custom prompt template for answering questions based on context
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Create the prompt template with the context and question as input variables
prompt = PromptTemplate(
    input_variables=["context", "question"], template=CUSTOM_PROMPT_TEMPLATE
)

# Create HuggingFace embeddings for document retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the persisted Chroma DB for storing vectorized documents
db = Chroma(persist_directory="db", embedding_function=embeddings)

# Create retriever from the vector store (Chroma)
retriever = db.as_retriever()

# Create the RetrievalQA chain by linking the LLM, retriever, and prompt
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Your Groq model
    chain_type="stuff",  # Or "map_reduce", "refine", etc., depending on your use case
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,  # Return source documents along with the response
)

# Invoke the chain with a user query
user_query = input("Write Query Here: ")
response = retrieval_chain.invoke({"query": user_query})

# Print the result and source documents
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
