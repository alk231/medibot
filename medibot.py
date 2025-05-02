import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()


# Set up embedding model and vector store
@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3})


# Set up LLM (Groq)
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1", model="llama3-70b-8192"
    )


# Custom prompt template
def get_prompt():
    template = """
    Answer the user's question using only the information provided in the context.

- If the answer isn't in the context, respond with "I don't know."
- Do not make up information.
- Do not mention that you're using context or refer to any source.
- Keep the response clear, concise, and natural—like a helpful assistant.
- No greetings or filler text; just give the answer directly.

Context:
{context}

Question:
{question}

Answer:
    """
    return PromptTemplate(input_variables=["context", "question"], template=template)


# Main Streamlit app
def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            llm = load_llm()
            retriever = get_retriever()
            prompt = get_prompt()

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
            )

            response = qa.invoke({"query": user_input})
            answer = response["result"]
            sources = response["source_documents"]

            full_response = f"{answer}\n\n"
            st.chat_message("assistant").markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
