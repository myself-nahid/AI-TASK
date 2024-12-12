import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Load API Keys
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize LLM and embeddings
llm = ChatGroq(groq_api_key=os.getenv(
    "GROQ_API_KEY"), model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define Prompt Template for Document Q&A
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Load Movie Knowledge Base for Level 1


def load_knowledge_base():
    return {
        "Action": ["The Last Warrior", "Speed Breakers"],
        "Romance": ["Love in the Air", "A Heart's Whisper"],
        "Comedy": ["Laugh Factory", "The Great Escape"],
        "Sci-Fi": ["Beyond the Stars", "Time Travelers"]
    }


knowledge_base = load_knowledge_base()

# Create Vector Embeddings for Level 2


def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader(
            "research_papers")  # Data Ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings)

# Perform Web Search for Level 3


def web_search(query):
    # Using a mock search API, replace with actual implementation if needed
    api_url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("Abstract", "No relevant information found.")
    else:
        return "Error fetching web search results."


# Streamlit App Layout
st.title("Unified Chatbot: Movie Recommendations, PDF Q&A, and Web Search")

# Tabbed Interface
tab1, tab2, tab3 = st.tabs(["Movie Recommendations", "PDF Q&A", "Web Search"])

# Tab 1: Level 1 - Movie Recommendations
with tab1:
    st.header("Movie Recommendations Chatbot")
    user_query = st.text_input(
        "Enter a movie genre (e.g., Action, Romance, Comedy):")

    if st.button("Get Recommendations"):
        genre = user_query.strip().capitalize()
        if genre in knowledge_base:
            movies = ", ".join(knowledge_base[genre])
            st.success(f"Here are some {genre} movies: {movies}")
        else:
            st.error(
                "Sorry, I couldn't find any recommendations for that genre. Please try another!")

# Tab 2: Level 2 - PDF Q&A
with tab2:
    st.header("PDF Q&A with Retrieval Augmented Generation")
    user_prompt = st.text_input("Enter your query for the research papers:")

    if st.button("Prepare Document Embedding"):
        create_vector_embedding()
        st.success("Vector Database is ready.")

    if user_prompt:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start

        st.write(f"Response: {response['answer']}")
        st.write(f"Response Time: {elapsed_time:.2f} seconds")

        # Display retrieved context
        with st.expander("Document Similarity Search Results"):
            for i, doc in enumerate(response['context']):
                st.write(f"Document {i + 1}:")
                st.write(doc.page_content)
                st.write('------------------------')

# Tab 3: Level 3 - Web Search
with tab3:
    st.header("Web Search with LLM Integration")
    search_query = st.text_input("Enter your search query:")

    if st.button("Search the Web"):
        search_results = web_search(search_query)
        st.write(f"Search Results: {search_results}")
