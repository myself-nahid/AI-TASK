import gradio as gr
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
from dotenv import load_dotenv

load_dotenv()

# Load API Keys
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize LLM and embeddings
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")
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
    loader = PyPDFDirectoryLoader("research_papers")  # Data Ingestion step
    docs = loader.load()  # Document Loading
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

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

# Define Gradio Functions
# Level 1: Movie Recommendations
def recommend_movies(genre):
    genre = genre.strip().capitalize()
    if genre in knowledge_base:
        return f"Here are some {genre} movies: {', '.join(knowledge_base[genre])}"
    else:
        return "Sorry, I couldn't find any recommendations for that genre. Please try another!"

# Level 2: PDF Q&A
def pdf_qa(query):
    vectors = create_vector_embedding()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': query})
    answer = response['answer']
    contexts = [doc.page_content for doc in response['context']]
    return answer, contexts

# Level 3: Web Search
def search_web(query):
    return web_search(query)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Conversational AI Assistant with Multi-Modal Knowledge Integration")

    with gr.Tab("Movie Recommendations"):
        genre_input = gr.Textbox(label="Enter a movie genre (e.g., Action, Romance, Comedy)")
        movie_output = gr.Textbox(label="Movie Recommendations")
        gr.Button("Get Recommendations").click(recommend_movies, inputs=genre_input, outputs=movie_output)

    with gr.Tab("PDF Q&A"):
        pdf_query = gr.Textbox(label="Enter your query for the research papers")
        pdf_answer = gr.Textbox(label="Answer")
        pdf_contexts = gr.Textbox(label="Relevant Contexts", lines=5)
        gr.Button("Get Answer").click(pdf_qa, inputs=pdf_query, outputs=[pdf_answer, pdf_contexts])

    with gr.Tab("Web Search"):
        search_query = gr.Textbox(label="Enter your search query")
        search_results = gr.Textbox(label="Search Results", lines=5)
        gr.Button("Search").click(search_web, inputs=search_query, outputs=search_results)

# Fetch the dynamically assigned port from Render
port = int(os.environ.get("PORT", 7860))  # Default to 7860 if PORT isn't set

# Launch the Gradio App
demo.launch(server_name="0.0.0.0", server_port=port)
