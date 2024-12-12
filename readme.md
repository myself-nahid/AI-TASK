# Conversational AI Assistant with Multi-Modal Knowledge Integration

This project demonstrates the development of a comprehensive conversational AI assistant that integrates three key functionalities:

1. **Movie Recommendations Chatbot**: Suggests movies based on user-provided genres.
2. **PDF-based Q&A with Retrieval Augmented Generation (RAG)**: Answers queries based on content extracted from PDF documents.
3. **Web Search with LLM Integration**: Fetches and combines web search results with LLM-generated responses.

## Features

- **Multi-Modal AI**: Combines traditional chatbot capabilities, document-based retrieval, and web search in a single interface.
- **Interactive UI**: Built with Gradio, providing an intuitive and user-friendly experience.
- **LLM-Driven Intelligence**: Powered by Groq's Llama3-8b-8192 and HuggingFace embeddings for accurate responses.
- **Customizable Knowledge Base**: Easy to extend and adapt for new domains or datasets.

## Technologies Used

- **Python**: Core programming language.
- **Gradio**: Web framework for interactive applications.
- **LangChain**: Framework for working with LLMs and building chains.
- **Groq LLM API**: For natural language processing tasks.
- **FAISS**: Vector database for efficient similarity search.
- **PyPDFLoader**: For ingesting and processing PDF documents.
- **DuckDuckGo API**: For fetching web search results.

## Getting Started

### Prerequisites

1. Python 3.8 or higher.
2. Required Python libraries:
   ```bash
   pip install gradio langchain_groq langchain_community langchain_huggingface
   ```
3. API Keys:
   - GROQ API Key
   - HuggingFace Token (HF_TOKEN)

### Project Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conversational-ai-assistant.git
   cd conversational-ai-assistant
   ```

2. Create a `.env` file in the root directory and add the following:
   ```env
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token
   ```

3. Prepare a folder named `research_papers` in the project directory and place your PDF documents there for Level 2 functionality.

### Running the Application

Run the Gradio app:
```bash
python app.py
```

### Usage

1. **Tab 1 - Movie Recommendations**:
   - Enter a movie genre (e.g., Action, Romance) to get tailored movie suggestions.

2. **Tab 2 - PDF Q&A**:
   - Embed documents into a vector database.
   - Query content from the research papers.

3. **Tab 3 - Web Search**:
   - Enter a query to fetch and display web search results alongside LLM-generated responses.

## Project Structure

```plaintext
.
├── app.py               # Main application script
├── research_papers/     # Folder for storing PDF documents
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## Future Enhancements

- Add more LLM models for broader support.
- Expand the knowledge base for movie recommendations.
- Integrate advanced web scraping for real-time data retrieval.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.readthedocs.io/)
- [Groq LLM](https://groq.com/)
- [Gradio](https://gradio.app/)