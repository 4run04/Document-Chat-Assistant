# Document Chat Assistant 📚

A Retrieval-Augmented Generation (RAG) based chatbot that enables intelligent conversations with documents using LangChain, Groq, and Google AI. The system uses MIME type detection for automatic file format handling and implements smart response formatting based on content type.


## Features

- 📄 Automatic file type detection and support for multiple formats (PDF, TXT, DOCX)
- 🔍 Semantic search using Google Embeddings with FAISS vector store
- 🤖 Context-aware responses using Llama 3.1 8B model via Groq
- 📊 Intelligent response formatting (tables, bullet points, paragraphs)
- 💻 Interactive Streamlit chat interface with message history

## Try It Out

🚀 **Live Demo**: [https://rag-chatbot-single.streamlit.app](https://rag-chatbot-single.streamlit.app)

## Technical Stack

- **LLM**: Llama 3.1 8B Instant model via Groq
- **Embeddings**: Google Generative AI Embeddings (models/embedding-001)
- **Vector Store**: FAISS with in-memory storage
- **Document Processing**: Magic MIME type detection
- **Framework**: LangChain with RAG implementation
- **UI**: Streamlit with chat message history

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Upload a document (PDF, TXT, or DOCX)
3. Ask questions about your document
4. Receive context-aware, well-formatted responses

## Response Formatting

The chatbot automatically formats responses based on content type:
- Numerical/tabular data → Markdown tables
- Steps/lists → Bullet points
- General information → Well-structured paragraphs

## Development Notes

- Uses RecursiveCharacterTextSplitter with 1000 chunk size and 200 overlap
- Implements semantic search with top-3 document retrieval
- Ensures responses are grounded in document context with clear formatting rules
- Automatically detects and handles file formats using MIME type detection
- Maintains chat history and vector store in Streamlit session state
