import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import List
import tempfile
import magic

# Streamlit Deployment
try:
    # Load environment variables
    load_dotenv()
except Exception:
    # If running in Streamlit Cloud, use secrets
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Configure API keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GROQ_API_KEY or not GOOGLE_API_KEY:
    st.error('Please set GROQ_API_KEY and GOOGLE_API_KEY in .env file')
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def get_file_loader(file_path: str):
    """Return appropriate loader based on file type."""
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    
    if 'pdf' in file_type:
        return PyPDFLoader(file_path)
    elif 'text' in file_type:
        return TextLoader(file_path)
    elif 'docx' in file_type or 'microsoft' in file_type:
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def process_document(file):
    """Process uploaded document and create vector store."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load document
        loader = get_file_loader(tmp_file_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

def format_response(content: str) -> str:
    """Format response based on content type."""
    # Check if content appears to be a list/steps
    if '\n1.' in content or '\n•' in content or '\n-' in content:
        return content  # Already formatted as bullet points
    
    # Check if content appears to be tabular
    if '|' in content and '\n' in content:
        return content  # Already formatted as table
    
    # Return as regular paragraph
    return content

def get_response(query: str, vectorstore) -> str:
    """Get response from LLM using RAG."""
    # Initialize LLM
    llm = ChatGroq(model="llama-3.1-8b-instant")
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt template
    template = """Answer the question based solely on the following context. If you cannot answer the question based on the context, say 'I cannot answer this question based on the provided document.'

Context:
{context}

Question: {question}

Format your response appropriately:
- If the information is numerical or tabular, present it as a table using markdown syntax
- If the information contains steps or lists, present it as bullet points
- Otherwise, present it as a well-structured paragraph

Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    # Get response
    response = chain.invoke(query)
    print(response)
    return format_response(response.content)

# Streamlit UI
st.title('📚 Document Chat Assistant')

# File upload
uploaded_file = st.file_uploader("Upload a document (PDF, TXT, DOCX)", type=['pdf', 'txt', 'docx'])

if uploaded_file:
    with st.spinner('Processing document...'):
        # Process document and update session state
        st.session_state.vectorstore = process_document(uploaded_file)
        st.success('Document processed successfully!')

# Chat interface
if st.session_state.vectorstore:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                response = get_response(query, st.session_state.vectorstore)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info('Please upload a document to start chatting.')
