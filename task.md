# RAG-based Document Chatbot

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) based Document Chatbot that enables users to interact with individual documents through natural language queries. The system retrieves relevant document sections and generates accurate, structured responses based on user input.

## Objective
Develop a chatbot that provides efficient semantic retrieval and generates structured outputs based on user queries while strictly adhering to the provided document content.

## Core Features
- RAG-based pipeline that effectively retrieves relevant document sections and generates accurate responses
- Context-bound responses that prevent hallucinations by answering only within the scope of the provided document
- User-friendly interface deployable on Streamlit or other suitable UI frameworks

## Technical Considerations
- Efficient retrieval mechanism to extract the most relevant content
- Structured response generation capability (tables, bullet points, formatted paragraphs)
- Flexibility in choice of embedding models, vector databases, and LLMs
- Low-latency processing with ability to handle reasonable document sizes

## Deliverables
1. Source code with proper documentation
2. Instructions for running the chatbot
3. Technical write-up including:
   - Tech stack choices and justifications
   - Response structuring approach
   - Challenges faced and solutions implemented

## Technical Implementation Suggestions
- Consider using Colab or Kaggle for GPU utilities
- Explore Hugging Face and Ollama libraries for open-source LLMs and embedding models
