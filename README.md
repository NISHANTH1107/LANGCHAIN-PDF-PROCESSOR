# PDF Chatbot

This project provides a PDF chatbot built with Streamlit, which enables users to upload a PDF file and ask questions about its content. The chatbot uses LangChain, FAISS for vector storage, and Google Generative AI embeddings to retrieve contextually relevant answers from the uploaded PDF.

## Features

- **PDF Parsing**: Extracts text from uploaded PDF files.
- **Text Chunking**: Splits extracted text into manageable chunks for efficient processing.
- **Vector Search**: Stores text embeddings for similarity search using FAISS.
- **Conversational Interface**: Allows users to query the PDF content and receive detailed answers.

## Requirements

This project requires the following packages:
- `streamlit`
- `PyPDF2`
- `langchain`
- `faiss-cpu`
- `langchain-google-genai`
- `python-dotenv`

##run 
pip install langchain_community

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
