# PDF Chatbot

This project provides a PDF chatbot built with Streamlit, which enables users to upload a PDF file and ask questions about its content. The chatbot uses LangChain, FAISS for vector storage, and Google Generative AI embeddings to retrieve contextually relevant answers from the uploaded PDF.

## Features

- **PDF Parsing**: Extracts text from uploaded PDF files.
- **Text Chunking**: Splits extracted text into manageable chunks for efficient processing.
- **Vector Search**: Stores text embeddings for similarity search using FAISS.
- **Conversational Interface**: Allows users to query the PDF content and receive detailed answers.
- **YouTube Video Downloader**: Downloads videos from YouTube using `yt-dlp`.
- **QR Code Generation**: Generates and allows the download of QR codes from user-provided data.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Setup API Key

To use Google Generative AI in the project, you need to set up an API key.

**1.** Create a .env file in the root directory of the project (if one doesn't exist already).
**2.** Obtain your API key from Google Cloud by following their documentation on creating API keys.
**3.** Add your API key to the .env file as shown below:

```bash
GOOGLE_API_KEY=your-api-key-here
```

## Requirements

This project requires the following packages:
- `streamlit`
- `PyPDF2`
- `langchain`
- `faiss-cpu`
- `langchain-google-genai`
- `python-dotenv`
- `yt-dlp`
- `qrcode`
- `Pillow`
- `reportlab`

## Run
```bash 
pip install langchain_community
```

### To Run the App 
```bash
streamlit run app.py
```
The application will launch locally, and you can interact with the PDF chatbot, download YouTube videos, and generate QR codes.

### How It Works
**1.** PDF Upload: Upload a PDF file using the file uploader.
**2.** Processing PDF: Once the file is uploaded, click on the "Process PDF" button to extract and split the text from the PDF.
**3.** Ask Questions: After processing, you can input a question, and the chatbot will provide answers based on the content of the uploaded PDF.
**4.** YouTube Video Downloader: Enter a YouTube video URL and click "Download Video" to download the video to your system.
**5.** QR Code Generator: Provide any text or URL, and the app will generate a QR code for you to download.