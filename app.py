import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings ,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import qrcode
from PIL import Image
import re
import platform
import yt_dlp as youtube_dl
import io 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables from .env file
load_dotenv()
API_KEY = st.secrets["API_KEY"]

# Configure the API with the provided key
os.environ["GOOGLE_API_KEY"] = API_KEY  # Set the API Key explicitly for the session
genai.configure(api_key=API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=API_KEY, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # This creates the index

def load_vector_store(embeddings):
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return new_db
    except FileNotFoundError:
        st.error("FAISS index not found. Please create it by processing PDF files first.")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed manner as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided
    context, then just say, "answer is not available in the context." Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=API_KEY, model="models/embedding-001")
    new_db = load_vector_store(embeddings)
    if new_db is None:  # If the index couldn't be loaded, exit early
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    return response["output_text"]

# Streamlit app starts here
st.title("PDF Chatbot")
st.write("Upload your PDF file and ask questions about its content.")

pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file:
    pdf_filename = pdf_file.name  # Use the uploaded file's original name
    with open(pdf_filename, "wb") as f:
        f.write(pdf_file.getbuffer())

    if st.button("Process PDF"):
        try:
            raw_text = get_pdf_text([pdf_filename])
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done processing the PDF!")

        except Exception as e:
            st.error(f"Error processing the PDF file: {e}")

    user_question = st.text_input("Ask a Question about the PDF:")
    if user_question:
        answer = user_input(user_question)
        st.write("Reply:", answer)

# YouTube Video Downloader using yt-dlp
def download_video_with_yt_dlp(url):
    download_folder = get_default_download_folder()
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),  # Save to the default download folder
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:  # Use youtube_dl (yt-dlp) here
            result = ydl.extract_info(url, download=True)
            return f"Video '{result['title']}' downloaded successfully to {download_folder}."
    except Exception as e:
        return f"Error: {e}"
    
# Get the user's default download folder based on the operating system
def get_default_download_folder():
    if platform.system() == "Windows":
        return os.path.join(os.environ["USERPROFILE"], "Downloads")  # For Windows
    else:
        return os.path.join(os.path.expanduser("~"), "Downloads")  # For macOS/Linux

# Streamlit app section for YouTube Downloader
st.header("YouTube Video Downloader")
video_url = st.text_input("Enter the YouTube video URL:")

if video_url:
    if st.button("Download Video"):
        try:
            message = download_video_with_yt_dlp(video_url)
            st.success(message)
        except Exception as e:
            st.error(f"An error occurred: {e}")


# Function to generate a QR code
def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,  # controls the size of the QR code
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,  # the size of each box in the QR code
        border=4,  # the thickness of the border
    )
    qr.add_data(data)  # Add the data to the QR code
    qr.make(fit=True)

    # Create an image from the QR code
    img = qr.make_image(fill='black', back_color='white')

    # Convert the image to a bytes object for displaying and downloading
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')  # Save the image as PNG
    img_byte_arr.seek(0)  # Rewind the byte stream to the beginning

    return img_byte_arr


# Streamlit app section for QR Code Generation
st.header("QR Code Generator")
qr_data = st.text_input("Enter the data for the QR Code:", key="qr_data_input")

if qr_data:
    if st.button("Generate QR Code"):
        try:
            # Generate the QR code
            qr_image = generate_qr_code(qr_data)

            # Display the generated QR code
            st.image(qr_image, caption="Generated QR Code")

            # Allow the user to specify a filename for the QR Code
            filename = st.text_input("Enter a name for the QR Code (without extension):", "qr_code", key="filename_input")

            # Provide an option to download as PNG
            st.download_button(
                label="Download QR Code",
                data=qr_image,
                file_name=f"{filename}.png",
                mime="image/png",
                key="download_button"
            )
            
        except Exception as e:
            st.error(f"Error: {e}")