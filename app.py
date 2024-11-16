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
API_KEY = st.secrets["general"]["API_KEY"]

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
    ydl_opts = {
        'format': 'best',
        'outtmpl': '%(title)s.%(ext)s',  # Save with the video title as the filename
    }

    try:
        # Download the video and return its filename and byte data
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)  # Download video immediately
            filename = f"{info['title']}.mp4"  # Ensure correct extension
            with open(filename, "rb") as video_file:
                video_bytes = video_file.read()  # Read video into memory
            return filename, video_bytes
    except Exception as e:
        return None, str(e)

# Streamlit app section for YouTube Downloader
st.header("YouTube Video Downloader")
video_url = st.text_input("Enter the YouTube video URL and press Enter:", key="youtube_url")

if video_url:  # Trigger processing when a URL is entered
    # Process the video immediately
    filename, result = download_video_with_yt_dlp(video_url)

    if filename:
        st.success(f"Video '{filename}' is ready for download!")

        # Display a single "Download Video" button for downloading the file
        st.download_button(
            label="Download Video",
            data=result,
            file_name=filename,
            mime="video/mp4",
            key="download_video_button"
        )
    else:
        st.error(f"An error occurred: {result}")


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

            # Initialize the filename in session state if not already set
            if "qr_filename" not in st.session_state:
                st.session_state.qr_filename = "qr_code"

            # Input field to specify filename
            qr_filename_input = st.text_input(
                "Enter a name for the QR Code (without extension):",
                value=st.session_state.qr_filename,  # Use the session state value
                key="qr_filename_input",
                on_change=lambda: setattr(
                    st.session_state, "qr_filename", st.session_state.qr_filename_input
                ),  # Update the session state when the input changes
            )

            # Use the session state filename with a .png extension
            download_filename = f"{st.session_state.qr_filename.strip() or 'qr_code'}.png"

            # Provide an option to download as PNG
            st.download_button(
                label="Download QR Code",
                data=qr_image,
                file_name=download_filename,
                mime="image/png",
                key="download_button"
            )

        except Exception as e:
            st.error(f"Error: {e}")