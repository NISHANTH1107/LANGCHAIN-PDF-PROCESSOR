import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import qrcode
from PIL import Image
import yt_dlp as youtube_dl
import io
import instaloader
import requests

# Configure the API with the provided key
try:
    genai.configure(api_key=st.secrets["general"]["API_KEY"])
except Exception as e:
    st.error("Failed to configure Google AI API. Please check your API key.")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None return from extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def load_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return new_db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed manner as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided
    context, then just say, "answer is not available in the context." Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def user_input(user_question):
    try:
        new_db = load_vector_store()
        if new_db is None:
            return "Please process a PDF file first"

        chain = get_conversational_chain()
        if chain is None:
            return "Error initializing chatbot"

        docs = new_db.similarity_search(user_question)
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        return "Sorry, I encountered an error processing your request"

# PDF Chatbot Section
st.title("PDF Chatbot")
st.write("Upload your PDF file and ask questions about its content.")

pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getbuffer())
        tmp_path = tmp.name

    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                raw_text = get_pdf_text([tmp_path])
                if not raw_text.strip():
                    st.error("No text could be extracted from the PDF")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    if get_vector_store(text_chunks):
                        st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                os.unlink(tmp_path)

    user_question = st.text_input("Ask a Question about the PDF:")
    if user_question:
        answer = user_input(user_question)
        st.write("Reply:", answer)

# [Rest of your existing code for YouTube downloader, QR code generator, and Instagram downloader...]
# Keep all those functions exactly as they were

# YouTube Video Downloader using yt-dlp
def download_video_with_yt_dlp(url):
    # Define output template for downloaded files
    ydl_opts = {
        'format': 'best',
        'outtmpl': '%(title)s.%(ext)s',  # Ensures the filename is based on the title
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get the exact file path from 'info'
            file_path = ydl.prepare_filename(info)  # Get the filename generated by yt-dlp
            
            # Verify if the file exists
            if os.path.exists(file_path):
                with open(file_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                filename = os.path.basename(file_path)  # Extract filename from the path
                return filename, video_bytes
            else:
                return None, f"Video file '{file_path}' not found in the expected location."

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

# Function to download Instagram posts/reels
def download_public_instagram_post(url):
    loader = instaloader.Instaloader()

    try:
        # Extract the shortcode from the URL
        post_shortcode = url.split("/")[-2]
        
        # Fetch the post object
        post = instaloader.Post.from_shortcode(loader.context, post_shortcode)

        # Check if the post contains a video (reel or regular video)
        if post.is_video:
            # Download the video content
            video_url = post.video_url
            response = requests.get(video_url, stream=True)

            if response.status_code == 200:
                filename = f"{post_shortcode}.mp4"
                file_bytes = response.content
                return filename, file_bytes
            else:
                return None, f"Failed to download the video from URL: {video_url}"
        else:
            # Download the image content
            image_url = post.url
            response = requests.get(image_url, stream=True)

            if response.status_code == 200:
                filename = f"{post_shortcode}.jpg"
                file_bytes = response.content
                return filename, file_bytes
            else:
                return None, f"Failed to download the image from URL: {image_url}"

    except Exception as e:
        return None, f"Error downloading Instagram post: {e}"
    
# Streamlit app section for Instagram Downloader
st.header("Instagram Public Post Downloader")
insta_url = st.text_input("Enter the public Instagram post URL:")

if insta_url:
    if st.button("Download Instagram Post"):
        filename, result = download_public_instagram_post(insta_url)

        if filename:
            media_type = "Video" if filename.endswith(".mp4") else "Image"
            st.success(f"{media_type} '{filename}' is ready for download!")
            st.download_button(
                label=f"Download {media_type}",
                data=result,
                file_name=filename,
                mime="video/mp4" if filename.endswith(".mp4") else "image/jpeg",
            )
        else:
            st.error(f"An error occurred: {result}")