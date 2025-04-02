import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
import instaloader
import glob
import requests

# Load environment variables from .env file
load_dotenv()
API_KEY = st.secrets["general"]["API_KEY"]
# Configure the API with the provided key
os.environ["GOOGLE_API_KEY"] = API_KEY  # Set the API Key explicitly for the session

# Configure the API with the provided key
genai.configure(api_key=API_KEY)

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files."""
    text_dict = {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            # Get text and clean any internal PDF paths or metadata
            page_text = page.extract_text()
            # Remove any GID patterns (like those seen in the paste.txt example)
            page_text = re.sub(r'\[From: /gid\d+/.*?\]', '', page_text)
            pdf_text += page_text
        text_dict[pdf.name] = pdf_text
    return text_dict

def get_text_chunks(text_dict):
    """Split text into chunks for each PDF, maintaining source information."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_with_source = []
    
    for pdf_name, text in text_dict.items():
        chunks = text_splitter.split_text(text)
        # Add source metadata to each chunk
        for chunk in chunks:
            chunks_with_source.append({
                "text": chunk,
                "source": pdf_name
            })
    
    return chunks_with_source

def get_vector_store(chunks_with_source):
    """Create vector store from text chunks with source metadata."""
    embeddings = GoogleGenerativeAIEmbeddings(
        api_key=API_KEY, 
        model="models/embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    # Extract just the text for creating embeddings
    texts = [item["text"] for item in chunks_with_source]
    
    # Create metadata for each text chunk
    metadatas = [{"source": item["source"]} for item in chunks_with_source]
    
    # Process in smaller batches
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        st.write(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        if i == 0:  # First batch, create the index
            vector_store = FAISS.from_texts(
                texts=batch_texts, 
                embedding=embeddings, 
                metadatas=batch_metadatas
            )
        else:  # Subsequent batches, add to existing index
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas
            )
    
    vector_store.save_local("faiss_index")

def load_vector_store(embeddings):
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return new_db
    except FileNotFoundError:
        st.error("FAISS index not found. Please process PDF files first.")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question in a detailed manner based on the provided context. 
    If the answer is in multiple documents, clearly indicate which document each part of your answer comes from.
    Include document sources in your response using the format: [Source: document_name]
    If the answer is not in the provided context, just say "Answer is not available in the context."
    Don't provide incorrect information.
    
    DO NOT include any internal PDF paths like "/gid00019/..." in your answers.
    If you see such paths in the context, ignore them completely.

    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def clean_text(text):
    """Clean text by removing any GID patterns or other PDF artifacts"""
    # Remove patterns like [From: /gid00019/...]
    cleaned = re.sub(r'\[From: /gid\d+/.*?\]', '', text)
    # Remove other potential PDF artifacts if needed
    return cleaned

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=API_KEY, model="models/embedding-001")
    new_db = load_vector_store(embeddings)
    if new_db is None:  # If the index couldn't be loaded, exit early
        return "Please process PDF files first before asking questions."
    
    docs = new_db.similarity_search(user_question)
    
    # Debug information - show clean document sources
    with st.expander("Debug Info (Sources)"):
        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}: {doc.metadata['source']}")
            # Show a preview of the document content without GID patterns
            preview = clean_text(doc.page_content[:200]) + "..." if len(doc.page_content) > 200 else clean_text(doc.page_content)
            st.text(preview)
    
    # Clean document content before sending to LLM
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    
    # Final cleaning of response (just in case)
    clean_response = clean_text(response["output_text"])
    return clean_response

# Streamlit app starts here
st.title("Multi-PDF Chatbot")
st.write("Upload multiple PDF files and ask questions about their content.")

# Allow uploading multiple PDF files
pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if pdf_files:
    # Display the list of uploaded files
    st.write(f"Uploaded {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        st.write(f"- {pdf.name}")
    
    if st.button("Process PDFs"):
        try:
            # Save uploaded files to disk temporarily
            saved_files = []
            for pdf in pdf_files:
                pdf_path = pdf.name
                with open(pdf_path, "wb") as f:
                    f.write(pdf.getbuffer())
                saved_files.append(pdf_path)
            
            # Extract text from PDFs with cleaning
            text_dict = get_pdf_text(pdf_files)
            
            # Split text into chunks with source information
            chunks_with_source = get_text_chunks(text_dict)
            
            # Create vector store
            get_vector_store(chunks_with_source)
            
            st.success(f"Done processing {len(pdf_files)} PDF files!")
            
            # Store processed file names in session state for reference
            st.session_state['processed_pdfs'] = [pdf.name for pdf in pdf_files]
            
        except Exception as e:
            st.error(f"Error processing PDF files: {e}")

    # Display list of processed files if available
    if 'processed_pdfs' in st.session_state:
        with st.expander("Currently processed PDFs"):
            for pdf_name in st.session_state['processed_pdfs']:
                st.write(f"- {pdf_name}")

    # User question input with improved UI
    st.subheader("Ask Questions")
    user_question = st.text_input("Enter your question about the PDFs:")
    if user_question:
        with st.spinner("Searching for answer..."):
            answer = user_input(user_question)
        
        st.markdown("### Answer:")
        st.markdown(answer)
        
        # Option to ask a follow-up question
        st.markdown("---")
        st.markdown("**Ask another question or follow-up:**")


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
