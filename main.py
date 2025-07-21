import streamlit as st
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
import threading
import time

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize components
@st.cache_resource
def init_groq_client():
    """Initialize Groq client with error handling"""
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not found. Please check your .env file.")
        return None
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def init_tts_engine():
    """Initialize text-to-speech engine"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except Exception as e:
        st.warning(f"TTS engine initialization failed: {e}")
        return None

# Set Tesseract path (adjust based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize clients
client = init_groq_client()
engine = init_tts_engine()

def speak(text, use_tts=True):
    """Enhanced speak function with better formatting and TTS"""
    st.markdown(f"**🤖 Assistant:** {text}")
    
    if use_tts and engine:
        try:
            # Run TTS in a separate thread to avoid blocking
            def run_tts():
                engine.say(text)
                engine.runAndWait()
            
            tts_thread = threading.Thread(target=run_tts)
            tts_thread.daemon = True
            tts_thread.start()
        except Exception as e:
            st.warning(f"TTS error: {e}")

def chat_with_llm(prompt, context=None):
    """Enhanced LLM chat with context support"""
    if not client:
        return "❌ AI service not available. Please check your API configuration."
    
    try:
        messages = [{"role": "user", "content": prompt}]
        
        if context:
            messages.insert(0, {"role": "system", "content": f"Context: {context}"})
        
        with st.spinner("🤔 Thinking..."):
            response = client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error communicating with AI: {str(e)}"

def extract_text_from_image(image_file):
    """Enhanced OCR with preprocessing"""
    try:
        # Load and preprocess image
        image = Image.open(image_file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply preprocessing for better OCR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Noise removal and enhancement
        denoised = cv2.medianBlur(gray, 5)
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
        
        # OCR with custom config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(enhanced, config=custom_config)
        
        return text.strip() if text.strip() else "No text detected in the image."
    
    except Exception as e:
        return f"❌ Error extracting text: {str(e)}"

def listen_to_voice():
    """Enhanced voice recognition with better error handling"""
    recognizer = sr.Recognizer()
    
    # Adjust for ambient noise
    with sr.Microphone() as source:
        st.info("🎤 Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak clearly!")
            
            # Create a placeholder for real-time feedback
            status_placeholder = st.empty()
            
            # Listen with timeout
            audio = recognizer.listen(source, phrase_time_limit=10, timeout=15)
            status_placeholder.info("🔄 Processing speech...")
            
            # Recognize speech
            query = recognizer.recognize_google(audio, language='en-US')
            status_placeholder.success(f"✅ You said: **{query}**")
            return query
            
    except sr.WaitTimeoutError:
        return "⏰ Listening timeout. Please try again."
    except sr.UnknownValueError:
        return "🤷 Sorry, I couldn't understand that. Please speak more clearly."
    except sr.RequestError as e:
        return f"❌ Speech recognition service error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# Streamlit UI Configuration
st.set_page_config(
    page_title="AI Voice Assistant", 
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Header
st.title("AI Voice and Chat Assistant")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    use_tts = st.checkbox("Enable Text-to-Speech", value=True)
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("• 💬 Text & Voice Chat")
    st.markdown("• 📸 OCR Text Extraction")
    st.markdown("• 🗣️ Real-time Speech Recognition")

# Main interface
option = st.radio(
    "Choose an action:", 
    ["💬 Text Chat", "🎤 Voice Chat", "📸 Extract Text from Image"],
    key="action_choice"
)

# Text Chat Section
if option == "💬 Text Chat":
    st.subheader("💬 Text Chat")
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f"**👤 You:** {user_msg}")
        st.markdown(f"**🤖 Assistant:** {ai_msg}")
        st.markdown("---")
    
    # Input form
    with st.form("text_chat_form"):
        query = st.text_input("Ask me anything:", placeholder="Type your question here...")
        submitted = st.form_submit_button("Send 📤")
        
        if submitted and query.strip():
            with st.spinner("🤔 Generating response..."):
                answer = chat_with_llm(query)
                st.session_state.chat_history.append((query, answer))
                speak(answer, use_tts)
                st.rerun()

# Voice Chat Section
elif option == "🎤 Voice Chat":
    st.subheader("🎤 Voice Chat")
    st.markdown("Click the button below and speak your question:")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎤 Start Listening", type="primary"):
            query = listen_to_voice()
            
            if query and not query.startswith(("⏰", "🤷", "❌")):
                st.markdown(f"**👤 You said:** {query}")
                answer = chat_with_llm(query)
                speak(answer, use_tts)
            else:
                st.error(query)
    
    with col2:
        st.info("💡 **Tips:**\n- Speak clearly and slowly\n- Use a quiet environment\n- Wait for the listening indicator")

# OCR Section
elif option == "📸 Extract Text from Image":
    st.subheader("📸 Extract Text from Image")
    
    uploaded_file = st.file_uploader(
        "Upload an image containing text:", 
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file:
        # Display image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("📝 **OCR Tips:**\n- Use clear, high-contrast images\n- Ensure text is not rotated\n- Good lighting helps accuracy")
        
        # Extract text button
        if st.button("🔍 Extract Text", type="primary"):
            with st.spinner("🔍 Extracting text..."):
                extracted_text = extract_text_from_image(uploaded_file)
                
                st.subheader("📄 Extracted Text:")
                st.text_area("", value=extracted_text, height=200, disabled=True)
                
                # Option to ask questions about the extracted text
                if extracted_text and not extracted_text.startswith("❌"):
                    st.subheader("💬 Ask about the extracted text:")
                    question = st.text_input("What would you like to know about this text?")
                    
                    if st.button("Ask Question") and question:
                        context_prompt = f"Based on this extracted text: '{extracted_text}'\n\nQuestion: {question}"
                        answer = chat_with_llm(context_prompt)
                        speak(f"Based on the extracted text: {answer}", use_tts)

