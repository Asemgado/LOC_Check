import streamlit as st
from google import genai
from google.genai import types
import PyPDF2
from PIL import Image
import os

def configure_gemini_api(api_key: str) -> bool:
    """Configure Google Gemini API with provided key"""
    try:
        os.environ['GOOGLE_API_KEY'] = api_key
        st.session_state.gemini_client = genai.Client(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {str(e)}")
        return False

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {str(e)}")
        return ""

def upload_pdf_to_gemini(pdf_file):
    """Upload PDF to Gemini and return file reference"""
    try:
        if not st.session_state.gemini_client:
            return None
            
        client = st.session_state.gemini_client
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{pdf_file.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        # Upload to Gemini
        uploaded_file = client.files.upload(file=temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return uploaded_file
    except Exception as e:
        st.error(f"Failed to upload PDF to Gemini: {str(e)}")
        return None

def analyze_cable_with_pdf(image: Image.Image, uploaded_pdf_ref) -> str:
    """Analyze cable image using Gemini API with uploaded PDF knowledge base"""
    try:
        if not st.session_state.gemini_client:
            return "Error: Gemini client not initialized"
            
        client = st.session_state.gemini_client
        
        # Convert PIL Image to bytes
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Create image part using the new API
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg"
        )
        
        prompt = """
        You are a cable inspection expert. Analyze the uploaded cable/slack image against the approval criteria from the PDF knowledge base provided.

        Please examine the image and the PDF document to determine:
        1. Whether the cable meets the approval criteria specified in the PDF
        2. Provide specific reasons for your decision based on the criteria
        3. Highlight any issues or concerns found
        4. Give a clear APPROVED or NOT APPROVED verdict

        Format your response as:
        **VERDICT: [APPROVED/NOT APPROVED]**
        
        **ANALYSIS:**
        [Your detailed analysis here referencing specific criteria from the PDF]
        
        **ISSUES FOUND (if any):**
        [List any issues with reference to PDF criteria]
        
        **RECOMMENDATIONS:**
        [Any recommendations for improvement based on the standards]
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, uploaded_pdf_ref, image_part]
        )
        return response.text
    except Exception as e:
        return f"Error analyzing image with PDF: {str(e)}"

def analyze_cable_image(image: Image.Image, knowledge_base: str) -> str:
    """Analyze cable image using Gemini API"""
    try:
        if not st.session_state.gemini_client:
            return "Error: Gemini client not initialized"
            
        client = st.session_state.gemini_client
        
        # Convert PIL Image to bytes
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Create image part using the new API
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg"
        )
        
        prompt = f"""
        You are a cable inspection expert. Analyze the uploaded cable/slack image against the following approval criteria from the knowledge base:

        KNOWLEDGE BASE:
        {knowledge_base}

        Please examine the image and determine:
        1. Whether the cable meets the approval criteria
        2. Provide specific reasons for your decision
        3. Highlight any issues or concerns found
        4. Give a clear APPROVED or NOT APPROVED verdict

        Format your response as:
        **VERDICT: [APPROVED/NOT APPROVED]**
        
        **ANALYSIS:**
        [Your detailed analysis here]
        
        **ISSUES FOUND (if any):**
        [List any issues]
        
        **RECOMMENDATIONS:**
        [Any recommendations for improvement]
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="Cable Inspection App",
    page_icon="🔌",
    layout="wide"
)

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ''
if 'uploaded_pdf_file' not in st.session_state:
    st.session_state.uploaded_pdf_file = None
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None
if 'default_pdf_loaded' not in st.session_state:
    st.session_state.default_pdf_loaded = False

def load_default_pdf():
    """Load the default PDF when API key is configured"""
    if st.session_state.default_pdf_loaded or st.session_state.uploaded_pdf_file or st.session_state.knowledge_base:
        return
    
    default_pdf_path = "/home/asemgado/github/LOC_Check/Splicing-Testing-Labeling-Deck.pdf"
    if os.path.exists(default_pdf_path) and st.session_state.gemini_client:
        try:
            pdf_ref = st.session_state.gemini_client.files.upload(file=default_pdf_path)
            st.session_state.uploaded_pdf_file = pdf_ref
            st.session_state.default_pdf_loaded = True
        except Exception:
            # Fallback to text extraction if upload fails
            try:
                with open(default_pdf_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    if text:
                        st.session_state.knowledge_base = text
                        st.session_state.default_pdf_loaded = True
            except Exception:
                pass

def main():
    st.title("🔌 Cable Inspection App")
    st.markdown("Upload cable/slack images and PDF criteria to get AI-powered approval decisions")
    
    # Try to load default PDF if API key is already configured
    if st.session_state.gemini_client and not st.session_state.default_pdf_loaded:
        load_default_pdf()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input (mandatory)
        api_key = st.text_input(
            "Enter Google Gemini API Key (Required):",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your API key from Google AI Studio",
            placeholder="Enter your Gemini API key here..."
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key to continue")
        
        if api_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = api_key
            if api_key:
                if configure_gemini_api(api_key):
                    st.success("✅ API Key configured successfully")
                    # Load default PDF after API key is configured
                    load_default_pdf()
                    if st.session_state.default_pdf_loaded:
                        st.info("📄 Default knowledge base PDF loaded automatically")
                else:
                    st.error("❌ Failed to configure API Key")
        
        st.divider()
        
        # PDF Knowledge Base Upload
        st.header("Knowledge Base")
        
        # Choose upload method
        upload_method = st.radio(
            "Choose PDF processing method:",
            ["Extract text (fallback)", "Upload to Gemini (recommended)"],
            help="Upload to Gemini provides better analysis as AI can directly read the PDF"
        )
        
        uploaded_pdf = st.file_uploader(
            "Upload PDF with approval criteria",
            type=['pdf'],
            help="Upload a PDF containing the cable approval criteria"
        )
        
        # Option to use existing PDF in workspace
        if st.button("📄 Use PDF from workspace"):
            pdf_path = "/home/asemgado/github/LOC_Check/Splicing-Testing-Labeling-Deck.pdf"
            if os.path.exists(pdf_path):
                if upload_method == "Upload to Gemini (recommended)":
                    with st.spinner("Uploading workspace PDF to Gemini..."):
                        try:
                            client = st.session_state.gemini_client
                            if client:
                                pdf_ref = client.files.upload(file=pdf_path)
                                st.session_state.uploaded_pdf_file = pdf_ref
                                st.success("✅ Workspace PDF uploaded to Gemini successfully")
                                st.info(f"File: Splicing-Testing-Labeling-Deck.pdf")
                            else:
                                st.error("❌ Gemini client not initialized")
                        except Exception as e:
                            st.error(f"Failed to upload workspace PDF: {str(e)}")
                else:
                    with st.spinner("Processing workspace PDF..."):
                        try:
                            with open(pdf_path, "rb") as f:
                                pdf_reader = PyPDF2.PdfReader(f)
                                text = ""
                                for page in pdf_reader.pages:
                                    text += page.extract_text() + "\n"
                                if text:
                                    st.session_state.knowledge_base = text
                                    st.success("✅ Workspace PDF processed successfully")
                                    st.info(f"File: Splicing-Testing-Labeling-Deck.pdf")
                        except Exception as e:
                            st.error(f"Failed to process workspace PDF: {str(e)}")
            else:
                st.error("❌ No PDF found in workspace")
        
        if uploaded_pdf is not None:
            if upload_method == "Upload to Gemini (recommended)":
                with st.spinner("Uploading PDF to Gemini..."):
                    pdf_ref = upload_pdf_to_gemini(uploaded_pdf)
                    if pdf_ref:
                        st.session_state.uploaded_pdf_file = pdf_ref
                        st.success("✅ PDF uploaded to Gemini successfully")
                        st.info(f"File ID: {pdf_ref.name}")
                    else:
                        st.error("❌ Failed to upload PDF to Gemini")
            else:
                with st.spinner("Processing PDF..."):
                    knowledge_text = extract_text_from_pdf(uploaded_pdf)
                    if knowledge_text:
                        st.session_state.knowledge_base = knowledge_text
                        st.success(f"✅ PDF processed - {len(knowledge_text)} characters extracted")
                        with st.expander("Preview extracted text"):
                            st.text(knowledge_text[:500] + "..." if len(knowledge_text) > 500 else knowledge_text)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Cable/Slack Images")
        
        uploaded_images = st.file_uploader(
            "Choose cable or slack images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images of cables or slack that need inspection"
        )
        
        if uploaded_images:
            st.success(f"✅ {len(uploaded_images)} image(s) uploaded")
            
            # Display uploaded images
            for idx, uploaded_image in enumerate(uploaded_images):
                st.image(uploaded_image, caption=f"Image {idx+1}: {uploaded_image.name}", width=300)
    
    with col2:
        st.header("Analysis Results")
        
        # Check if we have knowledge base (either text or uploaded PDF)
        has_knowledge_base = st.session_state.knowledge_base or st.session_state.uploaded_pdf_file
        
        if st.button("🔍 Analyze Images", disabled=not (uploaded_images and st.session_state.gemini_api_key and has_knowledge_base)):
            if not st.session_state.gemini_api_key:
                st.warning("⚠️ Please enter your Gemini API key in the sidebar")
            elif not has_knowledge_base:
                st.warning("⚠️ Please upload a PDF with approval criteria")
            elif not uploaded_images:
                st.warning("⚠️ Please upload at least one image")
            else:
                for idx, uploaded_image in enumerate(uploaded_images):
                    with st.expander(f"Analysis for {uploaded_image.name}", expanded=True):
                        with st.spinner(f"Analyzing image {idx+1}..."):
                            image = Image.open(uploaded_image)
                            
                            # Use PDF upload method if available, otherwise fallback to text extraction
                            if st.session_state.uploaded_pdf_file:
                                analysis_result = analyze_cable_with_pdf(image, st.session_state.uploaded_pdf_file)
                            else:
                                analysis_result = analyze_cable_image(image, st.session_state.knowledge_base)
                            
                            st.markdown(analysis_result)
        
        # Status indicators
        st.divider()
        st.subheader("Status Check")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            if st.session_state.gemini_api_key:
                st.success("✅ API Key")
            else:
                st.error("❌ API Key")
        
        with status_col2:
            if st.session_state.knowledge_base or st.session_state.uploaded_pdf_file:
                if st.session_state.uploaded_pdf_file:
                    st.success("✅ PDF Uploaded")
                else:
                    st.success("✅ Text Extracted")
            else:
                st.error("❌ Knowledge Base")
        
        with status_col3:
            if uploaded_images:
                st.success(f"✅ {len(uploaded_images)} Images")
            else:
                st.error("❌ No Images")

if __name__ == "__main__":
    main()