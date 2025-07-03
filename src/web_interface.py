import streamlit as st
import gc
import torch
import sys
from pathlib import Path
from typing import Optional
import PyPDF2
import docx

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from model import RAGModel
except ImportError:
    from .model import RAGModel

def extract_text_from_file(uploaded_file) -> Optional[str]:
    """Extract text from uploaded PDF or DOCX file"""
    try:
        if uploaded_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        return None

def generate_suggested_questions(text: str) -> list[str]:
    """Generate suggested questions from document text"""
    # Simple implementation - can be enhanced with NLP
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return [
        f"What is {lines[0].split('.')[0]}?" if lines else "What is this document about?",
        "What are the key points?",
        "Can you summarize this document?",
        "Who is the author?",
        "When was this created?"
    ][:5]  # Return max 5 questions

def main():
    st.set_page_config(layout="centered", page_title="DocAssist")
    
    # Initial state - only show upload
    if 'document_text' not in st.session_state:
        st.write("## Document Q&A Assistant")
        uploaded_file = st.file_uploader("Upload a PDF or DOCX file to begin", 
                                      type=['pdf', 'docx'])
        
        if uploaded_file:
            with st.spinner("Reading document..."):
                document_text = extract_text_from_file(uploaded_file)
                if document_text:
                    st.session_state.document_text = document_text
                    st.session_state.suggested_questions = generate_suggested_questions(document_text)
                    st.rerun()
                else:
                    st.error("Failed to read document text. Please try another file.")
        return
    
    # After upload - show Q&A interface
    st.write("## Document Q&A")
    st.success("Document uploaded successfully!")
    
    # Suggested questions from document
    st.sidebar.title("Suggested Questions")
    selected_question = None
    for q in st.session_state.suggested_questions:
        if st.sidebar.button(q):
            selected_question = q
    
    try:
        # Initialize with cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        model = RAGModel()
        
        # Use selected question if available
        question = st.text_input("Ask about the document:", value=selected_question or "")
        
        if question:
            with st.spinner("Analyzing document..."):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                answer = model.generate_answer(question, st.session_state.document_text)[0]
                
                if "I don't have enough information" in answer:
                    st.warning("This question doesn't seem related to the document. Please ask about the document content.")
                else:
                    st.text_area("Answer:", value=answer, height=200)
                
    except Exception as e:
        st.error(f"Error processing your request: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
