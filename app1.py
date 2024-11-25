# app.py
import streamlit as st
import torch
from gramformer import Gramformer
import time
import re

# Configure page
st.set_page_config(
    page_title="Grammar Checker",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the Gramformer model with caching"""
    try:
        use_gpu = torch.cuda.is_available()
        gf = Gramformer(models=1, use_gpu=use_gpu)
        return gf
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

def split_into_sentences(text):
    """Split text into sentences"""
    # Split on period followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def process_corrections(text, gf):
    """Process text and return corrections with retry mechanism"""
    sentences = split_into_sentences(text)
    corrected_sentences = []
    
    for sentence in sentences:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                corrections = list(gf.correct(sentence))
                if corrections:
                    corrected_sentences.append(corrections[0])
                    break
                time.sleep(0.5)
            except Exception as e:
                if attempt == max_retries - 1:
                    corrected_sentences.append(sentence)  # Keep original if correction fails
                time.sleep(0.5)
    
    # Join sentences back together
    corrected_text = ' '.join(corrected_sentences)
    return [corrected_text]  # Return in list format to maintain compatibility

def create_correction_card(edit):
    """Create a plain text card for each correction"""
    try:
        edit_type, orig_str, _, _, cor_str, _, _ = edit
    except ValueError:
        return ""
    
    explanation_map = {
        'SPELL': 'Spelling correction',
        'PUNCT': 'Punctuation fix',
        'MORPH': 'Word form correction',
        'VERB': 'Verb form correction',
        'DET': 'Article/Determiner correction',
        'PREP': 'Preposition correction',
        'OTHER': 'Grammar correction'
    }
    
    return f"""
    Error Type: {edit_type}
    Description: {explanation_map.get(edit_type, 'Grammar correction')}
    Original: {orig_str or '[NONE]'}
    Correction: {cor_str or '[NONE]'}
    ----------------------------------------
    """

def main():
    st.title("Grammar Checker")
    st.write("Enter your text below to check for grammar, spelling, and punctuation errors.")
    
    try:
        with st.spinner("Loading the model..."):
            gf = load_model()
            if gf is None:
                st.error("Failed to load the model. Please refresh the page.")
                return
        
        text_input = st.text_area("Enter your text:", height=150)
        max_chars = st.slider("Maximum characters per sentence", 100, 500, 200)
        
        if st.button("Check Grammar"):
            if not text_input:
                st.warning("Please enter some text to check.")
                return
            
            # Split long sentences if needed
            sentences = split_into_sentences(text_input)
            processed_sentences = []
            
            with st.spinner("Processing text..."):
                for sentence in sentences:
                    # Split long sentences further if needed
                    if len(sentence) > max_chars:
                        sub_sentences = [sentence[i:i+max_chars] 
                                      for i in range(0, len(sentence), max_chars)]
                    else:
                        sub_sentences = [sentence]
                    
                    for sub_sentence in sub_sentences:
                        corrections = process_corrections(sub_sentence, gf)
                        if corrections:
                            processed_sentences.append(corrections[0])
                        else:
                            processed_sentences.append(sub_sentence)
                
                final_text = ' '.join(processed_sentences)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Text")
                    st.text_area("", text_input, height=150, disabled=True)
                
                with col2:
                    st.subheader("Corrected Text")
                    st.text_area("", final_text, height=150, disabled=True)
                
                # Get and display edits
                try:
                    edits = gf.get_edits(text_input, final_text)
                    if edits:
                        st.subheader("Detailed Corrections")
                        correction_text = ""
                        for edit in edits:
                            correction_text += create_correction_card(edit)
                        st.text(correction_text)
                        
                except Exception as e:
                    st.error(f"Error processing edits: {str(e)}")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or check if all requirements are installed correctly.")

if __name__ == "__main__":
    main()