import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GrammarCorrector:
    def __init__(self):
        # Initialize with COEDIT-XL model
        self.model_name = "grammarly/coedit-xl"
        
        # Load with try-catch to handle potential connection issues
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise e
        
    def correct_text(self, text: str, instruction: str = "Fix grammar") -> str:
        try:
            # Prepare the input with instruction
            input_text = f"{instruction}: {text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            # Generate correction
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_beams=5,  # Use beam search for better quality
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=False,  # Disable sampling for more consistent results
                    early_stopping=True
                )
            
            # Decode and return the corrected text
            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected_text
        except Exception as e:
            st.error(f"Error during correction: {str(e)}")
            raise e

def main():
    st.set_page_config(
        page_title="Grammar Correction with COEDIT-XL",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Advanced Grammar Correction")
    st.markdown("Powered by Grammarly's COEDIT-XL model")
    
    # Initialize the corrector
    @st.cache_resource
    def load_corrector():
        with st.spinner("Loading COEDIT-XL model... (this may take a moment)"):
            return GrammarCorrector()
    
    try:
        corrector = load_corrector()
        
        # Create two columns for input and output
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Text")
            text_input = st.text_area(
                "Enter your text",
                height=300,
                placeholder="Type or paste your text here..."
            )
            
            # Add instruction selection
            instruction = st.selectbox(
                "Select correction type",
                [
                    "Fix grammar",
                    "Fix grammar and improve clarity",
                    "Fix grammar and make formal",
                    "Fix grammar and simplify"
                ]
            )
            
            if st.button("Correct Text", type="primary"):
                if text_input.strip():
                    with st.spinner("Processing..."):
                        try:
                            with col2:
                                st.subheader("Corrected Text")
                                corrected_text = corrector.correct_text(text_input, instruction)
                                st.text_area(
                                    "Result",
                                    value=corrected_text,
                                    height=300
                                )
                                
                                # Add metrics/comparison
                                st.markdown("---")
                                st.markdown("### Changes Summary")
                                original_words = len(text_input.split())
                                corrected_words = len(corrected_text.split())
                                
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Original Words", original_words)
                                with metrics_col2:
                                    st.metric("Corrected Words", corrected_words)
                                with metrics_col3:
                                    st.metric("Word Difference", corrected_words - original_words)
                                
                        except Exception as e:
                            st.error("An error occurred during processing. Please try again.")
                else:
                    st.warning("Please enter some text to correct.")
        
        # Add information about the model
        with st.expander("About COEDIT-XL"):
            st.markdown("""
            COEDIT-XL is an advanced text editing model developed by Grammarly. It features:
            - State-of-the-art grammar correction
            - Ability to handle complex editing instructions
            - Support for various types of text improvements
            - Better performance compared to smaller models
            
            Best practices:
            1. For best results, provide clear, complete sentences
            2. The model works best with English text
            3. While very accurate, always review suggested changes
            4. Longer texts may be processed in chunks for better results
            """)
            
    except Exception as e:
        st.error("Failed to initialize the model. Please refresh the page or try again later.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()