import os
os.environ["TRANSFORMERS_USE_LEGACY_IMPORT"] = "True"

import streamlit as st
from simpletransformers.t5 import T5Model
import torch
import warnings

# Suppress the FutureWarning about prepare_seq2seq_batch
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Translator",
    page_icon="🤖",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    # --- IMPORTANT: This path points to your FULLY TRAINED model ---
    # model_path = "outputs/mt5-sinhalese-english-100k" 
    model_path = "VivekNeer/mt5-sinhalese-english"
    
    print(f"Loading model from: {model_path}")
    model = T5Model("mt5", model_path)
    
    # Set better generation arguments for higher quality output
    model.args.num_beams = 5
    model.args.max_length = 50
    
    return model

st.title("AI Language Translator")
st.markdown("This app uses a fine-tuned `mT5-small` model trained on 100,000 sentences.")

with st.spinner("Loading the trained model... This may take a moment."):
    model = load_model()

# --- User Interface ---
st.header("Enter Text to Translate")
input_text = st.text_area("Input text:", height=100, placeholder="hello my name is vivek")

if st.button("Translate", type="primary"):
    if input_text:
        with st.spinner("Translating..."):
            # Use the model's predict method for translation
            translated_text = model.predict([input_text])
        
        # --- DEBUGGING: Print the raw output to the terminal ---
        print(f"Model output: '{translated_text[0]}'")

        st.subheader("Translated Text:")
        # --- UI FEEDBACK: Check if the output is empty ---
        if translated_text and translated_text[0]:
            st.success(translated_text[0])
        else:
            st.error("The model produced an empty translation. It may need more training or a different input.")
            
    else:
        st.warning("Please enter some text to translate.")