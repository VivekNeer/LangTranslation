import os
os.environ["TRANSFORMERS_USE_LEGACY_IMPORT"] = "True"

import streamlit as st
from simpletransformers.t5 import T5Model
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

st.set_page_config(page_title="AI Translator (Kannada)", page_icon="🤖", layout="centered")

@st.cache_resource
def load_model():
    # --- CHANGED: Point this to your newly trained local model directory ---
    model_path = "outputs/mt5-kannada-english-100k"
    
    print(f"Loading model from: {model_path}")
    # You MUST specify the model type ("mt5") when loading a local simpletransformers model
    model = T5Model("mt5", model_path)
    
    model.args.num_beams = 5
    model.args.max_length = 50
    
    return model

st.title("AI Language Translator (Kannada/English)")
st.markdown("This app uses a fine-tuned `mT5-small` model trained on 50,000 Kannada-English sentence pairs.")

with st.spinner("Loading the trained model... This may take a moment."):
    model = load_model()

st.header("Enter Text to Translate")
# --- Add a prefix so the model knows what to do ---
prefix = st.radio(
    "Select translation direction:",
    ("translate english to kannada", "translate kannada to english")
)
input_text = st.text_area("Input text:", height=100, placeholder="hello my name is vivek")

if st.button("Translate", type="primary"):
    if input_text:
        with st.spinner("Translating..."):
            # Combine the prefix and the input text
            prefixed_input = f"{prefix}: {input_text}"
            translated_text = model.predict([prefixed_input])
        
        print(f"Model output: '{translated_text[0]}'")

        st.subheader("Translated Text:")
        if translated_text and translated_text[0]:
            st.success(translated_text[0])
        else:
            st.error("The model produced an empty translation.")
            
    else:
        st.warning("Please enter some text to translate.")