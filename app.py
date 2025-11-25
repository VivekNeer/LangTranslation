import os
os.environ["TRANSFORMERS_USE_LEGACY_IMPORT"] = "True"

import streamlit as st
from simpletransformers.t5 import T5Model
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

TRANSLATION_DIRECTIONS = {
    "English → Kannada": {
        "prefix": "translate english to kannada",
        "placeholder": "Type English text to translate into Kannada",
    },
    "Kannada → Tulu": {
        "prefix": "translate kannada to tulu",
        "placeholder": "Type Kannada text to translate into Tulu",
    },
}


def _format_input(prefix: str, text: str) -> str:
    return f"{prefix}: {text.strip()}"


st.set_page_config(page_title="AI Translator", page_icon="🤖", layout="centered")


@st.cache_resource
def load_model():
    model_path = "VivekNeer/mt5-english-kannada-tulu"
    print(f"Loading model from: {model_path}")
    model = T5Model("mt5", model_path)
    model.args.num_beams = 5
    model.args.max_length = 50
    return model


st.title("AI Language Translator")
st.markdown("Translate between English → Kannada and Kannada → Tulu with a single fine-tuned `mT5-small` model.")

with st.spinner("Loading the trained model... This may take a moment."):
    model = load_model()

st.header("Enter Text to Translate")
direction_label = st.radio("Translation direction", list(TRANSLATION_DIRECTIONS.keys()), horizontal=True)
direction_config = TRANSLATION_DIRECTIONS[direction_label]
input_text = st.text_area("Input text:", height=120, placeholder=direction_config["placeholder"])

if st.button("Translate", type="primary"):
    if input_text.strip():
        prefixed_input = _format_input(direction_config["prefix"], input_text)
        with st.spinner("Translating..."):
            translated_text = model.predict([prefixed_input])

        print(f"Model output: '{translated_text[0]}'")

        st.subheader("Translated Text:")
        if translated_text and translated_text[0]:
            st.success(translated_text[0])
        else:
            st.error("The model produced an empty translation. It may need more training or a different input.")
    else:
        st.warning("Please enter some text to translate.")