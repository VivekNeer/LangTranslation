from flask import Flask, render_template, request, jsonify
from simpletransformers.t5 import T5Model
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- Global variables ---
translation_model = None
gemini_model = None
# --- NEW: Define prefixes for clarity ---
PREFIX_MAP = {
    "en-kn": "translate english to kannada",
    "kn-en": "translate kannada to english",
}

def clean_gemini_response(text):
    """Clean Gemini API response to extract JSON"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    return text

def load_translation_model():
    """Load the locally trained HuggingFace mT5 translation model for Kannada."""
    global translation_model
    
    if translation_model is None:
        # --- Point to your local Kannada model ---
        model_path = "outputs/mt5-kannada-english-100k"
        print(f"Loading translation model from: {model_path}")
        
        # Make sure to specify model_type='mt5'
        translation_model = T5Model("mt5", model_path, use_cuda=False)
        
        # Set generation arguments for good quality output
        translation_model.args.num_beams = 5
        translation_model.args.max_length = 50
        
        print("Translation model loaded successfully.")
    
    return translation_model

def load_gemini_model():
    """Initialize Google Gemini API"""
    global gemini_model
    
    if gemini_model is None:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables")
            return None
        
        genai.configure(api_key=api_key)
        # --- Using the stable 'gemini-pro' model ---
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("Gemini API initialized successfully.")
    
    return gemini_model

def translate_text(input_text, prefix):
    """Translate text using the HuggingFace model with a required prefix."""
    model = load_translation_model()
    
    prefixed_input = f"{prefix}: {input_text}"
    print(f"Translating: {prefixed_input}")
    
    translated_text = model.predict([prefixed_input])
    
    output = translated_text[0] if translated_text else ""
    print(f"Translation: {output}")
    
    return output

def get_examples_from_gemini(word):
    """Get example sentences using Gemini API and translate them to Kannada."""
    gemini = load_gemini_model()
    if not gemini: return []
    
    try:
        prompt = f"""
        Generate 3 practical example sentences for the English word/phrase "{word}".
        Provide the response as a JSON array of objects:
        [
            {{"english": "example sentence in English"}},
            {{"english": "example sentence in English"}},
            {{"english": "example sentence in English"}}
        ]
        Only return the JSON array, nothing else. Do not use markdown formatting.
        """
        response = gemini.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        examples = json.loads(cleaned_response)
        
        prefix_en_kn = PREFIX_MAP["en-kn"]
        for example in examples:
            english_text = example.get('english', '')
            if english_text:
                kannada_translation = translate_text(english_text, prefix_en_kn)
                # --- CHANGED: Use the key 'tulu' so the frontend can find it ---
                example['tulu'] = kannada_translation
        
        return examples
    except Exception as e:
        print(f"Error getting examples from Gemini: {e}")
        return []

def get_synonyms_from_gemini(word):
    """Get synonyms using Gemini API and translate them to Kannada."""
    gemini = load_gemini_model()
    if not gemini: return []
    
    try:
        prompt = f"""
        For the English word/phrase "{word}", provide 3-5 diverse synonyms or related words.
        Provide the response as a JSON array of strings:
        ["synonym1", "synonym2", "synonym3"]
        Only return the JSON array, nothing else. Do not use markdown formatting.
        """
        response = gemini.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        synonyms_list = json.loads(cleaned_response)
        
        prefix_en_kn = PREFIX_MAP["en-kn"]
        synonyms_with_translation = []
        for synonym in synonyms_list:
            if synonym:
                kannada_translation = translate_text(synonym, prefix_en_kn)
                synonyms_with_translation.append({
                    'english': synonym,
                    # --- CHANGED: Use the key 'tulu' so the frontend can find it ---
                    'tulu': kannada_translation
                })
        
        return synonyms_with_translation
    except Exception as e:
        print(f"Error getting synonyms from Gemini: {e}")
        return []

def get_word_info_from_gemini(word, translation):
    """Get detailed word information using Gemini API for Kannada."""
    gemini = load_gemini_model()
    if not gemini: return {}
    
    try:
        prompt = f"""
        For the English word/phrase "{word}" (Kannada: "{translation}"), provide:
        1. Part of speech (e.g., noun, verb, adjective).
        2. A simple phonetic pronunciation guide.
        3. A brief, one-sentence usage note.
        Provide the response as a clean JSON object:
        {{
            "part_of_speech": "noun",
            "pronunciation": "heh-loh",
            "usage_note": "A common greeting used in most situations."
        }}
        Only return the JSON object, nothing else. Do not use markdown formatting.
        """
        response = gemini.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        info = json.loads(cleaned_response)
        return info
    except Exception as e:
        print(f"Error getting word info from Gemini: {e}")
        return {}

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation requests with enhanced features."""
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip()
        direction = data.get('direction', 'en-kn')
        include_details = data.get('include_details', True)
        
        if not input_text:
            return jsonify({'success': False, 'error': 'Please enter text.'}), 400
        
        prefix = PREFIX_MAP.get(direction)
        if not prefix:
            return jsonify({'success': False, 'error': 'Invalid translation direction.'}), 400
        
        translated = translate_text(input_text, prefix)
        
        if not translated:
            return jsonify({'success': False, 'error': 'Model produced an empty translation.'}), 500
        
        response_data = {
            'success': True,
            'input': input_text,
            'output': translated
        }
        
        if include_details and direction == 'en-kn':
            response_data['word_info'] = get_word_info_from_gemini(input_text, translated)
            response_data['examples'] = get_examples_from_gemini(input_text)
            response_data['synonyms'] = get_synonyms_from_gemini(input_text)
        
        return jsonify(response_data)
            
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    print("Initializing Flask app...")
    print("Loading translation model (this might take a moment)...")
    load_translation_model()
    print("Initializing Gemini API...")
    load_gemini_model()
    
    print("\n--- Flask Server is Starting ---")
    print("Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

