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

# Global variables
translation_model = None
gemini_model = None

def clean_gemini_response(text):
    """Clean Gemini API response to extract JSON"""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # Strip whitespace
    text = text.strip()
    return text

def load_translation_model():
    """Load the HuggingFace mT5 translation model"""
    global translation_model
    
    if translation_model is None:
        model_path = "VivekNeer/mt5-sinhalese-english"
        print(f"Loading translation model from: {model_path}")
        
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
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("Gemini API initialized successfully.")
    
    return gemini_model

def translate_text(input_text):
    """Translate text using the HuggingFace model"""
    model = load_translation_model()
    
    print(f"Translating: {input_text}")
    translated_text = model.predict([input_text])
    
    output = translated_text[0] if translated_text else ""
    print(f"Translation: {output}")
    
    return output

def get_examples_from_gemini(word, translation):
    """Get example sentences using Gemini API"""
    gemini = load_gemini_model()
    if not gemini:
        return []
    
    try:
        prompt = f"""
        Generate 3 practical example sentences using the English word/phrase "{word}".
        
        Provide the response as a JSON array with this format:
        [
            {{"english": "example sentence in English"}},
            {{"english": "example sentence in English"}},
            {{"english": "example sentence in English"}}
        ]
        
        Only return the JSON array, nothing else. Do not use markdown formatting.
        """
        
        response = gemini.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        print(f"Gemini examples response: {cleaned_response[:200]}...")
        examples = json.loads(cleaned_response)
        
        # Translate each English example to Tulu using HuggingFace model
        model = load_translation_model()
        for example in examples:
            english_text = example.get('english', '')
            if english_text:
                tulu_translation = model.predict([english_text])
                example['tulu'] = tulu_translation[0] if tulu_translation else ""
        
        return examples
    except Exception as e:
        print(f"Error getting examples from Gemini: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
        return []

def get_synonyms_from_gemini(word):
    """Get synonyms and related words using Gemini API"""
    gemini = load_gemini_model()
    if not gemini:
        return []
    
    try:
        prompt = f"""
        For the English word/phrase "{word}", provide 3-5 synonyms or related words.
        
        Provide the response as a JSON array of strings:
        ["synonym1", "synonym2", "synonym3"]
        
        Only return the JSON array, nothing else. Do not use markdown formatting.
        """
        
        response = gemini.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        print(f"Gemini synonyms response: {cleaned_response[:200]}...")
        synonyms_list = json.loads(cleaned_response)
        
        # Translate each synonym to Tulu using HuggingFace model
        model = load_translation_model()
        synonyms_with_translation = []
        for synonym in synonyms_list:
            if synonym:
                tulu_translation = model.predict([synonym])
                synonyms_with_translation.append({
                    'english': synonym,
                    'tulu': tulu_translation[0] if tulu_translation else ""
                })
        
        return synonyms_with_translation
    except Exception as e:
        print(f"Error getting synonyms from Gemini: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
        return []

def get_word_info_from_gemini(word, translation):
    """Get detailed word information using Gemini API"""
    gemini = load_gemini_model()
    if not gemini:
        return {}
    
    try:
        prompt = f"""
        For the English word/phrase "{word}" (Tulu: "{translation}"), provide:
        1. Part of speech (noun, verb, adjective, etc.)
        2. Pronunciation guide (simple phonetic)
        3. Brief usage note (1 sentence)
        
        Provide the response as JSON:
        {{
            "part_of_speech": "noun",
            "pronunciation": "heh-loh",
            "usage_note": "Common greeting used in informal situations"
        }}
        
        Only return the JSON object, nothing else. Do not use markdown formatting.
        """
        
        response = gemini.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        print(f"Gemini word info response: {cleaned_response[:200]}...")
        info = json.loads(cleaned_response)
        return info
    except Exception as e:
        print(f"Error getting word info from Gemini: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
        return {}

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation requests with enhanced features"""
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip()
        include_details = data.get('include_details', True)
        
        if not input_text:
            return jsonify({
                'success': False,
                'error': 'Please enter some text to translate.'
            }), 400
        
        # Step 1: Perform translation using HuggingFace model
        translated = translate_text(input_text)
        
        if not translated:
            return jsonify({
                'success': False,
                'error': 'The model produced an empty translation.'
            }), 500
        
        # Prepare basic response
        response_data = {
            'success': True,
            'input': input_text,
            'output': translated
        }
        
        # Step 2: Get enhanced details from Gemini (if requested)
        if include_details:
            # Get word information
            word_info = get_word_info_from_gemini(input_text, translated)
            if word_info:
                response_data['word_info'] = word_info
            
            # Get examples
            examples = get_examples_from_gemini(input_text, translated)
            if examples:
                response_data['examples'] = examples
            
            # Get synonyms
            synonyms = get_synonyms_from_gemini(input_text)
            if synonyms:
                response_data['synonyms'] = synonyms
        
        return jsonify(response_data)
            
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/examples', methods=['POST'])
def get_examples():
    """Get example sentences for a word"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        translation = data.get('translation', '').strip()
        
        if not word or not translation:
            return jsonify({
                'success': False,
                'error': 'Word and translation are required.'
            }), 400
        
        examples = get_examples_from_gemini(word, translation)
        
        return jsonify({
            'success': True,
            'examples': examples
        })
            
    except Exception as e:
        print(f"Error getting examples: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/synonyms', methods=['POST'])
def get_synonyms():
    """Get synonyms for a word"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({
                'success': False,
                'error': 'Word is required.'
            }), 400
        
        synonyms = get_synonyms_from_gemini(word)
        
        return jsonify({
            'success': True,
            'synonyms': synonyms
        })
            
    except Exception as e:
        print(f"Error getting synonyms: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load models when starting the app
    print("Initializing Flask app...")
    print("Loading translation model...")
    load_translation_model()
    print("Initializing Gemini API...")
    load_gemini_model()
    
    # Run the Flask app
    print("Starting Flask server...")
    print("Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
