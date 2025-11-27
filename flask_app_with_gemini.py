"""
Flask web application for English-Tulu translation with database integration.
Uses locally trained model and stores translation history.
"""

from flask import Flask, render_template, request, jsonify
from simpletransformers.t5 import T5Model
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import uuid
from datetime import datetime

# Import database module
from database import TranslationDatabase

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
LOCAL_MODEL_PATH = "outputs/mt5-english-tulu"
MODEL_NAME = "mt5-english-tulu"
DEFAULT_DIRECTION = "en-tu"

# Translation model cache
translation_model = None
gemini_model = None
db = None


def load_translation_model():
    """Load the local trained translation model."""
    global translation_model
    if translation_model is None:
        try:
            print(f"Loading local model from: {LOCAL_MODEL_PATH}")
            translation_model = T5Model(
                "mt5",
                LOCAL_MODEL_PATH,
                use_cuda=False
            )
            print(f"Model loaded successfully from {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading local model: {e}")
            print("Make sure the model exists at outputs/mt5-english-tulu")
            raise
    return translation_model


def load_gemini_model():
    """Initialize Gemini AI for enrichment features."""
    global gemini_model
    if gemini_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in .env file")
            print("Enrichment features (examples, synonyms) will be disabled")
            return None
        try:
            genai.configure(api_key=api_key)
            # Use gemini-1.5-flash-latest for the latest stable version
            gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            print("Gemini API initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            return None
    return gemini_model


def load_database():
    """Initialize database connection."""
    global db
    if db is None:
        db = TranslationDatabase()
        print("Database initialized successfully")
    return db


def translate_text(text, direction="en-tu"):
    """
    Translate text using the local model.
    Currently only supports English to Tulu (en-tu).
    """
    if direction != "en-tu":
        raise ValueError(f"Only 'en-tu' (English to Tulu) is supported. Got: {direction}")
    
    model = load_translation_model()
    
    # Format input with direction prefix
    input_text = text
    
    try:
        predictions = model.predict([input_text])
        translation = predictions[0] if predictions else ""
        return translation.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return ""


def clean_gemini_response(response_text):
    """Clean and parse Gemini API response."""
    response_text = response_text.strip()
    
    # Remove markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first and last lines (```json and ```)
        response_text = "\n".join(lines[1:-1])
    
    # Remove any remaining ```
    response_text = response_text.replace("```json", "").replace("```", "")
    
    return response_text.strip()


def get_examples_from_gemini(word, direction="en-tu"):
    """Get example sentences using Gemini AI."""
    gemini = load_gemini_model()
    if not gemini:
        return []
    
    try:
        prompt = f"""
        For the English word or phrase "{word}", provide 3 example sentences.
        Each sentence should be simple and demonstrate proper usage.

        Provide the response as a JSON array:
        [
            {{"english": "Example sentence 1"}},
            {{"english": "Example sentence 2"}},
            {{"english": "Example sentence 3"}}
        ]

        Only return the JSON array, nothing else. Do not use markdown formatting.
        """
        
        response = gemini.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        print(f"Gemini examples response: {cleaned_response[:200]}...")
        
        examples = json.loads(cleaned_response)
        
        # Translate each example to Tulu
        for example in examples:
            english_text = example.get("english", "")
            if english_text:
                tulu_translation = translate_text(english_text, direction)
                example["tulu"] = tulu_translation
        
        return examples
    except Exception as e:
        print(f"Error getting examples from Gemini: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
        return []


def get_synonyms_from_gemini(word, direction="en-tu"):
    """Get synonyms using Gemini AI."""
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
        synonyms_with_translation = []
        
        for synonym in synonyms_list:
            if synonym:
                tulu_translation = translate_text(synonym, direction)
                synonyms_with_translation.append({
                    "english": synonym,
                    "tulu": tulu_translation,
                })
        
        return synonyms_with_translation
    except Exception as e:
        print(f"Error getting synonyms from Gemini: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
        return []


def get_word_info_from_gemini(word, translation):
    """Get word information using Gemini AI."""
    gemini = load_gemini_model()
    if not gemini:
        return {}
    
    try:
        prompt = f"""
        For the English word/phrase "{word}" (Tulu Translation: "{translation}"), provide:
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
    """Render the main page."""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """Main translation endpoint with database integration."""
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip()
        include_details = data.get('include_details', True)
        direction = data.get('direction', DEFAULT_DIRECTION)
        session_id = data.get('session_id') or str(uuid.uuid4())

        if not input_text:
            return jsonify({
                'success': False, 
                'error': 'Please enter some text to translate.'
            }), 400

        # Perform translation
        translated = translate_text(input_text, direction)
        
        if not translated:
            return jsonify({
                'success': False, 
                'error': 'The model produced an empty translation.'
            }), 500

        # Save to database
        database = load_database()
        translation_id = database.save_translation(
            input_text=input_text,
            output_text=translated,
            direction=direction,
            model_version=MODEL_NAME,
            session_id=session_id,
            metadata={'include_details': include_details}
        )

        response_data = {
            'success': True,
            'input': input_text,
            'output': translated,
            'direction': direction,
            'translation_id': translation_id,
            'session_id': session_id
        }

        # Add enrichment details if requested
        if include_details:
            word_info = get_word_info_from_gemini(input_text, translated)
            if word_info:
                response_data['word_info'] = word_info

            examples = get_examples_from_gemini(input_text, direction)
            if examples:
                response_data['examples'] = examples

            synonyms = get_synonyms_from_gemini(input_text, direction)
            if synonyms:
                response_data['synonyms'] = synonyms

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get translation history."""
    try:
        limit = request.args.get('limit', 50, type=int)
        direction = request.args.get('direction')
        
        database = load_database()
        translations = database.get_recent_translations(limit=limit, direction=direction)
        
        return jsonify({
            'success': True,
            'translations': translations,
            'count': len(translations)
        })
    except Exception as e:
        print(f"Error getting history: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/history/<int:translation_id>', methods=['GET'])
def get_translation(translation_id):
    """Get a specific translation by ID."""
    try:
        database = load_database()
        translation = database.get_translation_by_id(translation_id)
        
        if not translation:
            return jsonify({
                'success': False,
                'error': 'Translation not found'
            }), 404
        
        return jsonify({
            'success': True,
            'translation': translation
        })
    except Exception as e:
        print(f"Error getting translation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/history/<int:translation_id>', methods=['DELETE'])
def delete_translation_endpoint(translation_id):
    """Delete a translation by ID."""
    try:
        database = load_database()
        deleted = database.delete_translation(translation_id)
        
        if not deleted:
            return jsonify({
                'success': False,
                'error': 'Translation not found'
            }), 404
        
        return jsonify({
            'success': True,
            'message': 'Translation deleted successfully'
        })
    except Exception as e:
        print(f"Error deleting translation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/search', methods=['GET'])
def search_translations():
    """Search translations by text."""
    try:
        search_text = request.args.get('q', '').strip()
        limit = request.args.get('limit', 50, type=int)
        
        if not search_text:
            return jsonify({
                'success': False,
                'error': 'Search query is required'
            }), 400
        
        database = load_database()
        results = database.search_translations(search_text, limit=limit)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        print(f"Error searching translations: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get translation statistics."""
    try:
        database = load_database()
        stats = database.get_statistics()
        
        # Add model metadata
        model_meta = database.get_latest_model_metadata()
        if model_meta:
            stats['model'] = {
                'name': model_meta.get('model_name'),
                'bleu_score': model_meta.get('bleu_score'),
                'exact_match_rate': model_meta.get('exact_match_rate'),
                'char_accuracy': model_meta.get('char_accuracy'),
                'training_date': model_meta.get('training_date'),
                'total_epochs': model_meta.get('total_epochs'),
                'dataset_size': model_meta.get('dataset_size')
            }
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/examples', methods=['POST'])
def get_examples():
    """Get example sentences for a word."""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        direction = data.get('direction', DEFAULT_DIRECTION)

        if not word:
            return jsonify({
                'success': False,
                'error': 'Word is required.'
            }), 400

        examples = get_examples_from_gemini(word, direction)

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
    """Get synonyms for a word."""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        direction = data.get('direction', DEFAULT_DIRECTION)
        
        if not word:
            return jsonify({
                'success': False,
                'error': 'Word is required.'
            }), 400

        synonyms = get_synonyms_from_gemini(word, direction)

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
    # Initialize app components
    print("=" * 60)
    print("Initializing English-Tulu Translation App")
    print("=" * 60)
    
    print("\n1. Loading translation model...")
    load_translation_model()
    
    print("\n2. Initializing Gemini API...")
    load_gemini_model()
    
    print("\n3. Setting up database...")
    database = load_database()
    
    # Save model metadata if not already present
    if not database.get_latest_model_metadata():
        print("\n4. Saving model metadata...")
        database.save_model_metadata(
            model_name='mt5-english-tulu',
            model_path=LOCAL_MODEL_PATH,
            bleu_score=8.40,
            exact_match_rate=20.20,
            char_accuracy=83.32,
            training_date='2024-11-27',
            total_epochs=10,
            dataset_size=8300
        )
        print("   Model metadata saved successfully")
    else:
        print("\n4. Model metadata already exists in database")
    
    print("\n" + "=" * 60)
    print("Flask server starting...")
    print("Server will be available at: http://localhost:5000")
    print("=" * 60)
    print("\nEndpoints:")
    print("  - POST /translate          - Translate text")
    print("  - GET  /history            - Get translation history")
    print("  - GET  /history/<id>       - Get specific translation")
    print("  - DELETE /history/<id>     - Delete translation")
    print("  - GET  /search?q=<text>    - Search translations")
    print("  - GET  /statistics         - Get usage statistics")
    print("  - POST /examples           - Get example sentences")
    print("  - POST /synonyms           - Get synonyms")
    print("=" * 60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
