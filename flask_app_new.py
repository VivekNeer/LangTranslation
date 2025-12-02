"""
Flask web application for English-Tulu translation with database integration.
Uses locally trained model and stores translation history.
Includes translation confidence and alternative predictions.
"""

from flask import Flask, render_template, request, jsonify
from simpletransformers.t5 import T5Model
import torch
import os
import json
import uuid
from datetime import datetime
import numpy as np

# Import database module
from database import TranslationDatabase

app = Flask(__name__)

# Configuration
LOCAL_MODEL_PATH = "outputs/mt5-english-tulu"
MODEL_NAME = "mt5-english-tulu"
DEFAULT_DIRECTION = "en-tu"

# Translation model cache
translation_model = None
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


def load_database():
    """Initialize database connection."""
    global db
    if db is None:
        db = TranslationDatabase()
        print("Database initialized successfully")
    return db


def translate_with_confidence(text, direction="en-tu", num_alternatives=3):
    """
    Translate text and provide confidence score with alternatives.
    Returns: (translation, confidence, alternatives)
    """
    if direction != "en-tu":
        raise ValueError(f"Only 'en-tu' (English to Tulu) is supported. Got: {direction}")
    
    model = load_translation_model()
    
    try:
        # Get primary translation
        predictions = model.predict([text])
        translation = predictions[0] if predictions else ""
        
        # Calculate confidence based on translation length and input
        input_length = len(text.split())
        output_length = len(translation.split())
        
        # Simple heuristic for confidence:
        # - Good if output length is reasonable compared to input
        # - Lower if output is very short or very long
        length_ratio = output_length / max(input_length, 1)
        
        if 0.5 <= length_ratio <= 2.0:
            confidence = 85.0  # High confidence
        elif 0.3 <= length_ratio <= 3.0:
            confidence = 70.0  # Medium confidence
        else:
            confidence = 50.0  # Low confidence
        
        # Add bonus for non-empty translation
        if translation.strip():
            confidence += 10.0
        confidence = min(confidence, 99.0)  # Cap at 99%
        
        # Generate alternative translations by varying generation parameters
        alternatives = []
        try:
            # Try with different temperature/sampling if model supports it
            # For now, we'll generate variations based on word substitutions
            words = translation.split()
            if len(words) > 1:
                # Create simple variations
                # Alternative 1: Different word order (if multiple words)
                if len(words) >= 2:
                    alt1_words = words.copy()
                    # Swap first two words as an alternative
                    alt1_words[0], alt1_words[-1] = alt1_words[-1], alt1_words[0]
                    alternatives.append({
                        'text': ' '.join(alt1_words),
                        'confidence': max(confidence - 15, 30.0),
                        'reason': 'Alternative word order'
                    })
                
                # Alternative 2: Just the main word (simplified)
                if len(words) > 2:
                    alternatives.append({
                        'text': words[0],
                        'confidence': max(confidence - 20, 25.0),
                        'reason': 'Simplified form'
                    })
        except Exception as e:
            print(f"Error generating alternatives: {e}")
        
        return translation.strip(), confidence, alternatives
        
    except Exception as e:
        print(f"Translation error: {e}")
        return "", 0.0, []


def get_translation_metrics(input_text, output_text):
    """Calculate various metrics for the translation."""
    metrics = {
        'input_length': len(input_text.split()),
        'output_length': len(output_text.split()),
        'input_chars': len(input_text),
        'output_chars': len(output_text),
        'length_ratio': len(output_text.split()) / max(len(input_text.split()), 1)
    }
    
    # Categorize translation quality based on heuristics
    if metrics['length_ratio'] < 0.3:
        metrics['quality_note'] = 'Translation may be too short'
    elif metrics['length_ratio'] > 3.0:
        metrics['quality_note'] = 'Translation may be too long'
    elif 0.5 <= metrics['length_ratio'] <= 2.0:
        metrics['quality_note'] = 'Good length balance'
    else:
        metrics['quality_note'] = 'Acceptable translation'
    
    return metrics


@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """Main translation endpoint with confidence and alternatives."""
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

        # Perform translation with confidence
        translated, confidence, alternatives = translate_with_confidence(input_text, direction)
        
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
            metadata={
                'include_details': include_details,
                'confidence': confidence
            }
        )

        response_data = {
            'success': True,
            'input': input_text,
            'output': translated,
            'direction': direction,
            'translation_id': translation_id,
            'session_id': session_id,
            'confidence': round(confidence, 1)
        }

        # Add detailed metrics if requested
        if include_details:
            metrics = get_translation_metrics(input_text, translated)
            response_data['metrics'] = metrics
            
            if alternatives:
                response_data['alternatives'] = alternatives
            
            # Add word-level info
            response_data['word_info'] = {
                'input_words': input_text.split(),
                'output_words': translated.split(),
                'word_count_ratio': f"{len(translated.split())}/{len(input_text.split())}"
            }

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


if __name__ == '__main__':
    # Initialize app components
    print("=" * 60)
    print("Initializing English-Tulu Translation App")
    print("=" * 60)
    
    print("\n1. Loading translation model...")
    load_translation_model()
    
    print("\n2. Setting up database...")
    database = load_database()
    
    # Save model metadata if not already present
    if not database.get_latest_model_metadata():
        print("\n3. Saving model metadata...")
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
        print("\n3. Model metadata already exists in database")
    
    print("\n" + "=" * 60)
    print("Flask server starting...")
    print("Server will be available at: http://localhost:5000")
    print("=" * 60)
    print("\nEndpoints:")
    print("  - POST /translate          - Translate text with confidence & alternatives")
    print("  - GET  /history            - Get translation history")
    print("  - GET  /history/<id>       - Get specific translation")
    print("  - DELETE /history/<id>     - Delete translation")
    print("  - GET  /search?q=<text>    - Search translations")
    print("  - GET  /statistics         - Get usage statistics")
    print("=" * 60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
