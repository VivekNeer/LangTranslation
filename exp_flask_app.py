"""
Flask web application for English-Tulu translation with database integration.
Supports both custom trained mT5 model and IndicTrans2.
Includes translation confidence and alternative predictions.
"""

from flask import Flask, render_template, request, jsonify
import torch
import os
import json
import uuid
from datetime import datetime
import numpy as np

# Import database module
from database import TranslationDatabase

app = Flask(__name__)

# Configuration - Using trained IndicTrans2 model with LoRA adapters
BASE_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_PATH = "indictrans2-200m-en-tulu"  # Your trained LoRA adapter

MODEL_TYPE = "indictrans2-lora"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DEFAULT_DIRECTION = "en-tu"

# Language codes used during training
SRC_LANG = "eng_Latn"
TGT_LANG = "kan_Knda"  # Using Kannada script for Tulu

# Translation model cache
translation_model = None
translation_tokenizer = None
indic_processor = None
db = None


def load_translation_model():
    """Load the trained IndicTrans2 model with LoRA adapters."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    from IndicTransToolkit import IndicProcessor
    
    global translation_model, translation_tokenizer, indic_processor
    if translation_model is None or translation_tokenizer is None:
        try:
            print(f"Loading base model: {BASE_MODEL}")
            print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
            
            # Initialize IndicProcessor
            indic_processor = IndicProcessor(inference=True)
            print("IndicProcessor initialized")
            
            # Load tokenizer from base model (IndicTrans2 has special tokenizer requirements)
            translation_tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                token=HF_TOKEN
            )
            print("Tokenizer loaded from base model")
            
            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                token=HF_TOKEN
            )
            print("Base model loaded")
            
            # Load LoRA adapter
            translation_model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_PATH,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("LoRA adapter loaded")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                translation_model = translation_model.cuda()
                print("Model loaded on GPU")
            else:
                print("Model loaded on CPU")
                
            print(f"IndicTrans2 with LoRA adapter loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nPlease ensure:")
            print("1. Base model access: https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M")
            print("2. Login: huggingface-cli login")
            print("3. Install: pip install IndicTransToolkit peft")
            print(f"4. Adapter exists at: {ADAPTER_PATH}")
            raise
    return translation_model, translation_tokenizer, indic_processor


def load_database():
    """Initialize database connection."""
    global db
    if db is None:
        db = TranslationDatabase()
        print("Database initialized successfully")
    return db


def translate_with_indictrans2(text, src_lang=None, tgt_lang=None, num_beams=5):
    """Translate using trained IndicTrans2 model with LoRA."""
    # Use configured language codes
    if src_lang is None:
        src_lang = SRC_LANG
    if tgt_lang is None:
        tgt_lang = TGT_LANG
    
    model, tokenizer, processor = load_translation_model()
    
    try:
        # Preprocess the input using IndicProcessor
        batch = processor.preprocess_batch(
            [text],
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        
        # Tokenize
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=256,
                early_stopping=True
            )
        
        # Decode
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Postprocess using IndicProcessor
        translations = processor.postprocess_batch(translations, lang=tgt_lang)
        
        return translations[0].strip() if translations else ""
        
    except Exception as e:
        print(f"Translation error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def translate_text(text, **kwargs):
    """Translate text using trained IndicTrans2 model."""
    return translate_with_indictrans2(text, **kwargs)


def translate_with_confidence(text, direction="en-tu", num_alternatives=3):
    """
    Translate text and provide confidence score with alternatives.
    Returns: (translation, confidence, alternatives)
    """
    if direction != "en-tu":
        raise ValueError(f"Only 'en-tu' (English to Tulu) is supported. Got: {direction}")
    
    try:
        # Get primary translation
        translation = translate_with_indictrans2(text, num_beams=5)
        
        # Calculate confidence based on translation length and input
        input_length = len(text.split())
        output_length = len(translation.split())
        
        # Simple heuristic for confidence
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
        
        # Generate alternative translations using different beam sizes
        alternatives = []
        try:
            # Alternative with fewer beams
            alt1 = translate_with_indictrans2(text, num_beams=3)
            if alt1 and alt1 != translation:
                alternatives.append({
                    'text': alt1,
                    'confidence': max(confidence - 10, 30.0),
                    'reason': 'Alternative translation (beam=3)'
                })
            
            # Greedy decoding alternative
            alt2 = translate_with_indictrans2(text, num_beams=1)
            if alt2 and alt2 != translation and alt2 not in [a['text'] for a in alternatives]:
                alternatives.append({
                    'text': alt2,
                    'confidence': max(confidence - 15, 25.0),
                    'reason': 'Greedy translation'
                })
                
        except Exception as e:
            print(f"Error generating alternatives: {e}")
        
        return translation.strip(), confidence, alternatives
        
    except Exception as e:
        print(f"Translation error: {e}")
        import traceback
        traceback.print_exc()
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
    return render_template('index_shadcn.html')


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
                'error': 'The model produced an empty translation. Please try again.'
            }), 500

        # Save to database
        database = load_database()
        translation_id = database.save_translation(
            input_text=input_text,
            output_text=translated,
            direction=direction,
            model_version=MODEL_TYPE,
            session_id=session_id,
            metadata={
                'include_details': include_details,
                'confidence': confidence,
                'base_model': BASE_MODEL,
                'adapter_path': ADAPTER_PATH
            }
        )

        response_data = {
            'success': True,
            'input': input_text,
            'output': translated,
            'direction': direction,
            'translation_id': translation_id,
            'session_id': session_id,
            'confidence': round(confidence, 1),
            'model': MODEL_TYPE
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
        import traceback
        traceback.print_exc()
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
    
    print(f"\nUsing Model: IndicTrans2 with LoRA Adapter")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Adapter Path: {ADAPTER_PATH}")
    
    print("\n1. Loading translation model...")
    try:
        load_translation_model()
    except Exception as e:
        print(f"\nERROR: Could not load model!")
        print("\nPlease ensure:")
        print("1. Base model access: https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M")
        print("2. Login: huggingface-cli login")
        print("3. Install: pip install IndicTransToolkit peft")
        print(f"4. Adapter exists at: {ADAPTER_PATH}")
        exit(1)
    
    print("\n2. Setting up database...")
    database = load_database()
    
    # Save model metadata if not already present
    if not database.get_latest_model_metadata():
        print("\n3. Saving model metadata...")
        database.save_model_metadata(
            model_name='indictrans2-200m-en-tulu-lora',
            model_path=ADAPTER_PATH,
            bleu_score=None,  # Update after evaluation
            exact_match_rate=None,
            char_accuracy=None,
            training_date='2024-12-02',
            total_epochs=5,
            dataset_size=None  # Update with your dataset size
        )
        print("   Model metadata saved successfully")
    else:
        print("\n3. Model metadata already exists in database")
    
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
    print("=" * 60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)