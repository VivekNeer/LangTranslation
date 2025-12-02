"""
Test script to verify Flask app and database integration.
Tests all major functionality including translation, history, and statistics.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_translation():
    """Test translation endpoint."""
    print("\n=== Testing Translation ===")
    
    test_texts = [
        "Hello",
        "Good morning",
        "How are you?",
        "Thank you",
        "Welcome"
    ]
    
    for text in test_texts:
        print(f"\nTranslating: '{text}'")
        
        response = requests.post(
            f"{BASE_URL}/translate",
            json={
                "text": text,
                "direction": "en-tu",
                "include_details": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✓ Translation: {data['output']}")
                print(f"  Translation ID: {data['translation_id']}")
            else:
                print(f"✗ Error: {data.get('error')}")
        else:
            print(f"✗ HTTP Error: {response.status_code}")
        
        time.sleep(0.5)  # Avoid overwhelming the server

def test_history():
    """Test history endpoints."""
    print("\n=== Testing History ===")
    
    # Get recent history
    response = requests.get(f"{BASE_URL}/history?limit=10")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print(f"✓ Retrieved {data['count']} translations from history")
            
            if data['count'] > 0:
                # Show first translation
                first = data['translations'][0]
                print(f"\nMost recent translation:")
                print(f"  Input: {first['input_text']}")
                print(f"  Output: {first['output_text']}")
                print(f"  Timestamp: {first['timestamp']}")
        else:
            print(f"✗ Error: {data.get('error')}")
    else:
        print(f"✗ HTTP Error: {response.status_code}")

def test_search():
    """Test search functionality."""
    print("\n=== Testing Search ===")
    
    search_term = "hello"
    response = requests.get(f"{BASE_URL}/search?q={search_term}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print(f"✓ Found {data['count']} results for '{search_term}'")
            
            for i, result in enumerate(data['results'][:3], 1):
                print(f"\n  Result {i}:")
                print(f"    Input: {result['input_text']}")
                print(f"    Output: {result['output_text']}")
        else:
            print(f"✗ Error: {data.get('error')}")
    else:
        print(f"✗ HTTP Error: {response.status_code}")

def test_statistics():
    """Test statistics endpoint."""
    print("\n=== Testing Statistics ===")
    
    response = requests.get(f"{BASE_URL}/statistics")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            stats = data['statistics']
            print(f"✓ Statistics retrieved successfully:")
            print(f"\n  Total Translations: {stats['total_translations']}")
            print(f"  Last 24 Hours: {stats['last_24_hours']}")
            print(f"  By Direction: {stats.get('by_direction', {})}")
            
            if 'model' in stats:
                print(f"\n  Model Information:")
                model = stats['model']
                print(f"    Name: {model.get('name')}")
                print(f"    BLEU Score: {model.get('bleu_score')}")
                print(f"    Exact Match Rate: {model.get('exact_match_rate')}%")
                print(f"    Character Accuracy: {model.get('char_accuracy')}%")
                print(f"    Training Date: {model.get('training_date')}")
                print(f"    Epochs: {model.get('total_epochs')}")
                print(f"    Dataset Size: {model.get('dataset_size')}")
            
            if stats.get('popular_translations'):
                print(f"\n  Popular Translations:")
                for i, item in enumerate(stats['popular_translations'][:5], 1):
                    print(f"    {i}. '{item['text']}' ({item['count']} times)")
        else:
            print(f"✗ Error: {data.get('error')}")
    else:
        print(f"✗ HTTP Error: {response.status_code}")

def test_database_direct():
    """Test database directly."""
    print("\n=== Testing Database Directly ===")
    
    try:
        from database import TranslationDatabase
        
        db = TranslationDatabase()
        
        # Get statistics
        stats = db.get_statistics()
        print(f"✓ Database accessible")
        print(f"  Total translations in DB: {stats['total_translations']}")
        
        # Get recent translations
        recent = db.get_recent_translations(limit=5)
        print(f"  Recent translations: {len(recent)}")
        
        # Get model metadata
        model_meta = db.get_latest_model_metadata()
        if model_meta:
            print(f"  Model metadata present: {model_meta.get('model_name')}")
        
    except Exception as e:
        print(f"✗ Database error: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Flask App Integration Test Suite")
    print("=" * 60)
    print(f"\nTesting Flask app at: {BASE_URL}")
    print("Make sure the Flask app is running!")
    print("=" * 60)
    
    try:
        # Test if server is running
        response = requests.get(BASE_URL, timeout=2)
        print("✓ Server is running")
    except requests.exceptions.RequestException as e:
        print(f"✗ Server is not running or not accessible: {e}")
        print("\nPlease start the Flask app first:")
        print("  python flask_app.py")
        return
    
    # Run all tests
    test_translation()
    test_history()
    test_search()
    test_statistics()
    test_database_direct()
    
    print("\n" + "=" * 60)
    print("Test Suite Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
