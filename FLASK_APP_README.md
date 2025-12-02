# Flask Translation App with Database Integration

## Overview
This is a comprehensive English-to-Tulu translation web application that uses a locally trained mT5 model and stores all translation history in a SQLite database.

## Features

### 🎯 Core Translation
- **Local Model**: Uses the trained model from `outputs/mt5-english-tulu` (BLEU: 8.40)
- **English → Tulu**: Currently supports English to Tulu translation
- **Real-time Translation**: Fast translation with CPU-based inference

### 🗄️ Database Integration
- **SQLite Database**: Stores all translations in `translations.db`
- **Translation History**: Every translation is automatically saved
- **Search Functionality**: Search through past translations
- **Statistics Tracking**: Monitor usage patterns and popular translations

### 🤖 AI Enrichment (Optional)
- **Example Sentences**: AI-generated usage examples
- **Synonyms**: Related words and translations
- **Word Information**: Part of speech, pronunciation, usage notes
- **Powered by Gemini API**: Requires `GEMINI_API_KEY` in `.env` file

### 📊 Web Interface Features
- **Translation Tab**: Main translation interface with AI enrichment
- **History Tab**: View and manage translation history
- **Statistics Tab**: View usage statistics and model performance metrics
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
/home/vivek/LangTranslation/
├── flask_app.py              # Main Flask application with database integration
├── database.py               # SQLite database module
├── translations.db           # SQLite database file (created automatically)
├── templates/
│   ├── index.html            # Enhanced HTML frontend with history and stats
│   └── index_old.html        # Backup of original template
├── flask_app_old.py          # Backup of original Flask app
├── outputs/
│   └── mt5-english-tulu/     # Trained local model
└── .env                      # Environment variables (optional Gemini API key)
```

## Installation

### Prerequisites
- Python 3.9+
- Conda environment: `simple-t5-env`

### Dependencies
```bash
pip install flask python-dotenv google-generativeai simpletransformers
```

### Environment Setup (Optional)
Create a `.env` file in the project root for AI enrichment features:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Database Schema

### Tables

#### 1. translations
Stores all translation history:
- `id`: Primary key
- `input_text`: Original English text
- `output_text`: Translated Tulu text
- `direction`: Translation direction (en-tu)
- `model_version`: Model name used
- `timestamp`: When translation was performed
- `session_id`: User session identifier
- `metadata`: Additional JSON metadata

#### 2. sessions
Tracks user sessions (for future analytics):
- `session_id`: Primary key
- `created_at`: Session start time
- `last_active`: Last activity timestamp
- `user_agent`: Browser information
- `ip_address`: Client IP

#### 3. model_metadata
Stores model performance metrics:
- `model_name`: Model identifier
- `model_path`: Path to model files
- `bleu_score`: BLEU evaluation score (8.40)
- `exact_match_rate`: Percentage of exact matches (20.20%)
- `char_accuracy`: Character-level accuracy (83.32%)
- `training_date`: When model was trained
- `total_epochs`: Training epochs (10)
- `dataset_size`: Training samples (8,300)

## API Endpoints

### Translation
- `POST /translate` - Translate text
  ```json
  {
    "text": "Hello",
    "direction": "en-tu",
    "include_details": true,
    "session_id": "optional"
  }
  ```

### History Management
- `GET /history?limit=50&direction=en-tu` - Get translation history
- `GET /history/<id>` - Get specific translation by ID
- `DELETE /history/<id>` - Delete a translation
- `GET /search?q=<text>&limit=50` - Search translations

### Statistics
- `GET /statistics` - Get usage statistics and model metrics

### AI Enrichment (requires Gemini API key)
- `POST /examples` - Get example sentences
- `POST /synonyms` - Get synonyms

## Usage

### Starting the Server
```bash
cd /home/vivek/LangTranslation
python flask_app.py
```

The server will start on:
- Local: http://127.0.0.1:5000
- Network: http://0.0.0.0:5000

### Using the Web Interface

1. **Translation Tab**:
   - Enter English text in the input box
   - Click "Translate" to get Tulu translation
   - View AI-generated examples and synonyms (if enabled)
   - Copy translation to clipboard

2. **History Tab**:
   - Browse all past translations
   - Search through history
   - Reuse previous translations
   - Delete unwanted entries

3. **Statistics Tab**:
   - View total translations count
   - See translations in last 24 hours
   - Check model performance metrics
   - View most popular translations

### Programmatic Usage

#### Python Example
```python
import requests

# Translate text
response = requests.post('http://localhost:5000/translate', json={
    'text': 'Hello, how are you?',
    'direction': 'en-tu',
    'include_details': False
})

result = response.json()
print(f"Translation: {result['output']}")
print(f"Translation ID: {result['translation_id']}")

# Get history
history = requests.get('http://localhost:5000/history?limit=10')
print(history.json())

# Get statistics
stats = requests.get('http://localhost:5000/statistics')
print(stats.json())
```

#### JavaScript/Fetch Example
```javascript
// Translate text
fetch('/translate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        text: 'Good morning',
        direction: 'en-tu',
        include_details: true
    })
})
.then(response => response.json())
.then(data => {
    console.log('Translation:', data.output);
    console.log('Examples:', data.examples);
});

// Get history
fetch('/history?limit=20')
    .then(response => response.json())
    .then(data => console.log('History:', data.translations));
```

## Model Information

### Performance Metrics
- **BLEU Score**: 8.40
- **Exact Match Rate**: 20.20%
- **Character Accuracy**: 83.32%
- **Training Date**: November 27, 2024
- **Training Epochs**: 10
- **Dataset Size**: 8,300 English-Tulu pairs

### Model Architecture
- **Base Model**: google/mt5-small
- **Framework**: SimpleTransformers + T5Model
- **Training**: CPU-based, 75 minutes total
- **Location**: `outputs/mt5-english-tulu`

## Database Management

### Viewing Database
```bash
sqlite3 translations.db
.tables
SELECT * FROM translations LIMIT 10;
.quit
```

### Backing Up Database
```bash
cp translations.db translations_backup_$(date +%Y%m%d).db
```

### Resetting Database
```bash
rm translations.db
python database.py  # Reinitialize
```

### Exporting Data
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('translations.db')
df = pd.read_sql_query("SELECT * FROM translations", conn)
df.to_csv('translations_export.csv', index=False)
conn.close()
```

## Features Comparison

### Old vs New Flask App

| Feature | Old (flask_app_old.py) | New (flask_app.py) |
|---------|------------------------|-------------------|
| Model Source | HuggingFace | Local trained model |
| Database | None | SQLite with full history |
| Translation History | No | Yes, with search |
| Statistics | No | Yes, comprehensive |
| Session Tracking | No | Yes |
| API Endpoints | 5 | 11 |
| Frontend | Basic | Enhanced with tabs |
| Model Info Display | No | Yes, with metrics |

## Troubleshooting

### Model Not Found
```bash
# Check if model exists
ls -la outputs/mt5-english-tulu/

# If missing, train model:
python train_model.py
```

### Database Errors
```bash
# Reinitialize database
python database.py

# Check database integrity
sqlite3 translations.db "PRAGMA integrity_check;"
```

### Gemini API Not Working
- Ensure `.env` file exists with valid `GEMINI_API_KEY`
- Features will be disabled without API key (translation still works)
- Check API quota at https://makersuite.google.com/

### Port Already in Use
```bash
# Kill process on port 5000
lsof -i :5000
kill -9 <PID>

# Or use different port
python flask_app.py --port 5001
```

## Performance Tips

1. **Database Indexing**: Already optimized with indexes on timestamp and direction
2. **Model Caching**: Model is loaded once at startup and cached
3. **Batch Translations**: Process multiple translations in single request (future feature)
4. **Database Cleanup**: Periodically archive old translations
5. **Disable AI Enrichment**: Skip Gemini API calls for faster translations

## Future Enhancements

- [ ] Batch translation API endpoint
- [ ] User authentication and personalized history
- [ ] Translation quality feedback system
- [ ] Export history to CSV/Excel
- [ ] Multiple language pair support
- [ ] Translation confidence scores
- [ ] Real-time translation suggestions
- [ ] WebSocket support for live updates
- [ ] Model versioning and A/B testing
- [ ] Analytics dashboard with charts

## Credits

- **Model**: mT5-small by Google (fine-tuned locally)
- **Framework**: SimpleTransformers
- **Frontend**: Custom HTML/CSS/JavaScript
- **Database**: SQLite3
- **AI Enrichment**: Google Gemini API

## License

This project is for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the database schema
3. Verify model file integrity
4. Check Flask logs in terminal

---

**Last Updated**: November 2024
**Version**: 2.0 (Database-Integrated)
**Status**: Production Ready ✅
