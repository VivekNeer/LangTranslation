"""
Database module for translation history and user sessions.
Handles all database operations for the translation app.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json


class TranslationDatabase:
    def __init__(self, db_path='translations.db'):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def init_db(self):
        """Initialize database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Translations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                direction TEXT NOT NULL,
                model_version TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                metadata TEXT
            )
        ''')
        
        # Sessions table (optional, for future use)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                ip_address TEXT
            )
        ''')
        
        # Model metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                bleu_score REAL,
                exact_match_rate REAL,
                char_accuracy REAL,
                training_date DATETIME,
                total_epochs INTEGER,
                dataset_size INTEGER,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_translations_timestamp 
            ON translations(timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_translations_direction 
            ON translations(direction)
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def save_translation(self, 
                        input_text: str, 
                        output_text: str, 
                        direction: str,
                        model_version: str = None,
                        session_id: str = None,
                        metadata: Dict = None) -> int:
        """Save a translation to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO translations 
            (input_text, output_text, direction, model_version, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (input_text, output_text, direction, model_version, session_id, metadata_json))
        
        translation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return translation_id
    
    def get_recent_translations(self, limit: int = 50, direction: str = None) -> List[Dict]:
        """Get recent translations, optionally filtered by direction."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if direction:
            cursor.execute('''
                SELECT * FROM translations 
                WHERE direction = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (direction, limit))
        else:
            cursor.execute('''
                SELECT * FROM translations 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search_translations(self, search_text: str, limit: int = 50) -> List[Dict]:
        """Search translations by input or output text."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{search_text}%"
        cursor.execute('''
            SELECT * FROM translations 
            WHERE input_text LIKE ? OR output_text LIKE ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (search_pattern, search_pattern, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_translation_by_id(self, translation_id: int) -> Optional[Dict]:
        """Get a specific translation by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM translations WHERE id = ?', (translation_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def delete_translation(self, translation_id: int) -> bool:
        """Delete a translation by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM translations WHERE id = ?', (translation_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_statistics(self) -> Dict:
        """Get translation statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total translations
        cursor.execute('SELECT COUNT(*) as total FROM translations')
        total = cursor.fetchone()['total']
        
        # Translations by direction
        cursor.execute('''
            SELECT direction, COUNT(*) as count 
            FROM translations 
            GROUP BY direction
        ''')
        by_direction = {row['direction']: row['count'] for row in cursor.fetchall()}
        
        # Recent activity (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) as count 
            FROM translations 
            WHERE timestamp >= datetime('now', '-1 day')
        ''')
        last_24h = cursor.fetchone()['count']
        
        # Most translated texts
        cursor.execute('''
            SELECT input_text, COUNT(*) as count 
            FROM translations 
            GROUP BY LOWER(input_text)
            ORDER BY count DESC
            LIMIT 10
        ''')
        popular = [{'text': row['input_text'], 'count': row['count']} 
                   for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_translations': total,
            'by_direction': by_direction,
            'last_24_hours': last_24h,
            'popular_translations': popular
        }
    
    def save_model_metadata(self, 
                           model_name: str,
                           model_path: str,
                           bleu_score: float = None,
                           exact_match_rate: float = None,
                           char_accuracy: float = None,
                           training_date: str = None,
                           total_epochs: int = None,
                           dataset_size: int = None,
                           metadata: Dict = None) -> int:
        """Save model metadata to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO model_metadata 
            (model_name, model_path, bleu_score, exact_match_rate, char_accuracy,
             training_date, total_epochs, dataset_size, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, model_path, bleu_score, exact_match_rate, char_accuracy,
              training_date, total_epochs, dataset_size, metadata_json))
        
        metadata_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return metadata_id
    
    def get_latest_model_metadata(self) -> Optional[Dict]:
        """Get the most recent model metadata."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_metadata 
            ORDER BY created_at DESC 
            LIMIT 1
        ''')
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None


# Convenience function for quick database access
def get_db():
    """Get database instance."""
    return TranslationDatabase()


if __name__ == '__main__':
    # Test database creation
    db = TranslationDatabase()
    
    # Save sample model metadata
    db.save_model_metadata(
        model_name='mt5-english-tulu',
        model_path='outputs/mt5-english-tulu',
        bleu_score=8.40,
        exact_match_rate=20.20,
        char_accuracy=83.32,
        training_date='2024-11-27',
        total_epochs=10,
        dataset_size=8300
    )
    
    print("Database created successfully!")
    print("\nTest statistics:")
    stats = db.get_statistics()
    print(json.dumps(stats, indent=2))
