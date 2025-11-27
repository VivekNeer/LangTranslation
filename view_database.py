"""
Simple script to view database contents without triggering Flask reload.
"""
import sqlite3
from datetime import datetime

def view_database():
    """View database contents."""
    try:
        conn = sqlite3.connect('translations.db')
        cursor = conn.cursor()
        
        # Total translations
        cursor.execute("SELECT COUNT(*) FROM translations")
        total = cursor.fetchone()[0]
        print(f"\n📊 Database Statistics")
        print(f"=" * 60)
        print(f"Total Translations: {total}")
        
        if total > 0:
            # Recent translations
            print(f"\n📝 Recent Translations:")
            print(f"-" * 60)
            cursor.execute("""
                SELECT id, input_text, output_text, direction, timestamp 
                FROM translations 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                tid, inp, out, direction, timestamp = row
                print(f"\nID: {tid}")
                print(f"  Input:  {inp}")
                print(f"  Output: {out}")
                print(f"  Direction: {direction}")
                print(f"  Time: {timestamp}")
            
            # By direction
            print(f"\n📈 Translations by Direction:")
            print(f"-" * 60)
            cursor.execute("""
                SELECT direction, COUNT(*) as count 
                FROM translations 
                GROUP BY direction
            """)
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]} translations")
        
        # Model metadata
        print(f"\n🤖 Model Information:")
        print(f"-" * 60)
        cursor.execute("SELECT * FROM model_metadata ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            print(f"  Model: {row[1]}")
            print(f"  Path: {row[2]}")
            print(f"  BLEU Score: {row[3]}")
            print(f"  Exact Match Rate: {row[4]}%")
            print(f"  Character Accuracy: {row[5]}%")
            print(f"  Training Date: {row[6]}")
            print(f"  Epochs: {row[7]}")
            print(f"  Dataset Size: {row[8]}")
        
        conn.close()
        print(f"\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    view_database()
