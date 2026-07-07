import sqlite3
import datetime

DB_FILE = "chat_history.db"

def init_db():
    """Initializes the database and creates the history table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            base_answer TEXT NOT NULL,
            rag_answer TEXT NOT NULL,
            sources TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_interaction(question, base_answer, rag_answer, sources):
    """Saves a single Q&A interaction to the database."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources_str = ", ".join(sources) if sources else "None"
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (timestamp, question, base_answer, rag_answer, sources)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, question, base_answer, rag_answer, sources_str))
    conn.commit()
    conn.close()

# Initialize the database when the module is imported
init_db()
