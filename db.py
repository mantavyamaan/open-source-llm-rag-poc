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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
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

def save_document(filename, content):
    """Saves or updates a document in the database."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (filename, content, timestamp)
        VALUES (?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            content=excluded.content,
            timestamp=excluded.timestamp
    ''', (filename, content, timestamp))
    conn.commit()
    conn.close()

def get_all_documents():
    """Retrieves all documents from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, content FROM documents')
    rows = cursor.fetchall()
    conn.close()
    return [{"filename": row[0], "content": row[1]} for row in rows]

def delete_document(filename):
    """Deletes a document from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM documents WHERE filename = ?', (filename,))
    conn.commit()
    conn.close()

# Initialize the database when the module is imported
init_db()
