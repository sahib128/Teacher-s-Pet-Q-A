import sqlite3
import json
import nltk

nltk.download('punkt')


def init_db():
    """Initialize SQLite database with required tables."""
    try:
        print("🔄 Initializing database...")
        conn = sqlite3.connect('embeddings_metadata.db')
        c = conn.cursor()
        
        # Create documents table
        c.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                document_hash TEXT UNIQUE,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create chunks table
        c.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
        ''')
        
        # Create embeddings table
        c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk TEXT,
                section TEXT,
                page INTEGER,
                tables TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
        ''')
        
        conn.commit()
        print("✅ Database initialized successfully.")
        return conn
    
    except sqlite3.Error as e:
        print(f"❌ SQLite error during initialization: {e}")
        return None
    
    finally:
        conn.close()


def insert_metadata(db_conn, chunk, section, page, tables, document_id):
    """Insert chunk metadata into SQLite database linked to a document."""
    try:
        if db_conn is None:
            raise ConnectionError("❌ Database connection is not established.")
        
        c = db_conn.cursor()
        c.execute('''INSERT INTO embeddings 
                     (document_id, chunk, section, page, tables) 
                     VALUES (?, ?, ?, ?, ?)''', 
                  (document_id, chunk, section, page, json.dumps(tables)))
        db_conn.commit()
    except sqlite3.Error as e:
        print(f"❌ SQLite error during metadata insertion: {e}")
    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")


def extract_and_store_chunks(text, db_conn, document_id, section="General", page=1):
    """Chunk text and store it in the database, linked to a specific document."""
    if db_conn is None:
        raise ConnectionError("❌ Failed to initialize database connection during chunk extraction.")
    
    chunks = chunk_text(text, max_tokens=500, overlap=50)
    
    cursor = db_conn.cursor()
    for chunk in chunks:
        cursor.execute(
            "INSERT INTO chunks (document_id, chunk) VALUES (?, ?)",
            (document_id, chunk)
        )
    db_conn.commit()
    print("✅ Chunks successfully stored in the database.")


def chunk_text(text, max_tokens=700, overlap=100):
    """Split text into chunks with some overlap."""
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = sum(len(chunk.split()) for chunk in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


if __name__ == '__main__':
    init_db()
