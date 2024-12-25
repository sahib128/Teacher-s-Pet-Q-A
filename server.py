from flask import Flask, request, jsonify
from main_call import process_pdf, fetch_all_chunks, load_model, handle_prompt, rank_chunks_by_similarity,extract_text_from_pdf,extract_and_store_chunks,generate_document_hash
import sqlite3
import hashlib
import os

app = Flask(__name__)
DATABASE = 'embeddings_metadata.db'


# --- Utility Functions ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def file_hash(file_path):
    """Generate SHA256 hash for a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# --- Unified Endpoint ---
@app.route('/process-and-ask', methods=['POST'])
def process_and_ask():
    pdf_file = request.files['file']
    query = request.form.get('question')
    
    if not query:
        return jsonify({"error": "Missing 'question' parameter."}), 400
    
    # Save PDF Temporarily
    os.makedirs('uploads', exist_ok=True)
    pdf_path = f"uploads/{pdf_file.filename}"
    pdf_file.save(pdf_path)
    
    # Extract text for hashing
    extracted_text = extract_text_from_pdf(pdf_path)
    document_hash = generate_document_hash(pdf_file.filename, extracted_text)
    
    # Database Connection
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if the document already exists
    cursor.execute("SELECT id FROM documents WHERE document_hash = ?", (document_hash,))
    existing_document = cursor.fetchone()
    
    if existing_document:
        document_id = existing_document['id']
        print("✅ Document already exists. Fetching chunks directly from the database.")
    else:
        print("📄 Processing new document...")
        
        # Insert document metadata
        cursor.execute(
            "INSERT INTO documents (name, document_hash) VALUES (?, ?)",
            (pdf_file.filename, document_hash)
        )
        document_id = cursor.lastrowid
        
        # Process and store chunks
        process_pdf(pdf_path, conn, document_id)
        conn.commit()
    
    # Fetch chunks related to this document
    cursor.execute("SELECT chunk FROM chunks WHERE document_id = ?", (document_id,))
    chunks = [row['chunk'] for row in cursor.fetchall()]
    
    if not chunks:
        conn.close()
        return jsonify({"error": "No chunks found for this document."}), 404
    
    # Rank chunks and generate a response
    ranked_chunks = rank_chunks_by_similarity(query, chunks, top_k=5)
    context_text = " ".join(ranked_chunks)
    
    model = load_model("llama3.1")
    response = handle_prompt(query, context_text, model, 0.7, 0.9, 300)
    
    conn.close()
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
