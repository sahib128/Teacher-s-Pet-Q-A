import os
import sqlite3
import hashlib
import sys
import time
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from extract_text import extract_text_from_file
from db_setup import chunk_text

DATABASE = 'embeddings_metadata.db'


# ---- Utility Functions ----
def get_db_connection():
    """Establish a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def generate_document_hash(document_name, content):
    """Generate a unique hash for the document based on its name and content."""
    hasher = hashlib.sha256()
    hasher.update(document_name.encode('utf-8'))
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()


def process_pdf(pdf_path, db_conn, document_id):
    """Extract, chunk, and store text from a PDF."""
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file '{pdf_path}' not found.")
        return
    
    print("📄 Extracting text from PDF...")
    extracted_text = extract_text_from_file(pdf_path)
    
    print("🔄 Chunking text and storing in database...")
    chunks = chunk_text(extracted_text, max_tokens=500, overlap=50)
    
    cursor = db_conn.cursor()
    for chunk in chunks:
        cursor.execute(
            "INSERT INTO chunks (document_id, chunk) VALUES (?, ?)",
            (document_id, chunk)
        )
    db_conn.commit()
    print("✅ PDF processing complete: Text extracted, chunked, and stored in DB.")


# ---- Load Language Model ----
def load_model(model_name: str):
    """Load the Ollama language model."""
    print(f"🤖 Loading model: {model_name}")
    return OllamaLLM(model=model_name)


# ---- Fetch All Chunks ----
def fetch_all_chunks(db_conn, document_id):
    """Retrieve all chunks from the database for a specific document."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT chunk FROM chunks WHERE document_id = ?", (document_id,))
    return [row[0] for row in cursor.fetchall()]


# ---- Summary Prompts ----
def short_summary_prompt(context, complexity):
    return f"""
    Provide a short summary of the following context:
    Complexity: {complexity}
    Context:
    {context}
    Ensure the summary is concise, well-structured, and approximately 1/8th the length of the content.
    """


def long_summary_prompt(context, complexity):
    return f"""
    Provide a long summary of the following context:
    Complexity: {complexity}
    Context:
    {context}
    Ensure the summary captures all key details and is roughly 1/4th the length of the content.
    """


def abstractive_summary_prompt(context, length, complexity):
    return f"""
    Generate an abstractive summary of the following context:
    Length: {length} (If numeric, match exactly {length} lines)
    Complexity: {complexity}
    Context:
    {context}
    """


def extractive_summary_prompt(context, length, complexity):
    return f"""
    Generate an extractive summary of the following context:
    Length: {length} (If numeric, match exactly {length} lines)
    Complexity: {complexity}
    Context:
    {context}
    """


# ---- Final Summarization ----
def generate_summary(model, context, summary_type, length=None, complexity=None):
    """Generate the final summary using the specified type and options."""
    print("📝 Generating final summary with Ollama...")
    
    if summary_type == "short":
        prompt = short_summary_prompt(context, complexity)
    elif summary_type == "long":
        prompt = long_summary_prompt(context, complexity)
    elif summary_type == "abstractive":
        prompt = abstractive_summary_prompt(context, length, complexity)
    elif summary_type == "extractive":
        prompt = extractive_summary_prompt(context, length, complexity)
    else:
        print("❌ Invalid summary type.")
        return
    
    final_summary = ""
    for part in model.stream(prompt, temperature=0.7, top_p=0.9, max_length=2000):
        sys.stdout.write(part)
        sys.stdout.flush()
        final_summary += part
        time.sleep(0.05)
    
    print("\n✅ Final summary complete.")
    return final_summary


# ---- Main Workflow ----
def main():
    """Main workflow for PDF summarization."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Step 1: File Upload and Processing
    pdf_path = input("📂 Enter the path to your PDF file: ").strip()
    extracted_text = extract_text_from_file(pdf_path)
    document_hash = generate_document_hash(os.path.basename(pdf_path), extracted_text)
    
    cursor.execute("SELECT id FROM documents WHERE document_hash = ?", (document_hash,))
    existing_document = cursor.fetchone()
    
    if existing_document:
        document_id = existing_document['id']
        print("✅ Document already exists in the database.")
    else:
        print("📄 Processing new document...")
        cursor.execute(
            "INSERT INTO documents (name, document_hash) VALUES (?, ?)",
            (os.path.basename(pdf_path), document_hash)
        )
        document_id = cursor.lastrowid
        process_pdf(pdf_path, conn, document_id)
    
    # Step 2: Fetch All Chunks and Combine
    print("🔄 Retrieving all chunks from the database...")
    chunks = fetch_all_chunks(conn, document_id)
    context = " ".join(chunks)
    print(f"✅ Retrieved {len(chunks)} chunks. Combined into a single context.")
    
    # Step 3: Generate Final Summary with Ollama
    model = load_model("llama3.1")
    summary_type = input("📊 Choose summary type (short/long/extractive/abstractive): ").strip().lower()
    complexity = input("🎨 Choose complexity (simple/technical): ").strip().lower()
    length = input("🔢 Enter length (if applicable, e.g., 5 lines, short, medium): ").strip()
    
    final_summary = generate_summary(model, context, summary_type, length, complexity)
    print("\n📊 Final Summary:\n", final_summary)
    
    conn.close()


if __name__ == "__main__":
    main()
