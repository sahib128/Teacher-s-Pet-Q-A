# workflow.py

import os
import sqlite3
import json
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time
from db_setup import chunk_text
# ---- STEP 1: PDF PROCESSING ----
from extract_text import extract_text_from_pdf
from db_setup import extract_and_store_chunks

import hashlib

DATABASE = 'embeddings_metadata.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def generate_document_hash(document_name, content):
    """Generate a unique hash for the document based on name and content."""
    hasher = hashlib.sha256()
    hasher.update(document_name.encode('utf-8'))
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()

def process_pdf(pdf_path, db_conn, document_id):
    """Extract text from PDF, chunk it, and store it in the database with document_id."""
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file '{pdf_path}' not found.")
        return
    
    print("📄 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
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


# ---- STEP 2: LOAD LANGUAGE MODEL ----
def load_model(model_name: str):
    """Load the language model."""
    print(f"🤖 Loading model: {model_name}")
    return OllamaLLM(model=model_name)


# ---- STEP 3: RETRIEVE AND RANK CHUNKS ----
def fetch_all_chunks(db_conn):
    """Fetch all chunks from the database."""
    c = db_conn.cursor()
    c.execute("SELECT chunk FROM embeddings")
    results = c.fetchall()
    return [row[0] for row in results]


def rank_chunks_by_similarity(query_text, chunks, top_k=5):
    """Rank chunks by textual similarity using TF-IDF and cosine similarity."""
    if not chunks:
        return []

    # Combine the query and chunks into one list
    texts = [query_text] + chunks

    # Convert texts into TF-IDF matrix
    vectorizer = TfidfVectorizer().fit_transform(texts)

    # Compute cosine similarity between the query and all chunks
    cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    
    # Rank chunks by similarity score
    ranked_indices = cosine_similarities.argsort()[-top_k:][::-1]
    ranked_chunks = [chunks[i] for i in ranked_indices]
    
    return ranked_chunks


# ---- STEP 4: HANDLE PROMPT WITH MODEL (Streaming Enabled) ----
def handle_prompt(query_text: str, context_text: str, model, temperature: float, top_p: float, max_length: int):
    """Handle the query and stream the model response word-by-word."""
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("📝 Generating response...\n")
    
    # Stream response from the model
    for part in model.stream(prompt, temperature=temperature, top_p=top_p, max_length=max_length):
        # Print each part (usually a word or sentence fragment) with streaming effect
        sys.stdout.write(part)
        sys.stdout.flush()
        time.sleep(0.05)  # Add a slight delay for better streaming simulation
    
    print("\n✅ Response complete.")


# ---- STEP 5: MAIN WORKFLOW ----
def main():
    # --- PDF PROCESSING ---
    con= get_db_connection()
    pdf_path = input("📂 Enter the path to your PDF file: ").strip()
    process_pdf(pdf_path,con)
    
    # --- Initialize Database ---
    db_conn = sqlite3.connect('embeddings_metadata.db')
    
    # --- User Query ---
    query_text = input("\n💬 Enter your question: ").strip()
    
    # --- Fetch and Rank Chunks ---
    print("🔍 Retrieving all chunks from the database...")
    chunks = fetch_all_chunks(db_conn)
    ranked_chunks = rank_chunks_by_similarity(query_text, chunks, top_k=5)
    
    if not ranked_chunks:
        print("⚠️ No relevant chunks found. Please refine your question.")
        return
    
    context_text = " ".join(ranked_chunks)
    
    # --- Load Model ---
    model_name = "llama3.1"
    model = load_model(model_name)
    
    # --- Generate Response ---
    temperature = 0.7
    top_p = 0.9
    max_length = 300
    handle_prompt(query_text, context_text, model, temperature, top_p, max_length)
    
    # --- Cleanup ---
    db_conn.close()
    print("\n✅ Workflow complete. Goodbye!")


if __name__ == "__main__":
    main()
