import os
import chromadb
from chromadb.utils import embedding_functions

# --- Constants ---
# Ensure these constants match the ones used in main.py
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
COLLECTION_NAME = "smartmeter_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def test_retrieval_date_filter_only():
    """
    Tests if documents for a specific date exist in ChromaDB using only metadata filtering.
    """
    print("--- Starting Retrieval Test (Date Filter Only) ---")
    
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: ChromaDB path not found at '{CHROMA_DB_PATH}'")
        print("Please run main.py first to create and populate the database.")
        return

    # 1. Initialize ChromaDB client and embedding function
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # 2. Get the existing collection
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"Successfully connected to collection '{COLLECTION_NAME}' with {collection.count()} documents.")
    except Exception as e:
        print(f"Error connecting to collection: {e}")
        return

    # 3. Define the date to filter for
    target_date = "2025-09-01"
    print(f"\nAttempting to retrieve documents for date: {target_date}")

    # 4. Query the collection using only the 'where' clause
    try:
        results = collection.query(
            query_texts=[""], # Empty query_text as we are only filtering by metadata
            n_results=100, # Get up to 100 results for this date
            where={"date": target_date}
        )
    except Exception as e:
        print(f"Error during query with date filter: {e}")
        return

    # 5. Print the results
    print("\n--- Retrieved Documents with Date Filter ---")
    context_documents = results.get('documents', [[]])[0]
    if not context_documents:
        print(f"No documents found for date {target_date} using metadata filter.")
        return

    print(f"Found {len(context_documents)} documents for {target_date}.")
    for i, doc in enumerate(context_documents):
        print(f"\nResult {i+1}:")
        print(f"  Document: {doc}")
        metadata = results['metadatas'][0][i]
        print(f"  Metadata: {metadata}")

    print("\n--- Retrieval Test (Date Filter Only) Complete ---")

if __name__ == "__main__":
    test_retrieval_date_filter_only()
