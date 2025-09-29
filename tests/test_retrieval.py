
import os
import chromadb
from chromadb.utils import embedding_functions

# --- Constants ---
# Ensure these constants match the ones used in main.py
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
COLLECTION_NAME = "smartmeter_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def test_retrieval():
    """
    Tests the retrieval of documents from the ChromaDB based on a sample query.
    """
    print("--- Starting Retrieval Test ---")
    
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

    # 3. Define a sample query
    # Using a specific date that exists in the sample data for a targeted query
    query_text = "Wat was het totale stroomverbruik en gasverbruik op 15 augustus 2025?"
    print(f"\nQuery: \"{query_text}\"\n")

    # 4. Query the collection
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=3  # Get the top 3 most relevant documents
        )
    except Exception as e:
        print(f"Error during query: {e}")
        return

    # 5. Print the results
    print("\n--- Top 3 Retrieved Documents ---")
    if not results or not results.get('documents'):
        print("No results found.")
        return

    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}:")
        print(f"  Document: {doc}")
        # Also print metadata and distance if available
        if results.get('metadatas') and results['metadatas'][0][i]:
            print(f"  Metadata: {results['metadatas'][0][i]}")
        if results.get('distances') and results['distances'][0][i]:
            print(f"  Distance: {results['distances'][0][i]:.4f}")

    print("\n--- Retrieval Test Complete ---")

if __name__ == "__main__":
    test_retrieval()
