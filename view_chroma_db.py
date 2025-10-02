import os
import chromadb
import json
import argparse

# --- Constants ---
CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "smartmeter_master"

def view_database(limit: int, offset: int):
    """Initializes ChromaDB client, retrieves a page of results, and prints it."""
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"ChromaDB database not found at '{CHROMA_DB_PATH}'. "
            "Please run main.py to create it."
        )
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME
    )
    
    total_count = collection.count()
    
    results = collection.get(limit=limit, offset=offset)
    
    print(f"Showing {len(results['ids'])} of {total_count} documents (offset: {offset}):")
    # Pretty print the JSON output
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View the ChromaDB database with pagination.")
    parser.add_argument("--limit", type=int, default=5, help="The number of documents to retrieve.")
    parser.add_argument("--offset", type=int, default=0, help="The starting offset for retrieving documents.")
    args = parser.parse_args()

    view_database(args.limit, args.offset)