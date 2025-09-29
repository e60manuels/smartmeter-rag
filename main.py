
import os
import glob
import ujson
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- Constants ---
DATA_DIR = "C:\\Users\\emanu\\Documenten\\GitHub\\P1-energie-dashboard\\sample_logs"
CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "smartmeter_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_data():
    """
    Loads smart meter data from .jsonl files, processes it into natural language sentences,
    and extracts metadata.
    """
    print(f"Loading data from: {DATA_DIR}")
    jsonl_files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
    
    if not jsonl_files:
        print("No .jsonl files found in the specified directory.")
        return [], []

    documents = []
    metadatas = []
    ids = []
    
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = ujson.loads(line)
                    ts = pd.to_datetime(record.get("timestamp"))
                    data = record.get("data", {})

                    # Extract relevant data points
                    power_import = data.get("total_power_import_kwh")
                    power_export = data.get("total_power_export_kwh")
                    gas_m3 = data.get("total_gas_m3")

                    # Create a natural language sentence
                    doc_parts = [f"On {ts.strftime('%Y-%m-%d at %H:%M')}"]
                    if power_import is not None:
                        doc_parts.append(f"the total power import was {power_import:.3f} kWh")
                    if power_export is not None:
                        doc_parts.append(f"the total power export was {power_export:.3f} kWh")
                    if gas_m3 is not None:
                        doc_parts.append(f"and the total gas consumption was {gas_m3:.3f} m3")
                    
                    document = ", ".join(doc_parts) + "."
                    
                    # Create metadata
                    metadata = {
                        "timestamp": str(ts),
                        "source_file": os.path.basename(file_path)
                    }
                    if power_import is not None:
                        metadata["power_import_kwh"] = power_import
                    if power_export is not None:
                        metadata["power_export_kwh"] = power_export
                    if gas_m3 is not None:
                        metadata["gas_m3"] = gas_m3

                    documents.append(document)
                    metadatas.append(metadata)
                    ids.append(f"rec_{{ts.strftime('%Y%m%d%H%M%S')}}_{len(ids)}")

                except (ValueError, KeyError) as e:
                    print(f"Skipping malformed line in {os.path.basename(file_path)}: {e}")

    print(f"Loaded {len(documents)} records.")
    return documents, metadatas, ids

def setup_chroma_db():
    """
    Initializes the ChromaDB client and creates/gets the collection.
    """
    print("Setting up ChromaDB...")
    # 1. Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 2. Create an embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # 3. Get or create the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"} # Specify the distance metric
    )
    print(f"Collection '{COLLECTION_NAME}' is ready.")
    return collection

def main():
    """
    Main function to load data, setup ChromaDB, and index the data.
    """
    # Step 1: Load and process data
    documents, metadatas, ids = load_data()

    if not documents:
        print("No documents to process. Exiting.")
        return

    # Step 2: Setup ChromaDB
    collection = setup_chroma_db()

    # Step 3: Add data to the collection in batches
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        print(f"Adding batch {i//batch_size + 1} of {len(documents)//batch_size + 1} to ChromaDB...")
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )

    print("\n--- Indexing Complete ---")
    print(f"Total documents indexed: {collection.count()}")
    print(f"ChromaDB database stored at: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    main()
