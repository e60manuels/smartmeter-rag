import os
import json
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import ujson

# Pad naar de JSONL-bestanden
JSONL_PATH = "C:\\Users\\emanu\\Documenten\\GitHub\\P1-energie-dashboard\\sample_logs"
# Pad voor de persistente ChromaDB-opslag
CHROMA_PATH = "C:\\Users\\emanu\\Documenten\\GitHub\\smartmeter-rag\\chroma_db"
COLLECTION_NAME = "smartmeter_master"
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 4000

def generate_summary(record_data, dt_object):
    """Genereert een beschrijvende tekst voor een gegeven datarecord."""
    power = record_data.get("active_power_w", 0)
    hour = dt_object.hour

    if 0 <= hour < 6:
        time_of_day = "in de nacht"
    elif 6 <= hour < 12:
        time_of_day = "in de ochtend"
    elif 12 <= hour < 18:
        time_of_day = "in de middag"
    else:
        time_of_day = "in de avond"

    if power >= 0:
        action = "Verbruik"
        description = f"{action} van {power} watt {time_of_day}."
    else:
        action = "Teruglevering"
        description = f"{action} van {abs(power)} watt {time_of_day}."
    
    return description

def add_batch_to_chroma(collection, model, documents, metadatas, ids):
    """Genereert embeddings en voegt een batch data toe aan ChromaDB."""
    if not ids:
        return
    
    print(f"Batch van {len(ids)} wordt verwerkt. Genereren van embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)
    
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Batch succesvol toegevoegd.")

def migrate_data():
    """Voert de migratie van JSONL-data naar ChromaDB uit."""
    print("Starten van de datamigratie...")

    # 1. Chroma en model initialiseren
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    model = SentenceTransformer(MODEL_NAME)

    # Collectie opschonen voor een nieuwe, schone migratie
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Bestaande collectie '{COLLECTION_NAME}' verwijderd voor een schone start.")
    except Exception:
        print(f"Collectie '{COLLECTION_NAME}' bestond nog niet. Een nieuwe wordt aangemaakt.")

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"ChromaDB collectie '{COLLECTION_NAME}' succesvol aangemaakt.")

    documents = []
    metadatas = []
    ids = []
    
    # 2. JSONL-bestanden inlezen
    print(f"Inlezen van JSONL-bestanden uit: {JSONL_PATH}")
    filenames = [f for f in os.listdir(JSONL_PATH) if f.endswith(".jsonl")]
    
    if not filenames:
        print("Geen .jsonl bestanden gevonden in de map.")
        return

    total_records = 0
    for filename in sorted(filenames):
        filepath = os.path.join(JSONL_PATH, filename)
        with open(filepath, "r") as f:
            for line in f:
                try:
                    record = ujson.loads(line)
                    record_data = record.get("data", {})
                    
                    # 3. Data transformeren
                    record_id = record["timestamp"]
                    dt_object = datetime.fromisoformat(record_id.split(".")[0])
                    epoch_timestamp = int(dt_object.timestamp())
                    
                    metadata = {
                        "timestamp": epoch_timestamp,
                        "total_power_import_kwh": record_data.get("total_power_import_kwh", 0),
                        "total_power_export_kwh": record_data.get("total_power_export_kwh", 0),
                        "active_power_w": record_data.get("active_power_w", 0)
                    }
                    
                    document = generate_summary(record_data, dt_object)
                    
                    ids.append(record_id)
                    metadatas.append(metadata)
                    documents.append(document)
                    total_records += 1

                    # 5. Data in batches opslaan in ChromaDB
                    if len(ids) >= BATCH_SIZE:
                        add_batch_to_chroma(collection, model, documents, metadatas, ids)
                        documents, metadatas, ids = [], [], []

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Fout bij verwerken van regel in {filename}: {e}")
                    continue

    # Verwerk de laatste batch als die er is
    if ids:
        add_batch_to_chroma(collection, model, documents, metadatas, ids)

    print("\nMigratie voltooid!")
    print(f"Totaal {total_records} records verwerkt.")
    print(f"Totaal {collection.count()} documenten in de '{COLLECTION_NAME}' collectie.")

if __name__ == "__main__":
    migrate_data()