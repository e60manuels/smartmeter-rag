
import pandas as pd
import chromadb
import numpy as np

# --- Configuratie ---
CSV_PATH = "C:\\Users\\emanu\\Documenten\\GitHub\\smartmeter-rag\\overige\\P1metingen.csv"
CHROMA_PATH = "C:\\Users\\emanu\\Documenten\\GitHub\\smartmeter-rag\\chroma_db"
COLLECTION_NAME = "smartmeter_data"

def setup_database():
    """
    Leest de CSV, transformeert de data en migreert deze naar een nieuwe,
    schone ChromaDB collectie met uitgebreide metadata.
    """
    print("--- Stap 1: Database opzetten ---")
    print("Data inlezen en voorbereiden...")
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Bereken dagelijkse import/export
    df_day = df.resample('D').agg(
        daily_import=('total_power_import_kwh', lambda x: x.max() - x.min() if not x.empty else 0),
        daily_export=('total_power_export_kwh', lambda x: x.max() - x.min() if not x.empty else 0)
    ).dropna()

    # Bereken wekelijkse import/export
    df_week = df_day.resample('W').agg(
        weekly_import=('daily_import', 'sum'),
        weekly_export=('daily_export', 'sum')
    ).dropna()
    print("Data succesvol voorbereid.")

    # ChromaDB client initialiseren
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Oude collectie verwijderen voor een schone start
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Bestaande collectie '{COLLECTION_NAME}' verwijderd.")
    except Exception:
        pass # Collectie bestond niet, geen probleem.

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Nieuwe, schone collectie '{COLLECTION_NAME}' aangemaakt.")

    # Voeg week-data toe aan de collectie
    print("Wekelijkse data toevoegen aan de database...")
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for index, row in df_week.iterrows():
        week_id = f"week_{index.strftime('%Y-%U')}" # %U voor weeknummer (zondag als eerste dag)
        ids_to_add.append(week_id)
        
        # Lege documenten, we focussen op metadata
        documents_to_add.append(week_id)
        
        # Rijke metadata toevoegen
        metadatas_to_add.append({
            "level": "week",
            "year": index.year,
            "week_of_year": index.week,
            "total_import_kwh": row['weekly_import'],
            "total_export_kwh": row['weekly_export']
        })

    collection.add(
        documents=documents_to_add,
        metadatas=metadatas_to_add,
        ids=ids_to_add
    )
    print(f"{len(ids_to_add)} week-documenten succesvol toegevoegd aan de database.")
    return collection

def answer_export_question(collection):
    """
    Beantwoordt de specifieke vraag door direct de metadata in ChromaDB te bevragen.
    """
    print("\n--- Stap 2: Vraag beantwoorden ---")
    print("Zoeken naar de week in 2025 met de hoogste teruglevering...")

    # Haal alle weken uit 2025 op, inclusief hun metadata
    results = collection.get(
        where={"year": 2025}
    )

    if not results or not results['ids']:
        print("Geen data gevonden voor het jaar 2025.")
        return

    # Vind de week met de hoogste 'total_export_kwh' in de metadata
    max_export = -1
    best_week_id = None
    for metadata in results['metadatas']:
        if metadata.get('total_export_kwh', 0) > max_export:
            max_export = metadata['total_export_kwh']
            # We moeten de bijbehorende ID vinden. Dit is omslachtig zonder directe koppeling.
            # Laten we de metadata gebruiken om de ID te reconstrueren of te vinden.
            # Dit is een zwakte in de .get() API van Chroma. We lossen het op.
    
    # Efficiëntere manier: vind de ID direct.
    best_item = max(zip(results['ids'], results['metadatas']), key=lambda item: item[1].get('total_export_kwh', 0))
    
    best_week_id = best_item[0]
    max_export_value = best_item[1]['total_export_kwh']

    print("\n--- ANTWOORD ---")
    print(f"De week met de hoogste teruglevering in 2025 is: {best_week_id}")
    print(f"Totale teruglevering in die week: {max_export_value:.2f} kWh")
    print("----------------")


if __name__ == "__main__":
    # Stap 1: Zet de database op (of update deze)
    db_collection = setup_database()
    
    # Stap 2: Beantwoord de specifieke vraag
    answer_export_question(db_collection)
