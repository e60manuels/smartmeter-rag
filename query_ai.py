import argparse
import chromadb
import re
from datetime import datetime

# --- Configuratie ---
CHROMA_PATH = "C:\\Users\\emanu\\Documenten\\GitHub\\smartmeter-rag\\chroma_db"
COLLECTION_NAME = "smartmeter_data"

# --- Database Manager ---
class ChromaManager:
    """Beheert de connectie en queries naar ChromaDB."""
    def __init__(self):
        print("Verbinding maken met ChromaDB...")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        print("Verbinding succesvol.")

    def get_analytical_answer(self, level, year, sort_by, order, limit):
        """Voert een directe metadata query uit."""
        where_conditions = []
        if level:
            where_conditions.append({"level": {"$eq": level}})
        if year:
            where_conditions.append({"year": {"$eq": year}})

        if not where_conditions:
            results = self.collection.get()
        elif len(where_conditions) == 1:
            results = self.collection.get(where=where_conditions[0])
        else:
            results = self.collection.get(where={"$and": where_conditions})

        if not results or not results['ids']:
            return None

        # Combineer de metadata met de IDs en sorteer
        items = list(zip(results['ids'], results['metadatas']))
        
        reverse_order = (order == 'desc')
        sorted_items = sorted(items, key=lambda item: item[1].get(sort_by, 0), reverse=reverse_order)
        
        return sorted_items[:limit]

# --- AI Query Parser ---
class QueryParser:
    """Vertaalt menselijke taal naar een gestructureerd query-plan."""
    def __init__(self):
        self.feature_map = {
            'teruglevering': 'total_export_kwh',
            'export': 'total_export_kwh',
            'verbruik': 'total_import_kwh',
            'import': 'total_import_kwh'
        }

    def parse(self, query):
        query = query.lower()

        # Zoek naar de verschillende entiteiten, ongeacht de volgorde
        level_match = re.search(r'(week|weken|maand|maanden|dag|dagen)', query)
        qualifier_match = re.search(r'(top|hoogste|meeste|laagste|minste)', query)
        feature_match = re.search(r'(teruglevering|export|verbruik|import)', query)
        year_match = re.search(r'(20\d{2})', query)
        limit_match = re.search(r'top\s*(\d+)', query)

        # Controleer of de essentiële onderdelen aanwezig zijn voor een analytische vraag
        if qualifier_match and feature_match and level_match:
            level = self._normalize_level(level_match.group(1))
            sort_by = self.feature_map.get(feature_match.group(1))
            order = 'desc' if qualifier_match.group(1) in ['top', 'hoogste', 'meeste'] else 'asc'
            limit = int(limit_match.group(1)) if limit_match else 1
            year = int(year_match.group(1)) if year_match else None

            if not sort_by:
                return None

            return {
                'intent': 'analytical',
                'params': {
                    'level': level,
                    'year': year,
                    'sort_by': sort_by,
                    'order': order,
                    'limit': limit
                }
            }
        
        # Voeg hier later logica toe voor similariteit-vragen
        return None

    def _normalize_level(self, level_str):
        if level_str.startswith('week'): return 'week'
        if level_str.startswith('dag'): return 'day'
        if level_str.startswith('maand'): return 'month'
        return 'week' # fallback

# --- Hoofdfunctie ---
def main():
    parser = argparse.ArgumentParser(description='Een AI-assistent voor je slimme meter data.')
    parser.add_argument('query', type=str, help='Stel een vraag in natuurlijke taal.')
    args = parser.parse_args()

    try:
        db_manager = ChromaManager()
        query_parser = QueryParser()
    except Exception as e:
        print(f"Fout bij initialisatie: {e}")
        return

    plan = query_parser.parse(args.query)

    if plan and plan['intent'] == 'analytical':
        print(f"Analytische vraag herkend: {plan['params']}")
        results = db_manager.get_analytical_answer(**plan['params'])
        
        print("\n--- ANTWOORD ---")
        if not results:
            print("Geen resultaten gevonden die aan de criteria voldoen.")
        else:
            for i, (item_id, metadata) in enumerate(results):
                value = metadata[plan['params']['sort_by']]
                print(f"{i+1}. ID: {item_id:<15} | {plan['params']['sort_by']}: {value:.2f} kWh")
        print("----------------")
    else:
        print("Ik begrijp deze vraag nog niet. Probeer een analytische vraag zoals 'hoogste week qua teruglevering in 2025'.")

if __name__ == "__main__":
    main()
