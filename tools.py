import argparse
import chromadb
import pandas as pd
from datetime import datetime, time
from typing import Literal, Dict, Any, Optional

# --- Constants ---
CHROMA_DB_PATH = "C:\\Users\\emanu\\Documenten\\GitHub\\smartmeter-rag\\chroma_db"
COLLECTION_NAME = "smartmeter_master"

# --- Type definitions for clarity ---
Metric = Literal['active_power_w', 'total_power_import_kwh', 'total_power_export_kwh']
Aggregation = Literal['SUM', 'AVG', 'MAX', 'MIN', 'DELTA']
TimeOfDay = Literal['ochtend', 'middag', 'avond', 'nacht']
ValueType = Literal['ALL', 'CONSUMPTION', 'PRODUCTION']

TIME_OF_DAY_MAPPING: Dict[TimeOfDay, tuple[time, time]] = {
    'nacht': (time(0, 0), time(5, 59, 59)),
    'ochtend': (time(6, 0), time(11, 59, 59)),
    'middag': (time(12, 0), time(17, 59, 59)),
    'avond': (time(18, 0), time(23, 59, 59)),
}

def query_aggregator(
    metric: Metric,
    aggregation: Aggregation,
    start_date: str,
    end_date: str,
    time_of_day: Optional[TimeOfDay] = None,
    value_type: ValueType = 'ALL'
) -> Dict[str, Any]:
    """
    Queries ChromaDB, filters data based on value type (consumption/production),
    and performs a specified aggregation.
    """
    try:
        # 1. Connect to ChromaDB and get data
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        start_epoch, end_epoch = int(start_dt.timestamp()), int(end_dt.timestamp())

        results = collection.get(
            where={
                "$and": [
                    {"timestamp": {"$gte": start_epoch}},
                    {"timestamp": {"$lte": end_epoch}}
            ]
            },
            include=["metadatas"],
            limit=10000
        )

        if not results or not results['metadatas']:
            return {"error": "No data found for the specified date range."}

        # 2. Load and prepare DataFrame
        df = pd.DataFrame(results['metadatas'])
        if df.empty:
            return {"error": "No data found for the specified date range."}

        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df.dropna(subset=[metric], inplace=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Europe/Amsterdam')

        # 3. Apply time_of_day filter
        if time_of_day:
            start_time, end_time = TIME_OF_DAY_MAPPING[time_of_day]
            df = df[df['datetime'].dt.time.between(start_time, end_time)]
            if df.empty:
                return {"error": f"No data for time_of_day='{time_of_day}' in date range."}

        # 4. Apply value_type filter (Consumption/Production)
        if metric == 'active_power_w': # Only apply consumption/production filter for active_power_w
            if value_type == 'CONSUMPTION':
                df = df[df[metric] > 0].copy()
            elif value_type == 'PRODUCTION':
                df = df[df[metric] < 0].copy()
        # For total_power_import_kwh and total_power_export_kwh, value_type 'CONSUMPTION' or 'PRODUCTION'
        # implies using the metric directly, as they are inherently consumption/production.
        # 'ALL' would also use the metric directly.

        if df.empty:
            return {"error": f"No data found for value_type='{value_type}' in the selected period."}

        # 5. Perform Aggregation
        if aggregation == 'DELTA': # For cumulative kWh metrics
            if not metric.startswith('total'):
                return {"error": "DELTA aggregation is only for cumulative metrics like total_power_import_kwh."}
            result_value = df[metric].max() - df[metric].min()
        
        elif aggregation == 'SUM': # For active_power_w, sum is not meaningful in kWh
            if metric != 'active_power_w':
                 return {"error": "SUM aggregation is only meaningful for active_power_w in this context."}
            result_value = df[metric].sum()

        elif aggregation == 'AVG':
            result_value = df[metric].mean()
        
        elif aggregation == 'MAX':
            result_row = df.loc[df[metric].idxmax()]
            result_value = result_row[metric]
            return {
                "aggregation_type": aggregation, "metric": metric, "value_type": value_type,
                "value": float(result_value), "timestamp": result_row['datetime'].isoformat()
            }
        
        elif aggregation == 'MIN':
            result_row = df.loc[df[metric].idxmin()]
            result_value = result_row[metric]
            return {
                "aggregation_type": aggregation, "metric": metric, "value_type": value_type,
                "value": float(result_value), "timestamp": result_row['datetime'].isoformat()
            }
        else:
            return {"error": f"Invalid aggregation type: {aggregation}"}

        return {
            "aggregation_type": aggregation, "metric": metric, "value_type": value_type,
            "value": float(result_value)
        }

    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Universal Aggregator for Smart Meter Data.")
    parser.add_argument("--metric", required=True, type=str, choices=list(Metric.__args__))
    parser.add_argument("--aggregation", required=True, type=str, choices=list(Aggregation.__args__))
    parser.add_argument("--start_date", required=True, type=str)
    parser.add_argument("--end_date", required=True, type=str)
    parser.add_argument("--time_of_day", type=str, choices=list(TimeOfDay.__args__), default=None)
    parser.add_argument("--value_type", type=str, choices=list(ValueType.__args__), default='ALL')
    
    args = parser.parse_args()
    result = query_aggregator(**vars(args))
    print(result)

if __name__ == "__main__":
    main()