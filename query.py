import os
import argparse
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import re
from datetime import datetime

# --- Constants ---
CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "smartmeter_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

def get_chroma_collection():
    """Initializes ChromaDB client and returns the collection."""
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"ChromaDB database not found at '{CHROMA_DB_PATH}'. "
            "Please run main.py to create it."
        )
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=embedding_function
    )
    return collection

def extract_date_from_query(query_text: str) -> str | None:
    """
    Extracts a date in YYYY-MM-DD format from the query text.
    Supports patterns like 'DD maand YYYY', 'DD-MM-YYYY', 'YYYY-MM-DD'.
    """
    # Pattern for 'DD maand YYYY' (e.g., '1 september 2025')
    month_map = {
        'januari': '01', 'februari': '02', 'maart': '03', 'april': '04',
        'mei': '05', 'juni': '06', 'juli': '07', 'augustus': '08',
        'september': '09', 'oktober': '10', 'november': '11', 'december': '12'
    }
    pattern_dd_month_yyyy = r'\b(\d{1,2})\s+(' + '|'.join(month_map.keys()) + r')\s+(\d{4})\b'
    match = re.search(pattern_dd_month_yyyy, query_text, re.IGNORECASE)
    if match:
        day, month_str, year = match.groups()
        month_num = month_map[month_str.lower()]
        return f"{year}-{month_num}-{int(day):02d}"

    # Pattern for 'DD-MM-YYYY' or 'DD/MM/YYYY'
    pattern_dd_mm_yyyy = r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b'
    match = re.search(pattern_dd_mm_yyyy, query_text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # Pattern for 'YYYY-MM-DD'
    pattern_yyyy_mm_dd = r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b'
    match = re.search(pattern_yyyy_mm_dd, query_text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return None

def query_llm(query_text: str, n_results: int = 5):
    """
    Queries ChromaDB for relevant context and then queries the LLM with that context.
    """
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable not set. Please set it to your NVIDIA API key.")

    # 1. Retrieve context from ChromaDB
    print("1. Retrieving relevant documents from ChromaDB...")
    collection = get_chroma_collection()
    
    where_clause = {}
    date_filter = extract_date_from_query(query_text)
    if date_filter:
        where_clause["date"] = date_filter
        print(f"   Applying date filter: {date_filter}")

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_clause if where_clause else None
    )
    
    context_documents = results.get('documents', [[]])[0]
    if not context_documents:
        return "I could not find any relevant information in the database to answer your question.", []

    print(f"   Found {len(context_documents)} documents.")

    # 2. Prepare the prompt for the LLM
    context = "\n".join([f"- {doc}" for doc in context_documents])
    prompt = f"""You are an expert assistant for answering questions about smart meter energy data. Please answer the user's question based *only* on the context provided below. If the context does not contain the answer, say that you cannot answer the question with the given information. Do not make up any information.

---
CONTEXT ---
{context}

--- QUESTION ---
{query_text}

--- ANSWER ---"""

    # 3. Query the NVIDIA API
    print("2. Sending query and context to the NVIDIA LLM...")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    
    answer = completion.choices[0].message.content
    print("3. Received answer from LLM.")
    return answer, context_documents

def main():
    """
    Main function to handle command-line arguments and run the query.
    """
    parser = argparse.ArgumentParser(description="Query the smart meter RAG system.")
    parser.add_argument("query_text", type=str, help="The question you want to ask about your smart meter data.")
    parser.add_argument("--n_results", type=int, default=5, help="The number of documents to retrieve from the database.")
    args = parser.parse_args()

    try:
        answer, context = query_llm(args.query_text, args.n_results)
        print("\n----------------------------------------")
        print(f"Question: {args.query_text}")
        print(f"\nAnswer: {answer}")
        print("\n----------------------------------------")
        print("\nRetrieved context used for this answer:")
        for i, doc in enumerate(context):
            print(f"  [{i+1}] {doc}")
        print("----------------------------------------")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
