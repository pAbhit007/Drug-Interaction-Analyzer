# rag_pipeline.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import json
import os
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models and databases
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./drug_db")
collection = chroma_client.get_or_create_collection(
    name="drug_interactions",
    metadata={"hnsw:space": "cosine"}
)

# RxNorm API endpoints
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

def get_rxcui(drug_name: str) -> str:
    """Get RxNorm CUI for a drug name"""
    try:
        response = requests.get(f"{RXNORM_BASE_URL}/rxcui.json?name={drug_name}")
        data = response.json()
        if 'idGroup' in data and 'rxnormId' in data['idGroup']:
            return data['idGroup']['rxnormId'][0]
        return None
    except Exception as e:
        logger.error(f"Error getting RxCUI for {drug_name}: {str(e)}")
        return None

def get_drug_interactions_from_rxnorm(drug1: str, drug2: str) -> Dict[str, Any]:
    """Query RxNorm API for drug interactions"""
    try:
        # Get RxCUIs for both drugs
        rxcui1 = get_rxcui(drug1)
        rxcui2 = get_rxcui(drug2)
        
        if not rxcui1 or not rxcui2:
            return {
                "has_interaction": False,
                "method": "RxNorm API",
                "response": f"Could not find RxNorm identifiers for one or both drugs: {drug1}, {drug2}"
            }
        
        # Query for interactions
        url = f"{RXNORM_BASE_URL}/interaction/list.json?rxcuis={rxcui1}+{rxcui2}"
        response = requests.get(url)
        data = response.json()
        
        if 'fullInteractionTypeGroup' in data:
            interactions = data['fullInteractionTypeGroup'][0]['fullInteractionType']
            description = interactions[0]['interactionPair'][0]['description']
            return {
                "has_interaction": True,
                "method": "RxNorm API",
                "response": description
            }
        
        return {
            "has_interaction": False,
            "method": "RxNorm API",
            "response": f"No known interactions found between {drug1} and {drug2}"
        }
        
    except Exception as e:
        logger.error(f"Error querying RxNorm API: {str(e)}")
        return fallback_interaction_check(drug1, drug2)

def store_interaction_data(drug1: str, drug2: str, interaction_data: Dict[str, Any]):
    """Store interaction data in ChromaDB"""
    try:
        # Create embedding for the interaction text
        text = f"{drug1} and {drug2} interaction: {interaction_data['response']}"
        embedding = model.encode(text)
        
        # Store in ChromaDB
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[{
                "drug1": drug1,
                "drug2": drug2,
                "has_interaction": interaction_data["has_interaction"],
                "method": interaction_data["method"]
            }],
            ids=[f"{drug1}_{drug2}_{hash(text)}"]
        )
    except Exception as e:
        logger.error(f"Error storing interaction data: {str(e)}")

def query_local_database(drug1: str, drug2: str) -> Dict[str, Any]:
    """Query ChromaDB for existing interaction data"""
    try:
        query = f"{drug1} and {drug2} interaction"
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if results and results['documents'][0]:
            metadata = results['metadatas'][0][0]
            return {
                "has_interaction": metadata["has_interaction"],
                "method": "Local Database",
                "response": results['documents'][0][0]
            }
        return None
    except Exception as e:
        logger.error(f"Error querying local database: {str(e)}")
        return None

def fallback_interaction_check(drug1: str, drug2: str) -> Dict[str, Any]:
    """Fallback method using a conservative approach"""
    return {
        "has_interaction": True,
        "method": "Conservative Fallback",
        "response": (
            f"Unable to verify interactions between {drug1} and {drug2}. "
            "As a precautionary measure, please consult a healthcare professional "
            "before combining these medications."
        )
    }

def get_drug_interaction(drug1: str, drug2: str) -> Dict[str, Any]:
    """Main function to check drug interactions"""
    try:
        # First, check local database
        local_result = query_local_database(drug1, drug2)
        if local_result:
            return local_result
        
        # If not in local database, query RxNorm
        result = get_drug_interactions_from_rxnorm(drug1, drug2)
        
        # Store the result in local database
        store_interaction_data(drug1, drug2, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_drug_interaction: {str(e)}")
        return fallback_interaction_check(drug1, drug2)

# Initialize database with some common interactions if empty
def initialize_database():
    """Initialize the database with some common drug interactions"""
    common_interactions = [
        ("aspirin", "warfarin", {
            "has_interaction": True,
            "method": "Initial Data",
            "response": "Combining aspirin and warfarin increases the risk of bleeding."
        }),
        ("ibuprofen", "naproxen", {
            "has_interaction": True,
            "method": "Initial Data",
            "response": "Taking ibuprofen with naproxen increases the risk of gastrointestinal bleeding."
        })
    ]
    
    try:
        for drug1, drug2, interaction in common_interactions:
            if not query_local_database(drug1, drug2):
                store_interaction_data(drug1, drug2, interaction)
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Initialize database when module is loaded
initialize_database()