"""
Quick test: Single RAG query using your real ARGO data
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import requests
from groq import Groq
from src.database import DatabaseManager, search_similar_argo

def get_embedding_from_ollama(text: str) -> list:
    """Generate embedding for user query"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "embeddinggemma", "prompt": text},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json().get("embedding", [])
    except Exception as e:
        print(f"❌ Embedding error: {e}")
    return []

def test_single_query():
    """Test one query against your ARGO data"""
    
    query = "What temperature data do we have from the Indian Ocean?"
    print(f"🤖 Testing Query: '{query}'")
    print("=" * 60)
    
    # Generate embedding
    query_embedding = get_embedding_from_ollama(query)
    if len(query_embedding) != 768:
        print("❌ Failed to generate embedding")
        return
    
    print("✅ Generated query embedding")
    
    # Search your data
    db = DatabaseManager()
    similar_profiles = search_similar_argo(query_embedding, limit=3, db=db)
    
    print(f"✅ Found {len(similar_profiles)} profiles from your ARGO data:")
    
    for i, profile in enumerate(similar_profiles, 1):
        print(f"\n📊 Profile {i}:")
        print(f"   Float ID: {profile['float_id']}")
        print(f"   Location: ({profile['latitude']:.1f}°, {profile['longitude']:.1f}°)")
        print(f"   Region: {profile['region']}")
        print(f"   Date: {profile['profile_date']}")
        print(f"   Similarity: {profile['similarity']:.3f}")
        print(f"   Summary: {profile['summary']}")
        
        if profile.get('temperature_data'):
            temp = profile['temperature_data']
            print(f"   Temperature: {temp.get('min', 'N/A')}°C to {temp.get('max', 'N/A')}°C")
        
        if profile.get('salinity_data'):
            sal = profile['salinity_data']
            print(f"   Salinity: {sal.get('min', 'N/A')} to {sal.get('max', 'N/A')}")
    
    # Generate LLM answer
    print("\n🤖 Generating LLM answer...")
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API"))
        
        # Create context
        context = "Available ARGO data:\n"
        for profile in similar_profiles:
            context += f"- Float {profile['float_id']} at ({profile['latitude']:.1f}°, {profile['longitude']:.1f}°) in {profile['region']}\n"
            context += f"  {profile['summary']}\n"
            
            if profile.get('temperature_data'):
                temp = profile['temperature_data']
                context += f"  Temperature: {temp.get('min')}°C to {temp.get('max')}°C\n"
        
        prompt = f"""Answer this question using the provided ARGO oceanographic data:

Question: {query}

Data:
{context}

Answer based only on the provided data:"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=300
        )
        
        answer = response.choices[0].message.content
        print("\n🎯 LLM Answer:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ LLM error: {e}")

if __name__ == "__main__":
    test_single_query()