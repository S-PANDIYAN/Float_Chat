"""
Complete RAG Pipeline Test: User Query → Vector Search → LLM Answer
Uses ONLY your stored ARGO vector data (no demo data)
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

def answer_query_with_rag(user_query: str):
    """Complete RAG pipeline: Query → Vector Search → LLM Answer"""
    
    print(f"🤖 User Query: '{user_query}'")
    print("=" * 60)
    
    # Step 1: Generate embedding for user query
    print("🔄 Step 1: Generating query embedding...")
    query_embedding = get_embedding_from_ollama(user_query)
    
    if len(query_embedding) != 768:
        print(f"❌ Failed to generate query embedding")
        return
    
    print(f"✅ Generated 768-dim query embedding")
    
    # Step 2: Search similar profiles in vector database
    print("🔄 Step 2: Searching your stored ARGO data...")
    db = DatabaseManager()
    similar_profiles = search_similar_argo(query_embedding, limit=3, db=db)
    
    if not similar_profiles:
        print("❌ No similar profiles found in your data")
        return
    
    print(f"✅ Found {len(similar_profiles)} relevant profiles")
    
    # Step 3: Prepare context from your data
    context_data = []
    for profile in similar_profiles:
        context_data.append({
            'float_id': profile['float_id'],
            'location': f"({profile['latitude']:.1f}°, {profile['longitude']:.1f}°)",
            'region': profile['region'],
            'summary': profile['summary'],
            'similarity': profile['similarity'],
            'date': profile['profile_date'],
            'temperature': profile.get('temperature_data', {}),
            'salinity': profile.get('salinity_data', {})
        })
    
    print("📊 Retrieved context from your ARGO data:")
    for i, ctx in enumerate(context_data, 1):
        print(f"  {i}. {ctx['float_id']} - {ctx['region']} (similarity: {ctx['similarity']:.3f})")
        print(f"     {ctx['summary']}")
    
    # Step 4: Generate answer using Groq LLM
    print("🔄 Step 3: Generating answer with LLM...")
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API"))
        
        # Create context string from your ARGO data
        context_text = "Relevant ARGO oceanographic profiles from your data:\n\n"
        for i, ctx in enumerate(context_data, 1):
            context_text += f"Profile {i}:\n"
            context_text += f"- Float ID: {ctx['float_id']}\n"
            context_text += f"- Location: {ctx['location']} in {ctx['region']}\n"
            context_text += f"- Date: {ctx['date']}\n"
            context_text += f"- Summary: {ctx['summary']}\n"
            
            if ctx['temperature']:
                temp = ctx['temperature']
                context_text += f"- Temperature: {temp.get('min', 'N/A')}°C to {temp.get('max', 'N/A')}°C (mean: {temp.get('mean', 'N/A')}°C)\n"
            
            if ctx['salinity']:
                sal = ctx['salinity']
                context_text += f"- Salinity: {sal.get('min', 'N/A')} to {sal.get('max', 'N/A')} (mean: {sal.get('mean', 'N/A')})\n"
            
            context_text += f"- Similarity to query: {ctx['similarity']:.3f}\n\n"
        
        # Create prompt for LLM
        prompt = f"""You are an oceanographic data analyst. Answer the user's question using ONLY the provided ARGO profile data.

User Question: {user_query}

Available Data:
{context_text}

Instructions:
- Answer based ONLY on the provided ARGO profiles
- Be specific about locations, temperatures, and measurements
- Mention float IDs and regions when relevant
- If the data doesn't contain information to answer the question, say so clearly
- Keep the answer concise and factual

Answer:"""

        # Get response from Groq LLM
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=512
        )
        
        answer = chat_completion.choices[0].message.content
        
        print("🤖 LLM Answer based on your ARGO data:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
        return {
            'query': user_query,
            'answer': answer,
            'context': context_data,
            'source': 'Your stored ARGO NetCDF data'
        }
        
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return None

def test_rag_pipeline():
    """Test the complete RAG pipeline with sample queries"""
    
    print("🧪 Testing RAG Pipeline with Your ARGO Data")
    print("=" * 60)
    
    # Check database status
    db = DatabaseManager()
    stats = db.get_database_stats()
    
    print(f"📊 Your Database Status:")
    print(f"   Total profiles: {stats['total_profiles']}")
    print(f"   Profiles with vectors: {stats['profiles_with_vectors']}")
    print(f"   Unique floats: {stats['unique_floats']}")
    
    if stats['profiles_with_vectors'] == 0:
        print("❌ No vector data found. Run test_real_argo_storage.py first!")
        return
    
    print("\n🎯 Testing queries against your ARGO data:")
    
    # Test queries
    test_queries = [
        "What temperature profiles do we have?",
        "Show me salinity data from the Indian Ocean",
        "What are the temperature ranges in our data?",
        "Where are the ARGO floats located?",
        "What oceanographic regions are covered?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Test Query {i}: {query}")
        print('='*60)
        
        result = answer_query_with_rag(query)
        
        if result:
            print(f"✅ Query processed successfully!")
        else:
            print(f"❌ Query failed")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    test_rag_pipeline()