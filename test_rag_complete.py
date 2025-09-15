"""
Test RAG functionality end-to-end to ensure everything works
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

print("🔬 Testing Complete RAG Pipeline...")
print("=" * 50)

try:
    # Import the ArgoAnalytics class
    from app import ArgoAnalytics
    
    # Initialize analytics
    analytics = ArgoAnalytics()
    
    # Test a sample query
    test_query = "What temperature data do we have from the Indian Ocean?"
    print(f"Query: {test_query}")
    
    print("\n1. Testing RAG system...")
    result = analytics.query_rag_system(test_query, limit=2)
    
    if result:
        print("   ✅ RAG query successful!")
        print(f"   📊 Found {result['profiles_found']} profiles")
        print(f"   🤖 AI Response preview: {result['answer'][:100]}...")
        
        print(f"\n   📍 Retrieved profiles:")
        for i, profile in enumerate(result['profiles'], 1):
            print(f"      {i}. Float {profile['float_id']} - {profile['region']}")
            print(f"         Location: {profile['latitude']:.1f}°, {profile['longitude']:.1f}°")
            print(f"         Similarity: {profile['similarity']:.3f}")
    else:
        print("   ❌ RAG query failed")
    
    print("\n" + "=" * 50)
    print("🎉 Complete system test finished!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()