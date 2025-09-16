"""
Test the 2023 query specifically
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app import ArgoAnalytics

def test_2023_query():
    """Test query about 2023 data"""
    
    print("🧪 Testing 2023 Query")
    print("=" * 30)
    
    try:
        analytics = ArgoAnalytics()
        
        # Test the problematic query
        test_query = "how many ARGO profiles are in 2023"
        print(f"Query: '{test_query}'")
        
        result = analytics.query_rag_system(test_query, limit=3)
        
        if result:
            print(f"✅ Query processed successfully!")
            print(f"📊 Profiles retrieved: {result['profiles_found']}")
            print(f"🔍 Embedding generated: {result['query_embedding_generated']}")
            print(f"🤖 AI Response:")
            print(f"   {result['answer']}")
            
            if result['profiles_found'] == 0:
                print("✅ CORRECT: No profiles retrieved for 2023 query")
            else:
                print("❌ ISSUE: Profiles were retrieved incorrectly")
                
        else:
            print("❌ Query failed")
        
        print("\n" + "=" * 30)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_2023_query()