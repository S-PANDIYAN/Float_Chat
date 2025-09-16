"""
Test the fixed app functionality
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app import ArgoAnalytics

def test_fixed_app():
    """Test the app fixes"""
    
    print("🧪 Testing Fixed App Functionality")
    print("=" * 40)
    
    try:
        analytics = ArgoAnalytics()
        
        # Test database stats
        print("1. Testing database stats...")
        stats = analytics.get_database_stats()
        print(f"   ✅ Total profiles: {stats['total_profiles']}")
        print(f"   ✅ Vectorized profiles: {stats['profiles_with_vectors']}")
        print(f"   ✅ Unique floats: {stats['unique_floats']}")
        print(f"   ✅ Regional distribution: {stats['regional_distribution']}")
        
        # Test RAG query with improved prompt
        print("\n2. Testing improved RAG response...")
        test_query = "How many ARGO profiles are in our dataset?"
        result = analytics.query_rag_system(test_query, limit=3)
        
        if result:
            print(f"   ✅ Query successful!")
            print(f"   📊 Profiles found: {result['profiles_found']}")
            print(f"   🤖 AI Response preview:")
            print(f"      {result['answer'][:200]}...")
            
            # Check if dates are now 2025
            if result['profiles']:
                first_profile = result['profiles'][0]
                print(f"   📅 First profile date: {first_profile.get('profile_date', 'N/A')}")
        else:
            print("   ❌ Query failed")
        
        print("\n" + "=" * 40)
        print("🎉 App testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_app()