"""
Quick test to verify the frontend fixes
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Test the fixed ArgoAnalytics class
print("🧪 Testing Fixed Frontend Components...")
print("=" * 50)

try:
    from src.database import DatabaseManager
    
    # Test direct database stats
    print("1. Testing direct database stats...")
    db = DatabaseManager()
    stats = db.get_database_stats()
    print(f"   ✅ Direct stats: {stats['total_profiles']} profiles, {stats['unique_floats']} floats")
    
    # Test regional query fix
    print("\n2. Testing regional distribution query...")
    session = db.Session()
    try:
        from sqlalchemy import text
        regional_query = text("""
            SELECT region, COUNT(*) as count 
            FROM argo_profiles 
            WHERE region IS NOT NULL 
            GROUP BY region 
            ORDER BY count DESC
        """)
        regional_stats = session.execute(regional_query).fetchall()
        print(f"   ✅ Regional query: Found {len(regional_stats)} regions")
        for region, count in regional_stats:
            print(f"      - {region}: {count} profiles")
    finally:
        session.close()
    
    # Test ArgoAnalytics class
    print("\n3. Testing ArgoAnalytics class...")
    sys.path.append(str(Path(__file__).parent))
    from app import ArgoAnalytics
    
    analytics = ArgoAnalytics()
    analytics_stats = analytics.get_database_stats()
    print(f"   ✅ ArgoAnalytics stats: {analytics_stats['total_profiles']} profiles")
    print(f"   ✅ Regional distribution: {len(analytics_stats['regional_distribution'])} regions")
    
    print("\n" + "=" * 50)
    print("🎉 All frontend fixes verified!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()