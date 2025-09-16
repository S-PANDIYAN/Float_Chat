"""
Fix the profile dates in the database from 2023 to 2025
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.database import DatabaseManager, ArgoProfile

def fix_profile_dates():
    """Update profile dates from 2023 to 2025"""
    
    print("🔧 Fixing profile dates in database...")
    
    try:
        db = DatabaseManager()
        session = db.Session()
        
        # Update all profiles with 2023 dates to 2025
        updated_count = session.query(ArgoProfile).filter(
            ArgoProfile.profile_date == "2023-01-01"
        ).update({
            ArgoProfile.profile_date: "2025-01-01"
        })
        
        session.commit()
        
        print(f"✅ Updated {updated_count} profiles from 2023-01-01 to 2025-01-01")
        
        # Verify the update
        total_2025 = session.query(ArgoProfile).filter(
            ArgoProfile.profile_date == "2025-01-01"
        ).count()
        
        print(f"📊 Now have {total_2025} profiles with 2025-01-01 date")
        
        session.close()
        
    except Exception as e:
        print(f"❌ Error updating dates: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()

if __name__ == "__main__":
    fix_profile_dates()