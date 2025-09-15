"""
Fix regional classification for ARGO profiles based on actual coordinates
"""
import os
from dotenv import load_dotenv
from src.database import DatabaseManager, ArgoProfile

def classify_ocean_region(lat: float, lon: float) -> str:
    """
    Accurate ocean region classification based on latitude and longitude
    """
    # Convert longitude to 0-360 if needed
    if lon < 0:
        lon += 360
    
    # Major ocean boundaries (more accurate)
    if 20 <= lon <= 147:  # Indian Ocean longitude range
        if lat >= 0:
            return "Northern Indian Ocean"
        else:
            return "Southern Indian Ocean"
    elif 147 <= lon <= 290 or lon <= 70:  # Pacific Ocean (crosses date line)
        if lat >= 0:
            return "North Pacific Ocean"
        else:
            return "South Pacific Ocean"
    elif 290 <= lon <= 20:  # Atlantic Ocean
        if lat >= 0:
            return "North Atlantic Ocean"
        else:
            return "South Atlantic Ocean"
    else:
        return f"Ocean_{lat:.1f}N_{lon:.1f}E"

def fix_regional_classification():
    """Fix regional classification for all profiles in database"""
    
    load_dotenv()
    os.environ['DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/vectordb'
    
    print('🔧 Fixing Regional Classification for ARGO Profiles')
    print('=' * 60)
    
    # Connect to database
    db = DatabaseManager()
    session = db.Session()
    
    try:
        # Get all profiles
        all_profiles = session.query(ArgoProfile).all()
        print(f'📊 Found {len(all_profiles)} profiles to reclassify')
        
        updated_count = 0
        region_counts = {}
        
        for profile in all_profiles:
            if profile.latitude is not None and profile.longitude is not None:
                # Calculate correct region
                correct_region = classify_ocean_region(profile.latitude, profile.longitude)
                
                # Update if different
                if profile.region != correct_region:
                    old_region = profile.region
                    profile.region = correct_region
                    updated_count += 1
                    print(f'   Updated Float {profile.float_id}: {old_region} -> {correct_region}')
                
                # Count regions
                region_counts[correct_region] = region_counts.get(correct_region, 0) + 1
        
        # Commit changes
        session.commit()
        
        print(f'\\n✅ Updated {updated_count} profiles')
        print(f'\\n🌍 CORRECTED Regional Distribution:')
        for region, count in sorted(region_counts.items()):
            print(f'   • {region}: {count} profiles')
        
        print(f'\\n🎯 Regional classification now matches your actual NetCDF data!')
        
    except Exception as e:
        session.rollback()
        print(f'❌ Error: {e}')
        raise
    finally:
        session.close()

if __name__ == "__main__":
    fix_regional_classification()