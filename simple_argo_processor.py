"""
🌊 Simple ARGO Data Processor - Clean Start
Process ARGO NetCDF files with error handling for datetime issues.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import project modules
from src.database import DatabaseManager, store_argo_profile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_embedding(text: str) -> List[float]:
    """Generate embedding using local Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "embeddinggemma", "prompt": text},
            timeout=10
        )
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            if len(embedding) == 768:
                return embedding
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
    return []

def classify_region(lat: float, lon: float) -> str:
    """Classify ocean region"""
    if -10 <= lat <= 30 and 40 <= lon <= 100:
        return "Northern Indian Ocean"
    elif -60 <= lat < -10 and 20 <= lon <= 120:
        return "Southern Indian Ocean"
    else:
        return "Other Ocean Region"

def process_netcdf_simple(file_path: str) -> List[dict]:
    """Simple NetCDF processing with error handling"""
    import xarray as xr
    import numpy as np
    
    logger.info(f"Processing: {file_path}")
    
    try:
        ds = xr.open_dataset(file_path)
        profiles = []
        
        # Get dimensions
        n_prof = ds.sizes.get('N_PROF', 1)
        
        logger.info(f"Found {n_prof} profiles in file")
        
        for i in range(n_prof):
            try:
                # Extract basic data with error handling
                profile = {}
                
                # Float ID
                if 'PLATFORM_NUMBER' in ds.variables:
                    float_id = ds['PLATFORM_NUMBER'].values
                    if hasattr(float_id, '__len__') and len(float_id) > i:
                        profile['float_id'] = str(float_id[i]).strip()
                    else:
                        profile['float_id'] = str(float_id).strip()
                else:
                    profile['float_id'] = f"UNKNOWN_{i}"
                
                # Coordinates
                if 'LATITUDE' in ds.variables:
                    lat = ds['LATITUDE'].values
                    profile['latitude'] = float(lat[i] if hasattr(lat, '__len__') else lat)
                else:
                    profile['latitude'] = 0.0
                
                if 'LONGITUDE' in ds.variables:
                    lon = ds['LONGITUDE'].values
                    profile['longitude'] = float(lon[i] if hasattr(lon, '__len__') else lon)
                else:
                    profile['longitude'] = 0.0
                
                # Cycle number
                if 'CYCLE_NUMBER' in ds.variables:
                    cycle = ds['CYCLE_NUMBER'].values
                    profile['cycle_number'] = int(cycle[i] if hasattr(cycle, '__len__') else cycle)
                else:
                    profile['cycle_number'] = i
                
                # Date - extract from filename or use 2025 default
                if "2025" in file_path:
                    profile['profile_date'] = "2025-01-01"
                elif "2023" in file_path:
                    profile['profile_date'] = "2023-01-01"
                else:
                    profile['profile_date'] = "2025-01-01"  # Default for current data
                
                # Region classification
                profile['region'] = classify_region(profile['latitude'], profile['longitude'])
                
                # Temperature data
                profile['temperature_data'] = {}
                if 'TEMP' in ds.variables:
                    try:
                        temp_vals = ds['TEMP'].values
                        if len(temp_vals.shape) == 2:
                            temps = temp_vals[i, :]
                        else:
                            temps = temp_vals
                        
                        # Clean data
                        valid_temps = [float(t) for t in temps if not np.isnan(t) and t != 99999.0]
                        if valid_temps:
                            profile['temperature_data'] = {
                                'values': valid_temps,
                                'count': len(valid_temps),
                                'min': min(valid_temps),
                                'max': max(valid_temps)
                            }
                    except Exception as e:
                        logger.warning(f"Temperature processing error for profile {i}: {e}")
                
                # Salinity data
                profile['salinity_data'] = {}
                if 'PSAL' in ds.variables:
                    try:
                        sal_vals = ds['PSAL'].values
                        if len(sal_vals.shape) == 2:
                            sals = sal_vals[i, :]
                        else:
                            sals = sal_vals
                        
                        # Clean data
                        valid_sals = [float(s) for s in sals if not np.isnan(s) and s != 99999.0]
                        if valid_sals:
                            profile['salinity_data'] = {
                                'values': valid_sals,
                                'count': len(valid_sals),
                                'min': min(valid_sals),
                                'max': max(valid_sals)
                            }
                    except Exception as e:
                        logger.warning(f"Salinity processing error for profile {i}: {e}")
                
                # Pressure data
                profile['pressure_data'] = {}
                if 'PRES' in ds.variables:
                    try:
                        pres_vals = ds['PRES'].values
                        if len(pres_vals.shape) == 2:
                            pressures = pres_vals[i, :]
                        else:
                            pressures = pres_vals
                        
                        # Clean data
                        valid_pres = [float(p) for p in pressures if not np.isnan(p) and p != 99999.0]
                        if valid_pres:
                            profile['pressure_data'] = {
                                'values': valid_pres,
                                'count': len(valid_pres),
                                'min': min(valid_pres),
                                'max': max(valid_pres)
                            }
                    except Exception as e:
                        logger.warning(f"Pressure processing error for profile {i}: {e}")
                
                # Only add profile if it has valid coordinates
                if profile['latitude'] != 0.0 and profile['longitude'] != 0.0:
                    profiles.append(profile)
                    
            except Exception as e:
                logger.warning(f"Error processing profile {i}: {e}")
                continue
        
        ds.close()
        logger.info(f"Successfully extracted {len(profiles)} valid profiles")
        return profiles
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []

def create_summary(profile: dict) -> str:
    """Create profile summary for embedding"""
    float_id = profile.get('float_id', 'Unknown')
    lat = profile.get('latitude', 0)
    lon = profile.get('longitude', 0)
    region = profile.get('region', 'Unknown')
    date = profile.get('profile_date', 'Unknown')
    
    # Temperature summary
    temp_data = profile.get('temperature_data', {})
    if temp_data and 'values' in temp_data:
        temp_info = f"Temperature: {temp_data['min']:.1f}°C to {temp_data['max']:.1f}°C ({temp_data['count']} measurements)"
    else:
        temp_info = "No temperature data"
    
    # Salinity summary  
    sal_data = profile.get('salinity_data', {})
    if sal_data and 'values' in sal_data:
        sal_info = f"Salinity: {sal_data['min']:.1f} to {sal_data['max']:.1f} PSU ({sal_data['count']} measurements)"
    else:
        sal_info = "No salinity data"
    
    # Pressure summary
    pres_data = profile.get('pressure_data', {})
    if pres_data and 'values' in pres_data:
        depth_info = f"Depth: 0 to {pres_data['max']:.0f} dbar ({pres_data['count']} levels)"
    else:
        depth_info = "No pressure data"
    
    summary = f"""ARGO Float {float_id} - {region}
Location: {lat:.2f}°N, {lon:.2f}°E
Date: {date}
{temp_info}
{sal_info}
{depth_info}
Complete oceanographic profile with measurements from surface to depth."""
    
    return summary

def main():
    """Main processing function"""
    print("🌊 Simple ARGO Data Processor")
    print("=" * 40)
    
    # Check database
    try:
        db = DatabaseManager()
        stats = db.get_database_stats()
        print(f"📊 Database: {stats['total_profiles']} existing profiles")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return
    
    # Check embedding service
    test_embedding = get_embedding("test")
    if len(test_embedding) == 768:
        print("✅ Embedding service ready")
    else:
        print("⚠️ Embedding service not ready")
    
    # Find NetCDF files
    downloads = Path.home() / "Downloads"
    nc_files = list(downloads.glob("*.nc"))
    
    if not nc_files:
        print("❌ No NetCDF files found in Downloads")
        return
    
    print(f"\n📂 Found {len(nc_files)} NetCDF files:")
    for i, file in enumerate(nc_files):
        print(f"  {i+1}. {file.name}")
    
    # Select file
    try:
        choice = input(f"\nSelect file (1-{len(nc_files)}) or Enter for first file: ").strip()
        if choice:
            selected_file = nc_files[int(choice) - 1]
        else:
            selected_file = nc_files[0]
    except (ValueError, IndexError):
        selected_file = nc_files[0]
    
    print(f"🚀 Processing: {selected_file.name}")
    
    # Clear existing data?
    if stats['total_profiles'] > 0:
        clear = input(f"Clear {stats['total_profiles']} existing profiles? (y/N): ").strip().lower()
        if clear in ['y', 'yes']:
            session = db.Session()
            try:
                from src.database import ArgoProfile
                deleted = session.query(ArgoProfile).delete()
                session.commit()
                print(f"🗑️ Deleted {deleted} profiles")
            except Exception as e:
                session.rollback()
                print(f"❌ Clear failed: {e}")
            finally:
                session.close()
    
    # Process file
    profiles = process_netcdf_simple(str(selected_file))
    
    if not profiles:
        print("❌ No valid profiles extracted")
        return
    
    print(f"📊 Processing {len(profiles)} profiles...")
    
    # Store profiles
    stored = 0
    with_embeddings = 0
    
    for i, profile in enumerate(profiles):
        try:
            # Create summary
            summary = create_summary(profile)
            profile['summary'] = summary
            
            # Generate embedding
            embedding = get_embedding(summary)
            
            # Store in database
            profile_id = store_argo_profile(profile, embedding, db)
            stored += 1
            
            if len(embedding) == 768:
                with_embeddings += 1
            
            if (i + 1) % 10 == 0:
                print(f"  ✅ Stored {i+1}/{len(profiles)} profiles...")
                
        except Exception as e:
            print(f"  ❌ Error storing profile {i+1}: {e}")
            continue
    
    # Final results
    final_stats = db.get_database_stats()
    
    print("\n" + "=" * 40)
    print("🎉 Processing Complete!")
    print("=" * 40)
    print(f"📊 Profiles extracted: {len(profiles)}")
    print(f"💾 Profiles stored: {stored}")
    print(f"🎯 With embeddings: {with_embeddings}")
    print(f"📈 Total in database: {final_stats['total_profiles']}")
    print(f"🏷️ Unique floats: {final_stats['unique_floats']}")
    
    print("\n🚀 Ready to run frontend:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()