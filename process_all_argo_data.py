"""
Process ALL profiles from your ARGO NetCDF file.
No limits - processes all 68 profiles from your data.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import requests
import xarray as xr
import numpy as np
from src.database import DatabaseManager, store_argo_profile

def get_embedding_from_ollama(text: str) -> list:
    """Generate 768-dimensional embedding using embeddinggemma"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "embeddinggemma", "prompt": text},
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json().get("embedding", [])
    except Exception as e:
        print(f"❌ Embedding error: {e}")
    return []

def process_all_argo_profiles(netcdf_file: str):
    """Process ALL profiles from your ARGO NetCDF file - no limits!"""
    
    print(f"🌊 Processing ALL profiles from: {netcdf_file}")
    
    # Load your actual ARGO data
    try:
        ds = xr.open_dataset(netcdf_file)
        print(f"✅ Loaded NetCDF file successfully")
        
        # Get number of profiles
        n_profiles = ds.dims.get('N_PROF', 1)
        print(f"📊 Found {n_profiles} profiles in your file")
        print(f"🎯 Will process ALL {n_profiles} profiles (no limits)")
        
        # Initialize database
        db = DatabaseManager()
        stored_count = 0
        failed_count = 0
        
        # Process ALL profiles from your data
        for i in range(n_profiles):  # Process ALL profiles - no limit!
            try:
                print(f"\n🔄 Processing profile {i+1}/{n_profiles} ({((i+1)/n_profiles)*100:.1f}%)")
                
                # Extract float ID
                if 'PLATFORM_NUMBER' in ds.variables:
                    platform_bytes = ds['PLATFORM_NUMBER'].values[i]
                    if hasattr(platform_bytes, 'decode'):
                        float_id = platform_bytes.decode('utf-8').strip()
                    else:
                        float_id = str(platform_bytes).strip()
                    
                    # Remove any null characters
                    float_id = float_id.replace('\x00', '').strip()
                    if not float_id:
                        float_id = f"ARGO_{i}"
                else:
                    float_id = f"ARGO_{i}"
                
                # Extract coordinates
                if 'LATITUDE' in ds.variables and 'LONGITUDE' in ds.variables:
                    lat = float(ds['LATITUDE'].values[i])
                    lon = float(ds['LONGITUDE'].values[i])
                    
                    # Skip if coordinates are invalid
                    if np.isnan(lat) or np.isnan(lon):
                        print(f"⚠️ Skipping profile {i+1}: Invalid coordinates")
                        failed_count += 1
                        continue
                else:
                    print(f"⚠️ Skipping profile {i+1}: No coordinates found")
                    failed_count += 1
                    continue
                
                # Extract date
                profile_date = "2023-01-01"  # Default
                if 'JULD' in ds.variables:
                    try:
                        julian_day = ds['JULD'].values[i]
                        if not np.isnan(julian_day):
                            import datetime
                            base_date = datetime.datetime(1950, 1, 1)
                            profile_datetime = base_date + datetime.timedelta(days=float(julian_day))
                            profile_date = profile_datetime.strftime("%Y-%m-%d")
                    except:
                        pass
                
                # Extract temperature and salinity
                temp_data = None
                salinity_data = None
                
                if 'TEMP' in ds.variables:
                    temp_profile = ds['TEMP'].values[i, :]
                    valid_temps = temp_profile[~np.isnan(temp_profile)]
                    if len(valid_temps) > 0:
                        temp_data = {
                            'surface': float(valid_temps[0]) if len(valid_temps) > 0 else None,
                            'mean': float(np.mean(valid_temps)),
                            'min': float(np.min(valid_temps)),
                            'max': float(np.max(valid_temps)),
                            'profile_length': len(valid_temps)
                        }
                
                if 'PSAL' in ds.variables:
                    sal_profile = ds['PSAL'].values[i, :]
                    valid_sals = sal_profile[~np.isnan(sal_profile)]
                    if len(valid_sals) > 0:
                        salinity_data = {
                            'surface': float(valid_sals[0]) if len(valid_sals) > 0 else None,
                            'mean': float(np.mean(valid_sals)),
                            'min': float(np.min(valid_sals)),
                            'max': float(np.max(valid_sals)),
                            'profile_length': len(valid_sals)
                        }
                
                # Skip if no valid data
                if not temp_data and not salinity_data:
                    print(f"⚠️ Skipping profile {i+1}: No valid temperature or salinity data")
                    failed_count += 1
                    continue
                
                # Create descriptive summary for embedding
                temp_desc = ""
                if temp_data:
                    temp_desc = f"temperature range {temp_data['min']:.1f}°C to {temp_data['max']:.1f}°C"
                
                sal_desc = ""
                if salinity_data:
                    sal_desc = f"salinity range {salinity_data['min']:.1f} to {salinity_data['max']:.1f}"
                
                # Determine region based on actual coordinates
                region = "Unknown"
                # Indian Ocean classification
                if -10 <= lat <= 30 and 40 <= lon <= 100:
                    region = "Northern Indian Ocean"
                elif -40 <= lat <= -10 and 40 <= lon <= 100:
                    region = "Southern Indian Ocean"
                elif lat >= 0 and -30 <= lon < 0:
                    region = "North Atlantic"
                elif lat < 0 and -60 <= lon < 20:
                    region = "South Atlantic"
                elif lat >= 0 and lon >= 100:
                    region = "North Pacific"
                elif lat < 0 and lon >= 100:
                    region = "South Pacific"
                else:
                    region = f"Ocean_{lat:.1f}N_{lon:.1f}E"
                
                summary = f"ARGO profile {float_id} at ({lat:.1f}°, {lon:.1f}°) in {region}"
                if temp_desc:
                    summary += f" with {temp_desc}"
                if sal_desc:
                    summary += f" and {sal_desc}"
                
                # Generate 768-dimensional embedding
                embedding = get_embedding_from_ollama(summary)
                
                if len(embedding) == 768:
                    # Prepare profile data
                    profile_data = {
                        'float_id': float_id,
                        'profile_date': profile_date,
                        'cycle_number': i + 1,
                        'latitude': lat,
                        'longitude': lon,
                        'temperature_data': temp_data,
                        'salinity_data': salinity_data,
                        'summary': summary,
                        'region': region,
                        'data_quality': 'processed'
                    }
                    
                    # Store in database with vector
                    profile_id = store_argo_profile(profile_data, embedding, db)
                    print(f"✅ Stored profile {float_id} (ID: {profile_id})")
                    stored_count += 1
                
                else:
                    print(f"❌ Failed to generate embedding for profile {i+1}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing profile {i+1}: {e}")
                failed_count += 1
                continue
        
        print(f"\n🎉 Processing Complete!")
        print(f"✅ Successfully stored: {stored_count} profiles")
        print(f"❌ Failed to process: {failed_count} profiles")
        print(f"📊 Total attempted: {n_profiles} profiles")
        print(f"📈 Success rate: {(stored_count/n_profiles)*100:.1f}%")
        
        # Show final database stats
        stats = db.get_database_stats()
        print(f"\n📊 Final Database Status:")
        print(f"   Total profiles: {stats['total_profiles']}")
        print(f"   Profiles with vectors: {stats['profiles_with_vectors']}")
        print(f"   Unique floats: {stats['unique_floats']}")
        
        ds.close()
        
    except Exception as e:
        print(f"❌ Error loading NetCDF file: {e}")

if __name__ == "__main__":
    # Use your actual ARGO NetCDF file
    argo_file = "C:/Users/Pandiyan/Downloads/20230101_prof.nc"
    
    if os.path.exists(argo_file):
        # Set environment variables
        if not os.getenv("DATABASE_URI"):
            os.environ["DATABASE_URI"] = "postgresql://postgres:postgres@localhost:5432/vectordb"
        
        print("🚀 Starting to process ALL ARGO profiles...")
        print("⏳ This will take some time to generate embeddings for all profiles...")
        
        process_all_argo_profiles(argo_file)
    else:
        print(f"❌ ARGO file not found: {argo_file}")
        print("Please provide path to your actual ARGO NetCDF file")