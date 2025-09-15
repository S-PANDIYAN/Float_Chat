"""
Complete ARGO NetCDF processor - processes and displays ALL profiles
Shows every profile being processed, not just query results
"""
import os
import xarray as xr
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from src.database import DatabaseManager, ArgoProfile
from src.vector_store import VectorStore
from src.argo_processor import ArgoDataProcessor

def classify_ocean_region(lat: float, lon: float) -> str:
    """Accurate ocean region classification"""
    if 20 <= lon <= 147:  # Indian Ocean longitude range
        if lat >= 0:
            return "Northern Indian Ocean"
        else:
            return "Southern Indian Ocean"
    elif 147 <= lon <= 290 or lon <= 70:  # Pacific Ocean
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

def process_complete_netcdf_dataset():
    """Process ALL profiles from NetCDF file and show each one"""
    
    load_dotenv()
    os.environ['DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/vectordb'
    
    print('🌊 COMPLETE ARGO DATASET PROCESSING')
    print('=' * 60)
    print('Processing ALL profiles from your NetCDF file...')
    
    file_path = 'C:/Users/Pandiyan/Downloads/20230101_prof.nc'
    
    try:
        # Initialize components
        db = DatabaseManager()
        vs = VectorStore()
        session = db.Session()
        
        # Load NetCDF file
        ds = xr.open_dataset(file_path)
        
        # Get dimensions
        n_prof = ds.dims.get('N_PROF', 0)
        print(f'\n📁 File: {file_path}')
        print(f'📊 Total profiles to process: {n_prof}')
        
        processed_count = 0
        failed_count = 0
        region_counts = {}
        
        print(f'\n🔄 Processing each profile:')
        print('-' * 60)
        
        for prof_idx in range(n_prof):
            try:
                # Extract profile data
                profile_data = {}
                
                # Basic metadata
                if 'PLATFORM_NUMBER' in ds.variables:
                    float_id = str(ds.PLATFORM_NUMBER.isel(N_PROF=prof_idx).values).strip()
                    profile_data['float_id'] = float_id
                else:
                    profile_data['float_id'] = f'UNKNOWN_{prof_idx}'
                
                # Coordinates
                if 'LATITUDE' in ds.variables and 'LONGITUDE' in ds.variables:
                    lat = float(ds.LATITUDE.isel(N_PROF=prof_idx).values)
                    lon = float(ds.LONGITUDE.isel(N_PROF=prof_idx).values)
                    
                    if not (np.isnan(lat) or np.isnan(lon)):
                        profile_data['latitude'] = lat
                        profile_data['longitude'] = lon
                        profile_data['region'] = classify_ocean_region(lat, lon)
                    else:
                        print(f'   ❌ Profile {prof_idx+1:2d}: Invalid coordinates')
                        failed_count += 1
                        continue
                else:
                    print(f'   ❌ Profile {prof_idx+1:2d}: No coordinate data')
                    failed_count += 1
                    continue
                
                # Date
                profile_data['profile_date'] = '2023-01-01'
                
                # Cycle number
                if 'CYCLE_NUMBER' in ds.variables:
                    cycle = int(ds.CYCLE_NUMBER.isel(N_PROF=prof_idx).values)
                    profile_data['cycle_number'] = cycle
                else:
                    profile_data['cycle_number'] = prof_idx + 1
                
                # Extract measurement data
                measurements = {}
                
                # Temperature
                if 'TEMP' in ds.variables:
                    temp = ds.TEMP.isel(N_PROF=prof_idx).values
                    valid_temp = temp[~np.isnan(temp)]
                    if len(valid_temp) > 0:
                        measurements['temperature_data'] = valid_temp.tolist()
                        measurements['temp_range'] = f'{np.min(valid_temp):.1f}°C to {np.max(valid_temp):.1f}°C'
                
                # Salinity
                if 'PSAL' in ds.variables:
                    sal = ds.PSAL.isel(N_PROF=prof_idx).values
                    valid_sal = sal[~np.isnan(sal)]
                    if len(valid_sal) > 0:
                        measurements['salinity_data'] = valid_sal.tolist()
                        measurements['sal_range'] = f'{np.min(valid_sal):.1f} to {np.max(valid_sal):.1f} PSU'
                
                # Pressure
                if 'PRES' in ds.variables:
                    pres = ds.PRES.isel(N_PROF=prof_idx).values
                    valid_pres = pres[~np.isnan(pres)]
                    if len(valid_pres) > 0:
                        measurements['pressure_data'] = valid_pres.tolist()
                        measurements['depth_range'] = f'0 to {np.max(valid_pres):.0f} dbar'
                
                profile_data.update(measurements)
                
                # Generate summary
                temp_info = measurements.get('temp_range', 'N/A')
                sal_info = measurements.get('sal_range', 'N/A')
                region = profile_data['region']
                
                summary = f"ARGO profile {profile_data['float_id']} at ({lat:.1f}°, {lon:.1f}°) in {region} with temperature range {temp_info} and salinity range {sal_info}"
                profile_data['summary'] = summary
                
                # Generate embedding
                embedding = vs.get_embedding(summary)
                
                # Check if profile already exists
                existing = session.query(ArgoProfile).filter_by(
                    float_id=profile_data['float_id'],
                    cycle_number=profile_data['cycle_number']
                ).first()
                
                if not existing:
                    # Store in database
                    profile = ArgoProfile(
                        float_id=profile_data['float_id'],
                        profile_date=profile_data['profile_date'],
                        cycle_number=profile_data['cycle_number'],
                        latitude=profile_data['latitude'],
                        longitude=profile_data['longitude'],
                        temperature_data=profile_data.get('temperature_data'),
                        salinity_data=profile_data.get('salinity_data'),
                        pressure_data=profile_data.get('pressure_data'),
                        embedding=embedding,
                        summary=profile_data['summary'],
                        region=profile_data['region'],
                        data_quality='processed'
                    )
                    
                    session.add(profile)
                    session.commit()
                    status = '🆕 STORED'
                else:
                    status = '✅ EXISTS'
                
                # Count regions
                region = profile_data['region']
                region_counts[region] = region_counts.get(region, 0) + 1
                
                # Display progress
                print(f'   {status} Profile {prof_idx+1:2d}: Float {profile_data["float_id"]} | '
                      f'({lat:6.1f}°, {lon:6.1f}°) | {region} | '
                      f'Temp: {measurements.get("temp_range", "N/A")}')
                
                processed_count += 1
                
            except Exception as e:
                print(f'   ❌ Profile {prof_idx+1:2d}: Failed - {str(e)[:50]}...')
                failed_count += 1
                continue
        
        ds.close()
        session.close()
        
        print('-' * 60)
        print(f'\\n📊 PROCESSING COMPLETE:')
        print(f'   ✅ Successfully processed: {processed_count} profiles')
        print(f'   ❌ Failed to process: {failed_count} profiles')
        print(f'   📈 Success rate: {(processed_count/(processed_count+failed_count)*100):.1f}%')
        
        print(f'\\n🌍 Regional Distribution:')
        for region, count in sorted(region_counts.items()):
            print(f'   • {region}: {count} profiles')
        
        print(f'\\n🎯 ALL {processed_count} profiles are now available for querying!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_complete_netcdf_dataset()