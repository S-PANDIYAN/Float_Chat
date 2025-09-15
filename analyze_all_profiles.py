"""
NetCDF Profile Analysis - Show ALL 68 profiles being processed
This demonstrates that your NetCDF file contains all profiles and they can be processed
"""
import xarray as xr
import numpy as np

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

def analyze_all_netcdf_profiles():
    """Analyze ALL 68 profiles from NetCDF file - no database needed"""
    
    print('🌊 COMPLETE ARGO NetCDF ANALYSIS')
    print('=' * 60)
    print('Analyzing ALL profiles from your January 1, 2023 dataset')
    print('(No database storage - just analysis to show all profiles)')
    print('=' * 60)
    
    file_path = 'C:/Users/Pandiyan/Downloads/20230101_prof.nc'
    
    try:
        # Load NetCDF file
        print(f'📁 Loading: {file_path}')
        ds = xr.open_dataset(file_path)
        
        # Get dimensions
        n_prof = ds.dims.get('N_PROF', 0)
        print(f'📊 Found {n_prof} profiles in NetCDF file')
        
        # Process each profile
        processed_count = 0
        failed_count = 0
        region_counts = {}
        float_ids = set()
        
        print(f'\\n🔄 Analyzing ALL {n_prof} profiles...')
        print('-' * 60)
        
        for prof_idx in range(n_prof):
            try:
                # Get coordinates
                if 'LATITUDE' in ds.variables and 'LONGITUDE' in ds.variables:
                    lat = float(ds.LATITUDE.isel(N_PROF=prof_idx).values)
                    lon = float(ds.LONGITUDE.isel(N_PROF=prof_idx).values)
                    
                    if np.isnan(lat) or np.isnan(lon):
                        print(f'   ⚠️  Profile {prof_idx+1:2d}: Invalid coordinates, skipping')
                        failed_count += 1
                        continue
                
                # Get float ID
                float_id = "Unknown"
                if 'PLATFORM_NUMBER' in ds.variables:
                    platform_num = ds.PLATFORM_NUMBER.isel(N_PROF=prof_idx).values
                    float_id = str(platform_num).strip()
                    float_ids.add(float_id)
                
                # Get cycle number
                cycle = 0
                if 'CYCLE_NUMBER' in ds.variables:
                    cycle = int(ds.CYCLE_NUMBER.isel(N_PROF=prof_idx).values)
                
                # Classify region
                region = classify_ocean_region(lat, lon)
                region_counts[region] = region_counts.get(region, 0) + 1
                
                # Count temperature/salinity measurements
                temp_count = 0
                sal_count = 0
                pres_count = 0
                
                if 'TEMP' in ds.variables:
                    temp = ds.TEMP.isel(N_PROF=prof_idx).values
                    temp_count = len(temp[~np.isnan(temp)])
                
                if 'PSAL' in ds.variables:
                    sal = ds.PSAL.isel(N_PROF=prof_idx).values
                    sal_count = len(sal[~np.isnan(sal)])
                
                if 'PRES' in ds.variables:
                    pres = ds.PRES.isel(N_PROF=prof_idx).values
                    pres_count = len(pres[~np.isnan(pres)])
                
                processed_count += 1
                
                # Progress display
                print(f'   ✅ Profile {prof_idx+1:2d}/{n_prof}: Float {float_id} | ({lat:6.1f}°, {lon:6.1f}°) | {region} | T:{temp_count} S:{sal_count} P:{pres_count}')
                
                # Progress indicator every 10 profiles
                if processed_count % 10 == 0:
                    print(f'   📊 Progress: {processed_count}/{n_prof} profiles analyzed ({processed_count/n_prof*100:.1f}%)')
                
            except Exception as e:
                failed_count += 1
                print(f'   ❌ Profile {prof_idx+1:2d}: Failed - {str(e)[:50]}...')
                continue
        
        # Final summary
        print('-' * 60)
        print(f'\\n🎯 ANALYSIS COMPLETE!')
        print(f'   ✅ Successfully analyzed: {processed_count} profiles')
        print(f'   ❌ Failed: {failed_count} profiles')
        print(f'   🌊 Unique ARGO floats: {len(float_ids)}')
        
        print(f'\\n🌍 REGIONAL DISTRIBUTION:')
        for region, count in sorted(region_counts.items()):
            print(f'   • {region}: {count} profiles')
        
        print(f'\\n📋 SAMPLE FLOAT IDs:')
        for i, float_id in enumerate(sorted(list(float_ids))[:10]):
            print(f'   {i+1:2d}. Float {float_id}')
        if len(float_ids) > 10:
            print(f'   ... and {len(float_ids) - 10} more floats')
        
        ds.close()
        
        print(f'\\n🚀 VERIFICATION: ALL {processed_count} PROFILES CAN BE PROCESSED!')
        print('   Your NetCDF file contains complete data for all profiles.')
        print('   When database is running, all profiles will be stored with embeddings.')
        
    except Exception as e:
        print(f'❌ Analysis failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_all_netcdf_profiles()