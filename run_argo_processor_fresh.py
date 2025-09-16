"""
🌊 ARGO Data Processor - Fresh Start Script
Process ARGO NetCDF files from a new data path and store in database.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import project modules
from src.argo_processor import ArgoDataProcessor
from src.database import DatabaseManager, store_argo_profile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_embedding(text: str) -> List[float]:
    """Generate embedding using local Ollama embeddinggemma model"""
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma:latest")
        
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model_name, "prompt": text},
            timeout=15
        )
        
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            if len(embedding) == 768:  # Verify correct dimensions
                return embedding
                
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
    
    return []

def classify_ocean_region(latitude: float, longitude: float) -> str:
    """Classify ocean region based on coordinates (accurate for your data)"""
    # Accurate classification for Indian Ocean ARGO data
    if -10 <= latitude <= 30 and 40 <= longitude <= 100:
        return "Northern Indian Ocean"
    elif -60 <= latitude < -10 and 20 <= longitude <= 120:
        return "Southern Indian Ocean"
    elif -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return "Other Ocean Region"
    else:
        return "Unknown Region"

def create_profile_summary(profile_data: dict) -> str:
    """Create a comprehensive summary of the ARGO profile"""
    try:
        # Extract key information
        float_id = profile_data.get('float_id', 'Unknown')
        lat = profile_data.get('latitude', 0)
        lon = profile_data.get('longitude', 0)
        date = profile_data.get('profile_date', 'Unknown')
        cycle = profile_data.get('cycle_number', 'Unknown')
        region = classify_ocean_region(lat, lon)
        
        # Temperature data summary
        temp_data = profile_data.get('temperature_data', {})
        temp_summary = "No temperature data"
        if temp_data and 'values' in temp_data:
            temps = [t for t in temp_data['values'] if t is not None and not pd.isna(t)]
            if temps:
                temp_summary = f"Temperature range: {min(temps):.1f}°C to {max(temps):.1f}°C ({len(temps)} measurements)"
        
        # Salinity data summary
        sal_data = profile_data.get('salinity_data', {})
        sal_summary = "No salinity data"
        if sal_data and 'values' in sal_data:
            sals = [s for s in sal_data['values'] if s is not None and not pd.isna(s)]
            if sals:
                sal_summary = f"Salinity range: {min(sals):.1f} to {max(sals):.1f} PSU ({len(sals)} measurements)"
        
        # Pressure/depth data summary
        press_data = profile_data.get('pressure_data', {})
        depth_summary = "No pressure data"
        if press_data and 'values' in press_data:
            pressures = [p for p in press_data['values'] if p is not None and not pd.isna(p)]
            if pressures:
                depth_summary = f"Depth range: 0 to {max(pressures):.0f} dbar ({len(pressures)} levels)"
        
        # Create comprehensive summary
        summary = f"""ARGO Float {float_id}, Cycle {cycle}
Location: {lat:.2f}°N, {lon:.2f}°E ({region})
Date: {date}
{temp_summary}
{sal_summary} 
{depth_summary}
Complete oceanographic profile with temperature, salinity, and pressure measurements from surface to depth."""
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating profile summary: {e}")
        return f"ARGO profile from float {profile_data.get('float_id', 'Unknown')}"

def process_and_store_data(data_path: str, clear_existing: bool = False) -> dict:
    """Process ARGO data from new path and store in database"""
    
    logger.info(f"🌊 Starting ARGO data processing from: {data_path}")
    
    # Validate data path
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Initialize components
    db = DatabaseManager()
    processor = ArgoDataProcessor()
    
    # Clear existing data if requested
    if clear_existing:
        logger.info("🗑️ Clearing existing database data...")
        session = db.Session()
        try:
            from src.database import ArgoProfile
            deleted_count = session.query(ArgoProfile).delete()
            session.commit()
            logger.info(f"Deleted {deleted_count} existing profiles")
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing data: {e}")
        finally:
            session.close()
    
    # Process data
    stats = {
        'files_found': 0,
        'files_processed': 0,
        'profiles_extracted': 0,
        'profiles_stored': 0,
        'profiles_with_embeddings': 0,
        'errors': []
    }
    
    try:
        # Check if data_path is a file or directory
        if data_path_obj.is_file():
            # Single file
            if data_path_obj.suffix.lower() in ['.nc', '.netcdf']:
                logger.info(f"Processing single NetCDF file: {data_path}")
                stats['files_found'] = 1
                
                if processor.validate_file_path(str(data_path_obj)):
                    result = processor.process_file(str(data_path_obj))
                    stats['files_processed'] = 1
                    
                    # Process profiles from the file
                    profiles = result.get('profiles', [])
                    stats['profiles_extracted'] = len(profiles)
                    
                    logger.info(f"📊 Extracted {len(profiles)} profiles from file")
                    
                    # Store each profile with embedding
                    for i, profile_data in enumerate(profiles):
                        try:
                            # Create summary for embedding
                            summary = create_profile_summary(profile_data)
                            
                            # Generate embedding
                            embedding = get_embedding(summary)
                            
                            if len(embedding) == 768:
                                # Store in database
                                profile_id = store_argo_profile(profile_data, embedding, db)
                                stats['profiles_stored'] += 1
                                stats['profiles_with_embeddings'] += 1
                                
                                logger.info(f"✅ Stored profile {i+1}/{len(profiles)}: Float {profile_data.get('float_id')}")
                            else:
                                logger.warning(f"⚠️ No embedding for profile {i+1}, storing without vector")
                                profile_id = store_argo_profile(profile_data, [], db)
                                stats['profiles_stored'] += 1
                                
                        except Exception as e:
                            error_msg = f"Error storing profile {i+1}: {e}"
                            stats['errors'].append(error_msg)
                            logger.error(error_msg)
                            continue
                else:
                    stats['errors'].append(f"Invalid NetCDF file: {data_path}")
            else:
                raise ValueError(f"File is not a NetCDF file: {data_path}")
                
        elif data_path_obj.is_dir():
            # Directory of files
            logger.info(f"Processing directory: {data_path}")
            results = processor.process_directory(str(data_path_obj))
            
            stats['files_found'] = len(list(data_path_obj.glob("*.nc")))
            stats['files_processed'] = len(results)
            
            # Process all profiles from all files
            all_profiles = []
            for result in results:
                all_profiles.extend(result.get('profiles', []))
            
            stats['profiles_extracted'] = len(all_profiles)
            logger.info(f"📊 Extracted {len(all_profiles)} total profiles from {len(results)} files")
            
            # Store each profile with embedding
            for i, profile_data in enumerate(all_profiles):
                try:
                    # Create summary for embedding
                    summary = create_profile_summary(profile_data)
                    
                    # Generate embedding
                    embedding = get_embedding(summary)
                    
                    if len(embedding) == 768:
                        # Store in database
                        profile_id = store_argo_profile(profile_data, embedding, db)
                        stats['profiles_stored'] += 1
                        stats['profiles_with_embeddings'] += 1
                        
                        if (i + 1) % 10 == 0:  # Log every 10th profile
                            logger.info(f"✅ Stored {i+1}/{len(all_profiles)} profiles...")
                    else:
                        logger.warning(f"⚠️ No embedding for profile {i+1}, storing without vector")
                        profile_id = store_argo_profile(profile_data, [], db)
                        stats['profiles_stored'] += 1
                        
                except Exception as e:
                    error_msg = f"Error storing profile {i+1}: {e}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)
                    continue
        else:
            raise ValueError(f"Path is neither a file nor directory: {data_path}")
    
    except Exception as e:
        error_msg = f"Processing failed: {e}"
        stats['errors'].append(error_msg)
        logger.error(error_msg)
        raise
    
    return stats

def main():
    """Main function to run ARGO processor from scratch"""
    
    print("🌊 ARGO Data Processor - Fresh Start")
    print("=" * 50)
    
    # Check system prerequisites
    logger.info("Checking system prerequisites...")
    
    # Check database connection
    try:
        db = DatabaseManager()
        db_stats = db.get_database_stats()
        logger.info(f"✅ Database connected: {db_stats['total_profiles']} existing profiles")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return
    
    # Check Ollama service
    try:
        test_embedding = get_embedding("test")
        if len(test_embedding) == 768:
            logger.info("✅ Ollama embedding service ready")
        else:
            logger.warning("⚠️ Ollama embedding service not working properly")
    except Exception as e:
        logger.warning(f"⚠️ Ollama service check failed: {e}")
    
    # Get data path from user
    print("\n📂 Enter your ARGO data path:")
    print("   - Single NetCDF file: C:\\path\\to\\your\\file.nc")
    print("   - Directory with NetCDF files: C:\\path\\to\\your\\data\\directory")
    print("   - Press Enter to use default Downloads directory")
    
    data_path = input("\nData path: ").strip()
    
    if not data_path:
        # Default to Downloads directory
        downloads_path = Path.home() / "Downloads"
        nc_files = list(downloads_path.glob("*.nc"))
        if nc_files:
            data_path = str(nc_files[0])  # Use first NetCDF file found
            logger.info(f"Using default file from Downloads: {data_path}")
        else:
            logger.error("No NetCDF files found in Downloads directory")
            return
    
    # Ask about clearing existing data
    clear_existing = False
    if db_stats['total_profiles'] > 0:
        response = input(f"\n🗑️ Database has {db_stats['total_profiles']} existing profiles. Clear them? (y/N): ").strip().lower()
        clear_existing = response in ['y', 'yes']
    
    try:
        # Process the data
        logger.info(f"🚀 Starting processing of: {data_path}")
        stats = process_and_store_data(data_path, clear_existing)
        
        # Print results
        print("\n" + "=" * 50)
        print("🎉 ARGO Data Processing Complete!")
        print("=" * 50)
        print(f"📁 Files found: {stats['files_found']}")
        print(f"✅ Files processed: {stats['files_processed']}")
        print(f"📊 Profiles extracted: {stats['profiles_extracted']}")
        print(f"💾 Profiles stored: {stats['profiles_stored']}")
        print(f"🎯 Profiles with embeddings: {stats['profiles_with_embeddings']}")
        
        if stats['errors']:
            print(f"⚠️ Errors encountered: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
        
        # Final database stats
        final_stats = db.get_database_stats()
        print(f"\n📈 Final Database Status:")
        print(f"   Total profiles: {final_stats['total_profiles']}")
        print(f"   Profiles with vectors: {final_stats['profiles_with_vectors']}")
        print(f"   Unique floats: {final_stats['unique_floats']}")
        
        print("\n🚀 Ready to run Streamlit app:")
        print("   streamlit run app.py")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()