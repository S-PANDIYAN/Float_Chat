import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import load_config
from src.database import init_database, DatabaseManager
from src.argo_processor import ArgoDataProcessor
from src.vector_store import VectorStore

st.set_page_config(
    page_title="Data Upload & Processing",
    page_icon="📁",
    layout="wide"
)

def main():
    st.title("📁 ARGO Data Upload & Processing")
    
    # Configuration
    config = load_config()
    
    # File upload section
    st.header("1. Upload ARGO NetCDF Files")
    
    uploaded_files = st.file_uploader(
        "Select ARGO NetCDF files to process",
        type=['nc'],
        accept_multiple_files=True,
        help="Upload one or more ARGO profile NetCDF files. The system will process and index them for search and analysis."
    )
    
    if uploaded_files:
        st.success(f"✅ Selected {len(uploaded_files)} files for processing")
        
        # Show file details
        with st.expander("📋 File Details"):
            for file in uploaded_files:
                st.write(f"**{file.name}**")
                st.write(f"Size: {file.size:,} bytes")
        
        # Processing options
        st.header("2. Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quality_flags = st.multiselect(
                "Quality Control Flags",
                options=[1, 2, 3, 4, 5, 8, 9],
                default=[1, 2],
                help="Select which quality flags to include (1=Good, 2=Probably good)"
            )
        
        with col2:
            max_depth = st.number_input(
                "Maximum Depth (dbar)",
                min_value=0.0,
                max_value=6000.0,
                value=2000.0,
                help="Maximum depth to include in processing"
            )
        
        # Export options
        st.subheader("Export Options")
        export_parquet = st.checkbox("Export to Parquet format", value=True)
        generate_summary = st.checkbox("Generate profile summaries for AI search", value=True)
        
        # Process files
        if st.button("🚀 Process Files", type="primary"):
            process_files(uploaded_files, quality_flags, max_depth, export_parquet, generate_summary, config)
    
    # Database status
    st.header("3. Database Status")
    show_database_status(config)
    
    # Sample data generator
    st.header("4. Generate Sample Data")
    if st.button("Generate Sample ARGO Data"):
        generate_sample_data(config)

def process_files(files, quality_flags, max_depth, export_parquet, generate_summary, config):
    """Process uploaded ARGO files"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    # Initialize components
    try:
        db_session = init_database(config.database_uri)
        db_manager = DatabaseManager(db_session)
        processor = ArgoDataProcessor()
        
        if generate_summary:
            vector_store = VectorStore(db_session)
        
        processor.quality_flags = quality_flags
        processor.max_depth = max_depth
        
        results = []
        
        for i, file in enumerate(files):
            status_text.text(f"Processing {file.name}...")
            
            try:
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                
                # Process file
                processed_data = processor.process_file(tmp_path)
                
                # Store in database
                profiles_added = 0
                for profile in processed_data['profiles']:
                    # Add metadata to profile
                    profile.update(processed_data['metadata'])
                    
                    # Insert into database
                    profile_id = db_manager.insert_profile(profile)
                    profiles_added += 1
                    
                    # Generate embeddings if requested
                    if generate_summary and processed_data['summary']:
                        vector_store.store_profile_embedding(profile_id, processed_data['summary'])
                
                # Export to Parquet if requested
                parquet_path = None
                if export_parquet:
                    parquet_path = f"data/processed_{file.name}.parquet"
                    processor.to_parquet(processed_data, parquet_path)
                
                results.append({
                    'file': file.name,
                    'profiles': profiles_added,
                    'parquet': parquet_path,
                    'summary': processed_data['summary']
                })
                
                # Clean up temporary file
                import os
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                results.append({
                    'file': file.name,
                    'error': str(e)
                })
            
            progress_bar.progress((i + 1) / len(files))
        
        # Show results
        status_text.text("✅ Processing complete!")
        
        with results_container:
            st.subheader("📊 Processing Results")
            
            for result in results:
                if 'error' in result:
                    st.error(f"❌ {result['file']}: {result['error']}")
                else:
                    st.success(f"✅ {result['file']}: {result['profiles']} profiles processed")
                    
                    if result.get('summary'):
                        with st.expander(f"Summary for {result['file']}"):
                            st.write(result['summary'])
    
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")

def show_database_status(config):
    """Show current database status"""
    try:
        db_session = init_database(config.database_uri)
        vector_store = VectorStore(db_session)
        stats = vector_store.get_profile_statistics()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Profiles", stats.get('total_profiles', 0))
            
            with col2:
                st.metric("With Embeddings", stats.get('profiles_with_embeddings', 0))
            
            with col3:
                coverage = stats.get('embedding_coverage', 0)
                st.metric("Coverage", f"{coverage:.1%}")
            
            with col4:
                st.metric("Database", "🟢 Connected")
            
            # Geographic distribution
            if stats.get('latitude_range'):
                st.subheader("📍 Geographic Distribution")
                lat_range = stats['latitude_range']
                lon_range = stats['longitude_range']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Latitude Range:** {lat_range['min']:.2f}° to {lat_range['max']:.2f}°")
                    st.write(f"**Average Latitude:** {lat_range['avg']:.2f}°")
                
                with col2:
                    st.write(f"**Longitude Range:** {lon_range['min']:.2f}° to {lon_range['max']:.2f}°")
                    st.write(f"**Average Longitude:** {lon_range['avg']:.2f}°")
        else:
            st.info("No profiles found in database. Upload some ARGO files to get started!")
            
    except Exception as e:
        st.error(f"Cannot connect to database: {str(e)}")

def generate_sample_data(config):
    """Generate sample ARGO data for testing"""
    st.info("🧪 Generating sample ARGO profile data...")
    
    try:
        db_session = init_database(config.database_uri)
        db_manager = DatabaseManager(db_session)
        vector_store = VectorStore(db_session)
        
        # Generate 5 sample profiles
        sample_profiles = []
        
        for i in range(5):
            # Random location
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            
            # Random date within last year
            base_date = datetime.now() - timedelta(days=365)
            profile_date = base_date + timedelta(days=np.random.randint(0, 365))
            
            # Generate synthetic depth profile
            depths = np.linspace(0, 2000, 100)
            temperatures = 20 * np.exp(-depths/1000) + 2 + np.random.normal(0, 0.5, len(depths))
            salinities = 34 + 1 * np.exp(-depths/1500) + np.random.normal(0, 0.1, len(depths))
            
            profile = {
                'float_id': f'TEST_{1000 + i}',
                'cycle_number': np.random.randint(1, 100),
                'profile_date': profile_date,
                'latitude': lat,
                'longitude': lon,
                'temperature_data': temperatures.tolist(),
                'salinity_data': salinities.tolist(),
                'pressure_data': depths.tolist(),
                'temp_qc': [1] * len(depths),
                'sal_qc': [1] * len(depths),
                'institution': 'TEST_INSTITUTION',
                'platform_type': 'APEX',
                'data_mode': 'R',
                'summary': f"Test ARGO profile from float TEST_{1000 + i} at location {lat:.2f}°N, {lon:.2f}°E on {profile_date.strftime('%Y-%m-%d')}. Temperature range: {temperatures.min():.1f}°C to {temperatures.max():.1f}°C. Salinity range: {salinities.min():.2f} to {salinities.max():.2f} PSU."
            }
            
            # Insert profile
            profile_id = db_manager.insert_profile(profile)
            
            # Generate embedding
            vector_store.store_profile_embedding(profile_id, profile['summary'])
            
            sample_profiles.append(profile)
        
        st.success(f"✅ Generated {len(sample_profiles)} sample profiles!")
        
        # Show sample data
        with st.expander("📋 Sample Data Details"):
            for i, profile in enumerate(sample_profiles):
                st.write(f"**Profile {i+1}:** Float {profile['float_id']}")
                st.write(f"Location: {profile['latitude']:.2f}°N, {profile['longitude']:.2f}°E")
                st.write(f"Date: {profile['profile_date'].strftime('%Y-%m-%d')}")
                st.write(f"Temperature range: {min(profile['temperature_data']):.1f}°C to {max(profile['temperature_data']):.1f}°C")
                st.write("---")
        
    except Exception as e:
        st.error(f"Error generating sample data: {str(e)}")

if __name__ == "__main__":
    main()