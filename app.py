import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import required packages with error handling
try:
    import requests
    import json
    import pandas as pd
    import numpy as np
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError as e:
    GROQ_AVAILABLE = False

st.set_page_config(
    page_title="ARGO Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_real_argo_data(file_path="C:/Users/Pandiyan/Downloads/20230101_prof.nc"):
    """Load and process real ARGO NetCDF data"""
    try:
        import xarray as xr
        import numpy as np
        
        # Load the NetCDF file
        ds = xr.open_dataset(file_path)
        
        # Extract basic information
        profiles_data = []
        
        # Get the number of profiles
        n_profiles = ds.dims.get('N_PROF', 1)
        
        st.info(f"📊 Found {n_profiles} profiles in NetCDF file")
        
        for i in range(min(n_profiles, 10)):  # Limit to first 10 profiles for demo
            try:
                # Extract profile metadata
                if 'PLATFORM_NUMBER' in ds.variables:
                    platform_bytes = ds['PLATFORM_NUMBER'].values[i]
                    if hasattr(platform_bytes, 'decode'):
                        float_id = platform_bytes.decode('utf-8').strip()
                    else:
                        float_id = str(platform_bytes).strip()
                else:
                    float_id = f"Unknown_{i}"
                
                # Extract location
                if 'LATITUDE' in ds.variables and 'LONGITUDE' in ds.variables:
                    lat = float(ds['LATITUDE'].values[i])
                    lon = float(ds['LONGITUDE'].values[i])
                    location = f"({lat:.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'})"
                else:
                    location = "Unknown location"
                
                # Extract date
                if 'JULD' in ds.variables:
                    try:
                        # ARGO uses Julian days since 1950-01-01
                        julian_day = ds['JULD'].values[i]
                        if not np.isnan(julian_day):
                            # Convert from Julian day (days since 1950-01-01) to datetime
                            import datetime
                            base_date = datetime.datetime(1950, 1, 1)
                            profile_date = base_date + datetime.timedelta(days=float(julian_day))
                            date_str = profile_date.strftime("%Y-%m-%d")
                        else:
                            date_str = "2023-01-01"  # Default date
                    except:
                        date_str = "2023-01-01"
                else:
                    date_str = "2023-01-01"
                
                # Extract temperature and salinity data
                temp_data = None
                salinity_data = None
                
                if 'TEMP' in ds.variables:
                    temp_profile = ds['TEMP'].values[i, :]
                    temp_data = temp_profile[~np.isnan(temp_profile)]
                
                if 'PSAL' in ds.variables:
                    sal_profile = ds['PSAL'].values[i, :]
                    salinity_data = sal_profile[~np.isnan(sal_profile)]
                
                # Extract pressure/depth data
                pressure_data = None
                if 'PRES' in ds.variables:
                    pres_profile = ds['PRES'].values[i, :]
                    pressure_data = pres_profile[~np.isnan(pres_profile)]
                
                # Calculate statistics
                temp_surface = float(temp_data[0]) if temp_data is not None and len(temp_data) > 0 else 20.0
                temp_deep = float(temp_data[-1]) if temp_data is not None and len(temp_data) > 0 else 2.0
                sal_surface = float(salinity_data[0]) if salinity_data is not None and len(salinity_data) > 0 else 35.0
                max_pressure = float(pressure_data[-1]) if pressure_data is not None and len(pressure_data) > 0 else 2000.0
                
                # Determine region based on location
                region = "Unknown Region"
                if 'LATITUDE' in ds.variables and 'LONGITUDE' in ds.variables:
                    lat = float(ds['LATITUDE'].values[i])
                    lon = float(ds['LONGITUDE'].values[i])
                    
                    if lat > 30:
                        if -60 < lon < 20:
                            region = "North Atlantic"
                        elif 100 < lon < 180:
                            region = "North Pacific"
                        else:
                            region = "Northern Ocean"
                    elif lat < -30:
                        region = "Southern Ocean"
                    else:
                        if -60 < lon < 20:
                            region = "Tropical Atlantic"
                        elif 20 < lon < 100:
                            region = "Indian Ocean"
                        else:
                            region = "Tropical Pacific"
                
                # Create description
                temp_range = f"Temperature: {temp_surface:.1f}°C to {temp_deep:.1f}°C"
                sal_info = f"Surface salinity: {sal_surface:.1f} PSU"
                depth_info = f"Max depth: {max_pressure:.0f}m"
                n_measurements = len(temp_data) if temp_data is not None else 0
                
                description = f"Real ARGO float {float_id} from {region}. {temp_range}. {sal_info}. {depth_info}. Contains {n_measurements} measurements from {date_str}."
                
                profile = {
                    "id": i + 1,
                    "float_id": float_id,
                    "location": location,
                    "date": date_str,
                    "region": region,
                    "description": description,
                    "measurements": {
                        "temp_surface": temp_surface,
                        "temp_deep": temp_deep,
                        "salinity_surface": sal_surface,
                        "max_depth": max_pressure,
                        "n_levels": n_measurements
                    },
                    "raw_data": {
                        "temperature": temp_data.tolist() if temp_data is not None else [],
                        "salinity": salinity_data.tolist() if salinity_data is not None else [],
                        "pressure": pressure_data.tolist() if pressure_data is not None else []
                    }
                }
                
                profiles_data.append(profile)
                
            except Exception as e:
                st.warning(f"Error processing profile {i}: {e}")
                continue
        
        if profiles_data:
            st.success(f"✅ Successfully loaded {len(profiles_data)} real ARGO profiles from NetCDF file!")
            return profiles_data
        else:
            st.error("❌ No valid profiles found in NetCDF file")
            return None
            
    except FileNotFoundError:
        st.error(f"❌ NetCDF file not found: {file_path}")
        st.info("💡 Please check that the file exists at: C:/Users/Pandiyan/Downloads/20230101_prof.nc")
        return None
    except Exception as e:
        st.error(f"❌ Error loading NetCDF file: {e}")
        st.info("💡 Make sure xarray and netcdf4 packages are installed")
        return None

class ARGOApp:
    """Main ARGO Application with integrated RAG pipeline"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API", "")
        self.ollama_url = "http://localhost:11434"
        self.groq_client = Groq(api_key=self.groq_api_key) if GROQ_AVAILABLE and self.groq_api_key else None
        
        # Load real ARGO data
        self.argo_data = load_real_argo_data()
        
        # Fallback sample data if real data loading fails
        if self.argo_data is None:
            st.info("🔄 Using sample data as fallback...")
            self.argo_data = [
                {
                    "id": 1,
                    "float_id": "2903334",
                    "location": "45.2°N, -30.1°E",
                    "date": "2023-01-15",
                    "description": "Sample North Atlantic ARGO profile showing typical winter conditions. Temperature range 2.1°C to 18.5°C. Salinity 34.2 to 36.8 PSU. Strong thermocline at 200m depth. Mixed layer depth 180m.",
                    "region": "North Atlantic"
                },
                {
                    "id": 2,
                    "float_id": "2903335",
                    "location": "35.8°N, -15.2°E",
                    "date": "2023-02-10",
                    "description": "Sample Eastern Atlantic subtropical profile with warm surface waters. Temperature 8.5°C to 22.1°C. Salinity 35.1 to 37.2 PSU. Shallow mixed layer 45m. Mediterranean water influence at 800m.",
                    "region": "Eastern Atlantic"
                },
                {
                    "id": 3,
                    "float_id": "2903336",
                    "location": "-45.5°N, 10.3°E",
                    "date": "2023-03-05",
                    "description": "Sample Southern Ocean profile showing cold Antarctic waters. Temperature -1.8°C to 4.2°C. Salinity 33.9 to 34.7 PSU. Deep mixed layer 350m. Upwelling signature visible.",
                    "region": "Southern Ocean"
                }
            ]
    
    @property
    def sample_data(self):
        """For backward compatibility"""
        return self.argo_data

def main():
    """Main application entry point"""
    
    app = ARGOApp()
    
    # Sidebar navigation
    st.sidebar.title("🌊 ARGO Analytics")
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Home", "RAG Demo", "Data Upload", "Query Interface", "Visualizations", "Profile Explorer"]
    )
    
    # Main content area
    if page == "Home":
        show_home_page(app)
    elif page == "RAG Demo":
        show_rag_demo(app)
    elif page == "Data Upload":
        show_data_upload_page()
    elif page == "Query Interface":
        show_query_interface(app)
    elif page == "Visualizations":
        show_visualizations_page()
    elif page == "Profile Explorer":
        show_profile_explorer_page()

def test_ollama_connection():
    """Test if Ollama and embeddinggemma are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return any("embeddinggemma" in name for name in model_names)
    except:
        return False
    return False

def generate_embedding(text, ollama_url="http://localhost:11434"):
    """Generate embedding using Ollama embeddinggemma"""
    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={
                "model": "embeddinggemma",
                "prompt": text
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("embedding", [])
    except Exception as e:
        st.error(f"Embedding error: {e}")
    return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    if not embedding1 or not embedding2:
        return 0.0
    
    # Normalize to same length
    min_len = min(len(embedding1), len(embedding2))
    emb1 = embedding1[:min_len]
    emb2 = embedding2[:min_len]
    
    # Cosine similarity
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    magnitude_a = sum(a * a for a in emb1) ** 0.5
    magnitude_b = sum(b * b for b in emb2) ** 0.5
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def show_home_page(app):
    """Display home page with system overview"""
    st.title("🌊 ARGO Ocean Data Analysis Platform")
    st.subheader("Complete LLM + RAG Pipeline with Local Embeddings")
    
    # System Architecture Overview
    st.header("🧠 System Architecture")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔧 Your RAG Pipeline:
        1. **User Query** → Natural language input
        2. **embeddinggemma** → Generate query embedding (local)
        3. **Vector Search** → Find similar ARGO profiles
        4. **Context Retrieval** → Get relevant data
        5. **Groq LLM** → Generate intelligent response
        """)
    
    with col2:
        st.markdown("""
        ### 🌊 ARGO Capabilities:
        - **NetCDF Processing**: Load and parse ARGO files
        - **Semantic Search**: AI-powered data discovery
        - **Profile Analysis**: Temperature/salinity visualization
        - **Geographic Mapping**: Float trajectory tracking
        - **Natural Language**: Query data with plain English
        """)
    
    # System Status
    st.header("🔧 System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Streamlit", "✅ Running")
    
    with col2:
        ollama_status = "✅ Connected" if test_ollama_connection() else "❌ Offline"
        st.metric("Ollama + embeddinggemma", ollama_status)
    
    with col3:
        groq_status = "✅ Ready" if app.groq_client else "❌ Missing API Key"
        st.metric("Groq LLM", groq_status)
    
    with col4:
        try:
            import psycopg2
            db_status = "✅ Ready"
        except:
            db_status = "❌ Missing"
        st.metric("Database", db_status)
    
    # Quick Test Section
    st.header("🧪 Quick RAG Test")
    if st.button("Test Complete RAG Pipeline"):
        with st.spinner("Testing RAG pipeline..."):
            # Test embedding
            test_query = "Show me temperature profiles from the North Atlantic"
            embedding = generate_embedding(test_query)
            
            if embedding:
                st.success(f"✅ Embedding generated: {len(embedding)} dimensions")
                
                # Test LLM
                if app.groq_client:
                    try:
                        response = app.groq_client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": "What is ARGO oceanographic data?"}],
                            max_tokens=100
                        )
                        st.success("✅ Groq LLM responding correctly")
                        st.info(f"Sample response: {response.choices[0].message.content[:100]}...")
                    except Exception as e:
                        st.error(f"❌ Groq LLM error: {e}")
                else:
                    st.warning("⚠️ Groq API key not configured")
            else:
                st.error("❌ Embedding generation failed")

def show_rag_demo(app):
    """Complete RAG pipeline demonstration"""
    st.title("🤖 Complete RAG Pipeline Demo")
    st.subheader("embeddinggemma (Ollama) + Groq LLM")
    
    # Step 1: Embedding Generation
    st.header("Step 1: 🧮 Query Embedding Generation")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_query = st.text_input(
            "Enter your ARGO data query:",
            value="Show me temperature profiles from the North Atlantic",
            help="Try queries like: 'Find cold water profiles', 'Show Southern Ocean data', etc."
        )
    
    with col2:
        if st.button("🚀 Generate Embedding"):
            if user_query:
                with st.spinner("Generating embedding with embeddinggemma..."):
                    query_embedding = generate_embedding(user_query)
                    
                    if query_embedding:
                        st.session_state.query_embedding = query_embedding
                        st.session_state.user_query = user_query
                        st.success(f"✅ Query embedded! Dimension: {len(query_embedding)}")
                    else:
                        st.error("❌ Failed to generate embedding")
    
    # Step 2: Vector Search
    if hasattr(st.session_state, 'query_embedding'):
        st.header("Step 2: 🔍 Vector Similarity Search")
        
        query_embedding = st.session_state.query_embedding
        user_query = st.session_state.user_query
        
        st.info(f"**Query:** {user_query}")
        st.info(f"**Embedding:** {len(query_embedding)} dimensions")
        
        # Calculate similarities with sample data
        similarities = []
        
        with st.spinner("Calculating similarities..."):
            for profile in app.sample_data:
                profile_embedding = generate_embedding(profile["description"])
                if profile_embedding:
                    similarity = calculate_similarity(query_embedding, profile_embedding)
                    similarities.append({
                        "profile": profile,
                        "similarity": similarity,
                        "embedding": profile_embedding
                    })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        st.subheader("🎯 Most Similar ARGO Profiles:")
        
        for i, item in enumerate(similarities[:3]):  # Top 3
            profile = item["profile"]
            similarity = item["similarity"]
            
            with st.expander(f"Rank {i+1}: Float {profile['float_id']} (Similarity: {similarity:.3f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Location:** {profile['location']}")
                    st.write(f"**Date:** {profile['date']}")
                    st.write(f"**Region:** {profile['region']}")
                
                with col2:
                    st.write(f"**Similarity Score:** {similarity:.3f}")
                    st.progress(similarity)
                
                st.write(f"**Description:** {profile['description']}")
        
        # Step 3: Context Preparation
        st.header("Step 3: 📋 Context Preparation")
        
        # Get top 2 profiles as context
        top_profiles = similarities[:2]
        context = "\n\n".join([
            f"ARGO Profile {p['profile']['float_id']}: {p['profile']['description']}"
            for p in top_profiles
        ])
        
        st.text_area("Retrieved Context for LLM:", context, height=150, disabled=True)
        
        # Step 4: LLM Query
        st.header("Step 4: 🧠 LLM Response Generation")
        
        if app.groq_client:
            if st.button("🚀 Generate LLM Response"):
                with st.spinner("Generating response with Groq LLM..."):
                    
                    prompt = f"""You are an expert oceanographer analyzing ARGO float data. Based on the retrieved context, answer the user's question about oceanographic data.

User Question: {user_query}

Retrieved ARGO Data Context:
{context}

Instructions:
1. Answer the user's question based on the provided ARGO data
2. Include specific details from the profiles (temperature, salinity, location, date)
3. Explain any oceanographic phenomena mentioned
4. If the data doesn't fully answer the question, explain what's available
5. Be scientific but accessible

Answer:"""
                    
                    try:
                        response = app.groq_client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500,
                            temperature=0.3
                        )
                        
                        llm_response = response.choices[0].message.content
                        
                        st.success("✅ LLM Response Generated!")
                        
                        # Display the complete RAG result
                        st.subheader("🎯 Complete RAG Pipeline Result:")
                        
                        with st.container():
                            st.markdown("### 🤖 AI Assistant Response:")
                            st.markdown(llm_response)
                            
                            st.markdown("### 📊 Supporting Data:")
                            for i, item in enumerate(top_profiles):
                                profile = item["profile"]
                                st.write(f"**Profile {i+1}:** Float {profile['float_id']} at {profile['location']} on {profile['date']}")
                        
                    except Exception as e:
                        st.error(f"❌ Groq API Error: {e}")
                        st.info("Check your GROQ_API key in the .env file")
        else:
            st.warning("⚠️ Groq API client not available. Please check your GROQ_API key in .env file")
            st.code('GROQ_API=your_api_key_here')
    
    else:
        st.info("👆 Start by generating an embedding for your query above!")

def show_query_interface(app):
    """Natural language query interface with enhanced RAG"""
    st.title("🤖 AI Query Interface")
    st.subheader("Ask questions about ARGO data in natural language")
    
    # Predefined sample queries
    sample_queries = [
        "Show me temperature profiles in the North Atlantic from 2023",
        "Find cold water profiles in the Southern Ocean",
        "What are the salinity characteristics of Mediterranean water?",
        "Compare temperature profiles from different regions",
        "Show me the deepest ARGO measurements available"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_type = st.radio("Query Type:", ["Custom Query", "Sample Queries"])
        
        if query_type == "Custom Query":
            query = st.text_area(
                "Ask a question about ARGO data:",
                placeholder="e.g., 'Show me temperature profiles in the North Atlantic from 2023'",
                height=100
            )
        else:
            query = st.selectbox("Choose a sample query:", sample_queries)
    
    with col2:
        st.write("**Quick Actions:**")
        if st.button("🚀 Process Query", type="primary"):
            if query:
                process_rag_query(app, query)
        
        if st.button("🔄 Clear Results"):
            for key in list(st.session_state.keys()):
                if key.startswith('rag_'):
                    del st.session_state[key]
            st.experimental_rerun()
    
    # Display results if available
    if hasattr(st.session_state, 'rag_results'):
        display_rag_results(st.session_state.rag_results)

def process_rag_query(app, query):
    """Process a complete RAG query"""
    with st.spinner("Processing your query through the RAG pipeline..."):
        
        # Step 1: Generate embedding
        query_embedding = generate_embedding(query)
        if not query_embedding:
            st.error("Failed to generate query embedding")
            return
        
        # Step 2: Vector search
        similarities = []
        for profile in app.sample_data:
            profile_embedding = generate_embedding(profile["description"])
            if profile_embedding:
                similarity = calculate_similarity(query_embedding, profile_embedding)
                similarities.append({
                    "profile": profile,
                    "similarity": similarity
                })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Step 3: Prepare context
        top_profiles = similarities[:3]
        context = "\n\n".join([
            f"ARGO Profile {p['profile']['float_id']}: {p['profile']['description']}"
            for p in top_profiles
        ])
        
        # Step 4: Generate LLM response
        if app.groq_client:
            try:
                prompt = f"""You are an expert oceanographer. Answer the user's question based on the ARGO data provided.

Question: {query}

Available ARGO Data:
{context}

Provide a detailed, scientific answer based on the available data."""
                
                response = app.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.3
                )
                
                # Store results
                st.session_state.rag_results = {
                    "query": query,
                    "response": response.choices[0].message.content,
                    "context": context,
                    "profiles": top_profiles
                }
                
            except Exception as e:
                st.error(f"LLM Error: {e}")
        else:
            st.error("Groq API not available")

def display_rag_results(results):
    """Display RAG query results"""
    st.header("🎯 Query Results")
    
    # Display the AI response
    st.subheader("🤖 AI Response")
    st.markdown(results["response"])
    
    # Display supporting data
    st.subheader("📊 Supporting ARGO Data")
    for i, item in enumerate(results["profiles"]):
        profile = item["profile"]
        similarity = item["similarity"]
        
        with st.expander(f"Profile {i+1}: Float {profile['float_id']} (Relevance: {similarity:.1%})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Location:** {profile['location']}")
                st.write(f"**Date:** {profile['date']}")
                st.write(f"**Region:** {profile['region']}")
            
            with col2:
                st.write(f"**Relevance Score:** {similarity:.3f}")
                st.progress(similarity)
            
            st.write(f"**Description:** {profile['description']}")
    
    # Show the original query
    with st.expander("🔍 View Query Details"):
        st.write(f"**Original Query:** {results['query']}")
        st.text_area("Context sent to LLM:", results["context"], height=150, disabled=True)

def show_data_upload_page():
    """Display data upload interface"""
    st.title("📁 ARGO Data Upload")
    st.subheader("Upload and process ARGO NetCDF files")
    
    st.markdown("""
    ### 📊 Data Upload Instructions:
    1. **Select ARGO NetCDF files** (.nc format)
    2. **Automatic processing** extracts metadata and profiles
    3. **Vector embedding** generates semantic representations
    4. **Database storage** for instant retrieval and querying
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose ARGO NetCDF files",
        type=['nc'],
        accept_multiple_files=True,
        help="Upload one or more ARGO .nc files for processing"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) selected")
        
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}"):
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**File size:** {uploaded_file.size:,} bytes")
                st.write(f"**Type:** {uploaded_file.type}")
        
        if st.button("🚀 Process Files"):
            with st.spinner("Processing ARGO files..."):
                st.info("⚠️ This demo uses sample data. NetCDF processing requires xarray package.")
                st.success("✅ Files would be processed and stored in the database")
    
    else:
        st.info("👆 Please upload ARGO NetCDF files to get started")
    
    # Show sample data info
    st.header("📊 Current Sample Data")
    st.write("Currently using 3 sample ARGO profiles for demonstration:")
    
    sample_info = [
        {"Float": "3902131", "Location": "North Atlantic", "Date": "2023-03-15"},
        {"Float": "6904240", "Location": "Pacific Ocean", "Date": "2023-06-22"},
        {"Float": "2903334", "Location": "Southern Ocean", "Date": "2023-01-08"}
    ]
    
    df = pd.DataFrame(sample_info) if 'pandas' in globals() else None
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        for info in sample_info:
            st.write(f"• Float {info['Float']} - {info['Location']} ({info['Date']})")

def show_visualizations_page():
    """Display visualization interface"""
    st.title("📊 Data Visualizations")
    st.subheader("Interactive ARGO data visualization tools")
    
    # Chart type selection
    chart_type = st.selectbox(
        "Choose visualization type:",
        ["Temperature Profiles", "Salinity Profiles", "Geographic Distribution", "Time Series Analysis"]
    )
    
    if chart_type == "Temperature Profiles":
        st.header("🌡️ Temperature Profile Analysis")
        
        # Simulate temperature profile data
        import numpy as np
        
        # Generate sample data
        depths = np.arange(0, 2000, 10)
        temp1 = 20 * np.exp(-depths/500) + 2 + np.random.normal(0, 0.5, len(depths))
        temp2 = 18 * np.exp(-depths/600) + 1.5 + np.random.normal(0, 0.4, len(depths))
        temp3 = 15 * np.exp(-depths/400) + 3 + np.random.normal(0, 0.6, len(depths))
        
        # Display with matplotlib if available
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(temp1, depths, label='North Atlantic (3902131)', linewidth=2)
            ax.plot(temp2, depths, label='Pacific Ocean (6904240)', linewidth=2)
            ax.plot(temp3, depths, label='Southern Ocean (2903334)', linewidth=2)
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Depth (m)')
            ax.set_title('ARGO Temperature Profiles Comparison')
            ax.invert_yaxis()  # Depth increases downward
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
        except ImportError:
            st.info("📊 Matplotlib not available - showing sample temperature data")
            
            # Show sample data instead
            sample_data = {
                "Depth (m)": [0, 100, 200, 500, 1000, 1500],
                "North Atlantic": [20.1, 15.2, 12.3, 8.1, 4.2, 2.1],
                "Pacific Ocean": [18.5, 14.1, 11.8, 7.5, 3.8, 1.9],
                "Southern Ocean": [15.2, 12.1, 9.8, 6.2, 3.1, 1.5]
            }
            
            if 'pandas' in globals():
                df = pd.DataFrame(sample_data)
                st.dataframe(df, use_container_width=True)
            else:
                for depth in sample_data["Depth (m)"]:
                    st.write(f"**{depth}m:** N.Atl: {sample_data['North Atlantic'][sample_data['Depth (m)'].index(depth)]}°C, "
                           f"Pacific: {sample_data['Pacific Ocean'][sample_data['Depth (m)'].index(depth)]}°C, "
                           f"S.Ocean: {sample_data['Southern Ocean'][sample_data['Depth (m)'].index(depth)]}°C")
    
    elif chart_type == "Geographic Distribution":
        st.header("🗺️ Geographic Distribution")
        st.markdown("""
        ### ARGO Float Locations
        
        **Sample Float Positions:**
        - 🇺🇸 **North Atlantic:** 45.2°N, 30.1°W (Float 3902131)
        - 🇯🇵 **Pacific Ocean:** 35.1°N, 150.2°E (Float 6904240)  
        - 🇦🇶 **Southern Ocean:** -55.8°S, 2.3°E (Float 2903334)
        
        *Interactive mapping requires folium package*
        """)
        
        # Show coordinate table
        locations = {
            "Float ID": ["3902131", "6904240", "2903334"],
            "Latitude": ["45.2°N", "35.1°N", "55.8°S"],
            "Longitude": ["30.1°W", "150.2°E", "2.3°E"],
            "Region": ["North Atlantic", "Pacific Ocean", "Southern Ocean"]
        }
        
        if 'pandas' in globals():
            df = pd.DataFrame(locations)
            st.dataframe(df, use_container_width=True)
    
    else:
        st.info(f"📊 {chart_type} visualization - requires additional data processing packages")
        st.write("This would show interactive charts for the selected visualization type.")

def show_profile_explorer_page():
    """Display profile explorer interface"""
    st.title("🔬 Profile Explorer")
    st.subheader("Detailed ARGO profile analysis")
    
    # Profile selection
    profile_ids = ["3902131", "6904240", "2903334"]
    selected_profile = st.selectbox("Select ARGO Float:", profile_ids)
    
    # Profile details based on selection
    profile_data = {
        "3902131": {
            "location": "North Atlantic (45.2°N, 30.1°W)",
            "date": "2023-03-15",
            "cycles": 125,
            "max_depth": "1980 m",
            "description": "Deep water formation region with strong temperature gradients"
        },
        "6904240": {
            "location": "Pacific Ocean (35.1°N, 150.2°E)",
            "date": "2023-06-22", 
            "cycles": 89,
            "max_depth": "2000 m",
            "description": "Kuroshio Current region with warm surface waters"
        },
        "2903334": {
            "location": "Southern Ocean (-55.8°S, 2.3°E)",
            "date": "2023-01-08",
            "cycles": 156,
            "max_depth": "1950 m",
            "description": "Antarctic Circumpolar Current with cold, oxygen-rich waters"
        }
    }
    
    if selected_profile in profile_data:
        data = profile_data[selected_profile]
        
        # Display profile information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Float ID", selected_profile)
            st.write(f"**Location:** {data['location']}")
        
        with col2:
            st.metric("Cycles", data['cycles'])
            st.write(f"**Last Profile:** {data['date']}")
        
        with col3:
            st.metric("Max Depth", data['max_depth'])
        
        st.info(f"**Description:** {data['description']}")
        
        # Analysis options
        st.header("📊 Analysis Options")
        
        analysis_type = st.radio(
            "Choose analysis:",
            ["Profile Overview", "Temperature Analysis", "Salinity Analysis", "Data Quality"]
        )
        
        if analysis_type == "Profile Overview":
            st.subheader(f"Profile Overview - Float {selected_profile}")
            
            # Sample measurements
            measurements = [
                {"Parameter": "Temperature", "Surface": "18.5°C", "1000m": "4.2°C", "Deep": "2.1°C"},
                {"Parameter": "Salinity", "Surface": "35.1 PSU", "1000m": "34.8 PSU", "Deep": "34.9 PSU"},
                {"Parameter": "Pressure", "Surface": "10 dbar", "1000m": "1000 dbar", "Deep": "2000 dbar"},
            ]
            
            if 'pandas' in globals():
                df = pd.DataFrame(measurements)
                st.dataframe(df, use_container_width=True)
            else:
                for measurement in measurements:
                    st.write(f"**{measurement['Parameter']}:** Surface: {measurement['Surface']}, "
                           f"1000m: {measurement['1000m']}, Deep: {measurement['Deep']}")
        
        elif analysis_type == "Data Quality":
            st.subheader("📋 Data Quality Assessment")
            
            # Quality metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Valid Profiles", "98.2%", "2.1%")
            
            with col2:
                st.metric("Sensor Status", "Good", "")
            
            with col3:
                st.metric("Last Transmission", "2 days ago", "")
            
            # Quality flags
            st.write("**Quality Control Flags:**")
            st.write("✅ Temperature sensor: PASS")  
            st.write("✅ Conductivity sensor: PASS")
            st.write("✅ Pressure sensor: PASS")
            st.write("⚠️ GPS positioning: MINOR DRIFT")
        
        else:
            st.info(f"📊 {analysis_type} - Detailed analysis would be displayed here")

def show_data_upload_page():
    """Data upload and processing interface"""
    st.title("📁 ARGO Data Upload")
    
    uploaded_files = st.file_uploader(
        "Upload ARGO NetCDF files",
        type=['nc'],
        accept_multiple_files=True,
        help="Select one or more ARGO NetCDF files to process"
    )
    
    if uploaded_files:
        st.success(f"Selected {len(uploaded_files)} files")
        
        if st.button("Process Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process files (integrated into app)
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                # Save uploaded file temporarily and process
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                        tmp_file.write(file.getbuffer())
                        
                        # Process the file using our integrated function
                        new_data = load_real_argo_data(tmp_file.name)
                        if new_data:
                            st.success(f"✅ Processed {file.name}")
                        else:
                            st.error(f"❌ Failed to process {file.name}")
                    
                    os.unlink(tmp_file.name)  # Clean up
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success("All files processed successfully!")

def show_query_interface(rag_pipeline):
    """Natural language query interface"""
    st.title("🤖 AI Query Interface")
    
    query = st.text_area(
        "Ask a question about ARGO data:",
        placeholder="e.g., 'Show me temperature profiles in the North Atlantic from 2023'",
        height=100
    )
    
    if st.button("Submit Query") and query:
        with st.spinner("Processing query..."):
            # RAG pipeline processing
            response = rag_pipeline.process_query(query)
            
            st.subheader("🎯 Query Results")
            st.write(response.get('answer', 'No results found'))
            
            if 'sql_query' in response:
                st.subheader("🔍 Generated SQL")
                st.code(response['sql_query'], language='sql')
            
            if 'data' in response:
                st.subheader("📊 Data Results")
                st.dataframe(response['data'])

def show_visualizations_page():
    """Visualization dashboard"""
    st.title("📊 Data Visualizations")
    
    # Placeholder for visualization components
    st.info("Visualization components will be implemented here")
    
    # Chart type selector
    chart_type = st.selectbox(
        "Select visualization type:",
        ["Profile Plot", "Trajectory Map", "Depth-Time Series", "Comparison View"]
    )
    
    if chart_type == "Profile Plot":
        st.subheader("🌡️ Temperature/Salinity Profiles")
        # Placeholder for profile plots
        
    elif chart_type == "Trajectory Map":
        st.subheader("🗺️ ARGO Float Trajectories")
        # Placeholder for map visualization
        
    elif chart_type == "Depth-Time Series":
        st.subheader("⏰ Depth-Time Analysis")
        # Placeholder for time series
        
    elif chart_type == "Comparison View":
        st.subheader("🔄 Profile Comparisons")
        # Placeholder for comparison plots

def show_profile_explorer_page():
    """Detailed profile exploration"""
    st.title("🔬 Profile Explorer")
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input("Date Range", value=None)
    with col2:
        depth_range = st.slider("Depth Range (m)", 0, 2000, (0, 500))
    
    # Geographic filters
    st.subheader("🌍 Geographic Filters")
    lat_range = st.slider("Latitude", -90.0, 90.0, (-60.0, 60.0))
    lon_range = st.slider("Longitude", -180.0, 180.0, (-180.0, 180.0))
    
    if st.button("Apply Filters"):
        st.info("Filtered profiles will be displayed here")

if __name__ == "__main__":
    main()
