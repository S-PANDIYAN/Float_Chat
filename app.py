"""
🌊 ARGO Ocean Data Analytics Platform
Production-ready Streamlit frontend for AI-powered oceanographic data analysis.
Uses actual ARGO NetCDF data - no demo/test data.
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Load environment configuration
from dotenv import load_dotenv
load_dotenv()

# Import project modules
import requests
from groq import Groq
from src.database import DatabaseManager, search_similar_argo

# Configure Streamlit page
st.set_page_config(
    page_title="🌊 ARGO Ocean Analytics", 
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .profile-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ArgoAnalytics:
    """Main class for ARGO data analytics operations"""
    
    def __init__(self):
        self.db = None
        self.groq_client = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database and API connections"""
        try:
            # Set up database connection
            if not os.getenv("DATABASE_URI"):
                os.environ["DATABASE_URI"] = "postgresql://postgres:postgres@localhost:5432/vectordb"
            
            self.db = DatabaseManager()
            
            # Set up Groq API client
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                self.groq_client = Groq(api_key=groq_key)
                
        except Exception as e:
            st.error(f"Connection initialization failed: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
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
            st.error(f"Embedding generation failed: {e}")
        
        return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            if self.db:
                stats = self.db.get_database_stats()
                
                # Get additional regional statistics
                session = self.db.Session()
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
                    stats['regional_distribution'] = {row[0]: row[1] for row in regional_stats}
                finally:
                    session.close()
                
                return stats
                
        except Exception as e:
            st.error(f"Database stats error: {e}")
        
        return {'total_profiles': 0, 'profiles_with_vectors': 0, 'unique_floats': 0, 'regional_distribution': {}}
    
    def query_rag_system(self, user_query: str, limit: int = 3) -> Optional[Dict[str, Any]]:
        """Execute RAG query with intelligent query understanding"""
        if not user_query.strip():
            return None
        
        query_lower = user_query.lower().strip()
        
        # Handle greetings and simple interactions without profile search
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'sup', 'yo']
        if query_lower in greetings or len(query_lower) <= 3:
            ai_response = self._generate_ai_response(user_query, "")
            return {
                "answer": ai_response,
                "profiles": [],
                "query_embedding_generated": False,
                "profiles_found": 0
            }
        
        # Handle help requests without profile search
        if any(word in query_lower for word in ['help', 'what can you do', 'capabilities', 'functions']):
            ai_response = self._generate_ai_response(user_query, "")
            return {
                "answer": ai_response,
                "profiles": [],
                "query_embedding_generated": False,
                "profiles_found": 0
            }
        
        # Check if query needs data retrieval
        needs_data = any(word in query_lower for word in [
            'profile', 'temperature', 'salinity', 'pressure', 'depth', 'measurement',
            'data', 'float', 'region', 'ocean', 'atlantic', 'pacific', 'indian',
            'how many', 'count', 'show', 'find', 'compare', 'analysis', 'range'
        ])
        
        if not needs_data:
            ai_response = self._generate_ai_response(user_query, "")
            return {
                "answer": ai_response,
                "profiles": [],
                "query_embedding_generated": False,
                "profiles_found": 0
            }
        
        # Check if query is asking about wrong year (2023, 2024, etc.) 
        # Our data is from 2025, so don't retrieve profiles for other years
        wrong_years = ['2023', '2024', '2022', '2021', '2020']
        if any(year in query_lower for year in wrong_years):
            ai_response = self._generate_ai_response(user_query, "")
            return {
                "answer": ai_response,
                "profiles": [],
                "query_embedding_generated": False,
                "profiles_found": 0
            }
        
        # Check if it's a counting question - these can be answered without profile retrieval
        if any(word in query_lower for word in ['how many', 'count', 'number of', 'total']):
            ai_response = self._generate_ai_response(user_query, "")
            return {
                "answer": ai_response,
                "profiles": [],
                "query_embedding_generated": False,
                "profiles_found": 0
            }
        
        # Only retrieve profiles for queries that actually need to show profile details
        needs_profile_details = any(word in query_lower for word in [
            'show', 'display', 'list', 'details', 'specific', 'example', 'sample'
        ])
        
        if not needs_profile_details:
            # Answer analytically without retrieving specific profiles
            ai_response = self._generate_ai_response(user_query, "")
            return {
                "answer": ai_response,
                "profiles": [],
                "query_embedding_generated": False,
                "profiles_found": 0
            }
        
        # For queries that specifically need profile details, do the retrieval
        query_embedding = self.get_embedding(user_query)
        if len(query_embedding) != 768:
            st.error("Failed to generate query embedding")
            return None
        
        # Search similar profiles
        try:
            profiles = search_similar_argo(query_embedding, limit=limit, db=self.db)
            
            # Generate AI response using Groq
            if self.groq_client:
                context = self._build_context(profiles) if profiles else ""
                ai_response = self._generate_ai_response(user_query, context)
            else:
                ai_response = "AI response unavailable - check Groq API configuration"
            
            return {
                "answer": ai_response,
                "profiles": profiles or [],
                "query_embedding_generated": True,
                "profiles_found": len(profiles) if profiles else 0
            }
            
        except Exception as e:
            st.error(f"RAG query failed: {e}")
            return None
    
    def _build_context(self, profiles: List[Dict]) -> str:
        """Build focused context from retrieved profiles"""
        if not profiles:
            return "No relevant profiles found."
        
        # Extract key information for analysis
        regions = list(set([p.get('region', 'Unknown') for p in profiles]))
        floats = list(set([p.get('float_id', 'Unknown') for p in profiles]))
        
        # Temperature ranges
        temp_ranges = []
        sal_ranges = [] 
        depth_ranges = []
        
        for profile in profiles:
            # Extract temperature info from summary
            summary = profile.get('summary', '')
            if 'Temperature:' in summary:
                temp_part = summary.split('Temperature:')[1].split('\n')[0]
                temp_ranges.append(temp_part.strip())
            
            # Extract salinity info
            if 'Salinity:' in summary:
                sal_part = summary.split('Salinity:')[1].split('\n')[0]
                sal_ranges.append(sal_part.strip())
                
            # Extract depth info
            if 'Depth:' in summary:
                depth_part = summary.split('Depth:')[1].split('\n')[0]
                depth_ranges.append(depth_part.strip())
        
        context = f"""RELEVANT DATA FOUND:
- Regions represented: {', '.join(regions)}
- ARGO floats involved: {len(floats)} floats ({', '.join(floats[:3])}{'...' if len(floats) > 3 else ''})
- Temperature data: {'; '.join(temp_ranges[:2])}{'...' if len(temp_ranges) > 2 else ''}
- Salinity data: {'; '.join(sal_ranges[:2])}{'...' if len(sal_ranges) > 2 else ''}
- Depth coverage: {'; '.join(depth_ranges[:2])}{'...' if len(depth_ranges) > 2 else ''}
"""
        
        return context
    
    def _generate_ai_response(self, query: str, context: str) -> str:
        """Generate intelligent AI response based on query type"""
        try:
            query_lower = query.lower().strip()
            
            # Handle greetings and simple interactions WITHOUT showing profiles
            greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'sup', 'yo']
            if query_lower in greetings or len(query_lower) <= 3:
                return """Hello! 👋 I'm your ARGO Ocean Data Assistant.

I can help you explore oceanographic data from 2025 including:
🌊 **Data Overview**: "How many profiles do we have?"
🗺️ **Geographic Analysis**: "What regions are covered?"
🌡️ **Temperature Studies**: "Show temperature ranges by region"
🧂 **Salinity Analysis**: "Compare salinity in different oceans"
📊 **Statistical Insights**: "What's the data distribution?"

Just ask me anything about the ARGO float data! What interests you?"""

            # Handle help and capability requests
            if any(word in query_lower for word in ['help', 'what can you do', 'capabilities', 'functions']):
                return """I can analyze ARGO oceanographic data and answer questions like:

📈 **Statistical Queries**: "How many profiles are available?"
🗺️ **Geographic Questions**: "What ocean regions have data?"
🌡️ **Environmental Analysis**: "Temperature ranges in different regions"
🔍 **Data Exploration**: "Show me profiles from the Pacific"

What would you like to explore?"""

            # Check if query needs oceanographic data
            needs_data = any(word in query_lower for word in [
                'profile', 'temperature', 'salinity', 'pressure', 'depth', 'measurement',
                'data', 'float', 'region', 'ocean', 'atlantic', 'pacific', 'indian',
                'how many', 'count', 'show', 'find', 'compare', 'analysis', 'range'
            ])
            
            if not needs_data:
                return f"I understand you're asking about '{query}', but I specialize in ARGO oceanographic data analysis. Could you ask me something about ocean temperature, salinity, float locations, or data measurements?"

            # Get database stats for data-related queries
            db_stats = self.get_database_stats()
            total_profiles = db_stats.get('total_profiles', 0)
            unique_floats = db_stats.get('unique_floats', 0)
            regional_dist = db_stats.get('regional_distribution', {})
            
            dataset_info = f"""
ARGO DATASET SUMMARY (2025):
- Total profiles: {total_profiles}
- Unique floats: {unique_floats}
- Regions: {', '.join([f'{region}({count})' for region, count in regional_dist.items()])}
"""

            # Check if asking about wrong years
            wrong_years = ['2023', '2024', '2022', '2021', '2020']
            if any(year in query_lower for year in wrong_years):
                year_mentioned = next((year for year in wrong_years if year in query_lower), 'that year')
                return f"There is no information about ARGO profiles in {year_mentioned} in the provided dataset. The dataset contains data from 2025 with {total_profiles} profiles from {unique_floats} unique ARGO floats."

            # Create focused prompt based on query type
            if any(word in query_lower for word in ['how many', 'count', 'number of', 'total']):
                prompt = f"""Answer this counting question directly using the dataset summary.

{dataset_info}
Question: {query}

Give the specific number with brief context. Be direct and precise. If asking about a specific year, remember our data is from 2025 only."""

            elif any(word in query_lower for word in ['where', 'location', 'region', 'area', 'geographic']):
                prompt = f"""Answer this geographic question about ARGO data distribution.

{dataset_info}
Question: {query}

Focus on locations, regions, and spatial coverage."""

            elif any(word in query_lower for word in ['show', 'display', 'list', 'profile']) and context:
                prompt = f"""The user wants to see specific profiles. Use the retrieved data below.

{dataset_info}
Retrieved Data: {context}
Question: {query}

Show the requested profile information clearly."""

            else:
                prompt = f"""Answer this oceanographic question using available data.

{dataset_info}
Retrieved Data: {context}
Question: {query}

Provide analytical insights relevant to the question."""

            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI response generation failed: {e}"

# Initialize the analytics system
@st.cache_resource
def get_argo_analytics():
    return ArgoAnalytics()

analytics = get_argo_analytics()

# Main Application UI
def main():
    """Main application interface"""
    
    # Header
    st.markdown('<h1 class="main-header">🌊 ARGO Ocean Data Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Analysis of Real ARGO Float Data from NetCDF Files**")
    
    # Sidebar for system information
    with st.sidebar:
        st.header("📊 System Status")
        
        # Database connection status
        try:
            stats = analytics.get_database_stats()
            
            st.success("✅ Database Connected")
            st.metric("Total Profiles", stats['total_profiles'])
            st.metric("Vectorized Profiles", stats['profiles_with_vectors'])
            st.metric("Unique ARGO Floats", stats['unique_floats'])
            
            # Regional distribution
            if stats['regional_distribution']:
                st.subheader("🌍 Regional Distribution")
                for region, count in stats['regional_distribution'].items():
                    st.write(f"**{region}:** {count} profiles")
                    
        except Exception as e:
            st.error("❌ Database Connection Failed")
            st.error(f"Error: {e}")
            st.info("Start Docker: `docker-compose up -d`")
        
        # Model status
        st.header("🤖 AI Models")
        
        # Embedding model status
        embedding_status = "✅ Connected" if analytics.get_embedding("test") else "❌ Offline"
        st.write(f"**Embedding Model:** {embedding_status}")
        st.write("Model: embeddinggemma:latest")
        st.write("Dimensions: 768")
        
        # LLM status
        llm_status = "✅ Connected" if analytics.groq_client else "❌ No API Key"
        st.write(f"**LLM:** {llm_status}")
        st.write("Model: Groq LLaMA-3.1-8b")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query interface
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        st.subheader("🔍 Query Your ARGO Data")
        
        # Query input
        user_query = st.text_area(
            "Enter your question about oceanographic data:",
            placeholder="What temperature profiles do we have from the 2025 ARGO data?",
            height=100
        )
        
        # Query button
        if st.button("🚀 Analyze Data", type="primary"):
            if user_query.strip():
                with st.spinner("🔄 Processing query..."):
                    result = analytics.query_rag_system(user_query, limit=3)
                    
                    if result:
                        # AI Response
                        st.subheader("🤖 AI Analysis")
                        st.success(result["answer"])
                        
                        # Only show profiles section if profiles were retrieved and relevant
                        if result["profiles"] and result.get("profiles_found", 0) > 0:
                            st.subheader(f"📊 Retrieved Profiles ({len(result['profiles'])})")
                            
                            for i, profile in enumerate(result["profiles"], 1):
                                with st.expander(f"🏷️ Profile {i}: ARGO Float {profile['float_id']} - {profile['region']}", expanded=True):
                                    
                                    # Profile metadata
                                    prof_col1, prof_col2 = st.columns(2)
                                    
                                    with prof_col1:
                                        st.write(f"**📍 Location:** {profile['latitude']:.2f}°N, {profile['longitude']:.2f}°E")
                                        st.write(f"**📅 Date:** {profile['profile_date']}")
                                        st.write(f"**🎯 Similarity:** {profile['similarity']:.3f}")
                                    
                                    with prof_col2:
                                        st.write(f"**🌊 Region:** {profile['region']}")
                                        st.write(f"**🔢 Cycle:** {profile.get('cycle_number', 'N/A')}")
                                        st.write(f"**🆔 Float ID:** {profile['float_id']}")
                                    
                                    # Data summary
                                    st.write("**📋 Data Summary:**")
                                    st.info(profile['summary'])
                        # No warning message when profiles are intentionally not retrieved (greetings, help, etc.)
                    else:
                        st.error("Query processing failed. Please check your connections and try again.")
            else:
                st.warning("Please enter a query to analyze your data.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick analysis options
        st.subheader("📝 Quick Queries")
        
        sample_queries = [
            "How many ARGO profiles are in our 2025 dataset?",
            "What temperature ranges do we have in the data?",
            "Show me salinity measurements from different ocean regions",
            "Where are the ARGO floats located globally?",
            "What are the depth ranges in our measurements?",
            "Which ocean regions have the most data coverage?",
            "Show me profiles with the most complete measurements"
        ]
        
        for query in sample_queries:
            if st.button(f"💡 {query}", key=f"sample_{hash(query)}"):
                st.text_area("Query", value=query, key="auto_query", disabled=True)
        
        # Data overview
        st.subheader("📈 Data Overview")
        
        try:
            stats = analytics.get_database_stats()
            if stats['total_profiles'] > 0:
                # Create a simple visualization
                if stats['regional_distribution']:
                    regions = list(stats['regional_distribution'].keys())
                    counts = list(stats['regional_distribution'].values())
                    
                    fig = px.pie(
                        values=counts, 
                        names=regions, 
                        title="Profiles by Region"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Enable Docker for data visualization")

if __name__ == "__main__":
    main()

