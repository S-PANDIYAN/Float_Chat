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
        """Execute RAG query with comprehensive results"""
        if not user_query.strip():
            return None
            
        # Generate query embedding
        query_embedding = self.get_embedding(user_query)
        if len(query_embedding) != 768:
            st.error("Failed to generate query embedding")
            return None
        
        # Search similar profiles
        try:
            profiles = search_similar_argo(query_embedding, limit=limit, db=self.db)
            if not profiles:
                return {"answer": "No relevant data found for your query.", "profiles": []}
            
            # Generate AI response using Groq
            if self.groq_client:
                context = self._build_context(profiles)
                ai_response = self._generate_ai_response(user_query, context)
            else:
                ai_response = "AI response unavailable - check Groq API configuration"
            
            return {
                "answer": ai_response,
                "profiles": profiles,
                "query_embedding_generated": True,
                "profiles_found": len(profiles)
            }
            
        except Exception as e:
            st.error(f"RAG query failed: {e}")
            return None
    
    def _build_context(self, profiles: List[Dict]) -> str:
        """Build context from retrieved profiles"""
        context = "ARGO Ocean Data Context:\n\n"
        
        for i, profile in enumerate(profiles, 1):
            context += f"Profile {i}:\n"
            context += f"- ARGO Float: {profile['float_id']}\n"
            context += f"- Location: {profile['latitude']:.2f}°N, {profile['longitude']:.2f}°E\n"
            context += f"- Region: {profile['region']}\n"
            context += f"- Date: {profile['profile_date']}\n"
            context += f"- Data Summary: {profile['summary']}\n"
            context += f"- Similarity Score: {profile['similarity']:.3f}\n\n"
        
        return context
    
    def _generate_ai_response(self, query: str, context: str) -> str:
        """Generate AI response using Groq LLM"""
        try:
            prompt = f"""You are an oceanographic data analyst. Answer the user's question based on the ARGO float data provided.

{context}

User Question: {query}

Instructions:
- Provide a comprehensive answer based on the data
- Include specific details like locations, measurements, and dates
- If temperature/salinity data is mentioned, provide ranges and averages
- Be scientific but accessible
- Mention the ARGO float IDs and regions when relevant

Answer:"""

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
            placeholder="What are the temperature profiles in the Northern Indian Ocean?",
            height=100
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            result_limit = st.slider("Number of profiles to retrieve", 1, 10, 3)
            include_raw_data = st.checkbox("Include raw data in results", False)
        
        # Query button
        if st.button("🚀 Analyze Data", type="primary"):
            if user_query.strip():
                with st.spinner("🔄 Processing query..."):
                    result = analytics.query_rag_system(user_query, limit=result_limit)
                    
                    if result:
                        # AI Response
                        st.subheader("🤖 AI Analysis")
                        st.success(result["answer"])
                        
                        # Retrieved profiles
                        if result["profiles"]:
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
                                    
                                    # Raw data option
                                    if include_raw_data and 'temperature_data' in profile:
                                        st.write("**🌡️ Temperature Data Preview:**")
                                        temp_data = profile.get('temperature_data', {})
                                        if temp_data:
                                            st.json(temp_data, expanded=False)
                        else:
                            st.warning("No profiles found matching your query.")
                    else:
                        st.error("Query processing failed. Please check your connections and try again.")
            else:
                st.warning("Please enter a query to analyze your data.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick analysis options
        st.subheader("📝 Quick Queries")
        
        sample_queries = [
            "Show temperature profiles from the Indian Ocean",
            "What salinity data do we have?", 
            "Where are ARGO floats located?",
            "What are the depth ranges in our data?",
            "Show me data from January 2023",
            "What are the temperature variations by region?",
            "Which floats have the most complete data?",
            "Show profiles with highest salinity values"
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

