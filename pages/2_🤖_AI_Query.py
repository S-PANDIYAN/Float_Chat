import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import load_config
from src.database import init_database
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="AI Query Interface",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 AI-Powered Query Interface")
    st.markdown("Ask questions about ARGO oceanographic data in natural language")
    
    # Initialize components
    config = load_config()
    
    try:
        db_session = init_database(config.database_uri)
        vector_store = VectorStore(db_session)
        rag_pipeline = RAGPipeline(config.groq_api_key, vector_store)
        
        # Query interface
        st.header("💬 Ask Your Question")
        
        # Query suggestions
        with st.expander("💡 Example Queries"):
            suggestions = rag_pipeline.get_query_suggestions()
            for suggestion in suggestions[:5]:
                if st.button(suggestion, key=f"suggestion_{suggestion[:20]}"):
                    st.session_state.query_input = suggestion
        
        # Main query input
        query = st.text_area(
            "Enter your question about ARGO data:",
            value=st.session_state.get('query_input', ''),
            placeholder="e.g., 'Show me temperature profiles in the North Atlantic from 2023 with temperatures below 10°C'",
            height=100,
            key="main_query"
        )
        
        # Query options
        col1, col2 = st.columns(2)
        
        with col1:
            include_sql = st.checkbox("Show generated SQL query", value=True)
            include_context = st.checkbox("Show retrieved context", value=False)
        
        with col2:
            max_results = st.slider("Maximum results", 1, 20, 5)
            response_detail = st.selectbox("Response detail", ["Concise", "Detailed", "Technical"])
        
        # Execute query
        if st.button("🔍 Submit Query", type="primary") and query.strip():
            with st.spinner("🤔 Processing your query..."):
                try:
                    # Process query through RAG pipeline
                    response = rag_pipeline.process_query(query)
                    
                    # Display results
                    display_results(response, include_sql, include_context, max_results)
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info("Please check your API keys and database connection.")
        
        # Query history
        show_query_history()
        
        # Query analytics
        show_query_analytics(vector_store)
        
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.info("Please ensure the database is running and API keys are configured.")

def display_results(response, include_sql, include_context, max_results):
    """Display query results"""
    
    # Main answer
    st.header("🎯 Answer")
    if response.get('answer'):
        st.markdown(response['answer'])
    else:
        st.warning("No answer generated. Please try rephrasing your question.")
    
    # Generated SQL query
    if include_sql and response.get('sql_query'):
        st.header("🔍 Generated SQL Query")
        st.code(response['sql_query'], language='sql')
        
        # Query explanation
        with st.expander("📖 Query Explanation"):
            explain_sql_query(response['sql_query'])
    
    # Retrieved context
    if include_context and response.get('context'):
        st.header("📚 Retrieved Context")
        context = response['context'][:max_results]
        
        for i, profile in enumerate(context):
            with st.expander(f"Profile {i+1}: Float {profile.get('float_id', 'Unknown')}"):
                display_profile_context(profile)
    
    # Data results
    if response.get('data'):
        st.header("📊 Data Results")
        data = response['data']
        
        if isinstance(data, list) and len(data) > 0:
            # Convert to DataFrame if possible
            try:
                if isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="argo_query_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.write("Data preview:")
                    for i, item in enumerate(data[:max_results]):
                        st.write(f"{i+1}. {item}")
            except Exception as e:
                st.write("Raw data:")
                st.json(data[:max_results])
        else:
            st.info("No data results returned.")
    
    # Query analysis
    if response.get('query_analysis'):
        with st.expander("🔬 Query Analysis"):
            st.json(response['query_analysis'])

def explain_sql_query(sql_query):
    """Provide explanation for generated SQL query"""
    explanations = []
    
    if "SELECT" in sql_query.upper():
        explanations.append("• **SELECT**: Retrieves specific columns from the database")
    
    if "WHERE" in sql_query.upper():
        explanations.append("• **WHERE**: Applies filters based on your query conditions")
    
    if "JOIN" in sql_query.upper():
        explanations.append("• **JOIN**: Combines data from multiple related tables")
    
    if "ORDER BY" in sql_query.upper():
        explanations.append("• **ORDER BY**: Sorts results in a specific order")
    
    if "LIMIT" in sql_query.upper():
        explanations.append("• **LIMIT**: Restricts the number of results returned")
    
    if explanations:
        st.markdown("\n".join(explanations))
    else:
        st.info("This query performs a basic data retrieval operation.")

def display_profile_context(profile):
    """Display individual profile context"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Float ID:** {profile.get('float_id', 'N/A')}")
        st.write(f"**Cycle:** {profile.get('cycle_number', 'N/A')}")
        st.write(f"**Date:** {profile.get('profile_date', 'N/A')}")
        st.write(f"**Institution:** {profile.get('institution', 'N/A')}")
    
    with col2:
        st.write(f"**Latitude:** {profile.get('latitude', 'N/A')}")
        st.write(f"**Longitude:** {profile.get('longitude', 'N/A')}")
        st.write(f"**Platform:** {profile.get('platform_type', 'N/A')}")
        st.write(f"**Data Mode:** {profile.get('data_mode', 'N/A')}")
    
    if profile.get('summary'):
        st.write("**Summary:**")
        st.write(profile['summary'])
    
    # Data preview
    if profile.get('temperature_data'):
        temp_data = profile['temperature_data']
        st.write(f"**Temperature:** {min(temp_data):.2f}°C to {max(temp_data):.2f}°C ({len(temp_data)} measurements)")
    
    if profile.get('salinity_data'):
        sal_data = profile['salinity_data']
        st.write(f"**Salinity:** {min(sal_data):.2f} to {max(sal_data):.2f} PSU ({len(sal_data)} measurements)")

def show_query_history():
    """Show recent query history"""
    st.header("📜 Recent Queries")
    
    # Initialize session state for query history
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Add current query to history if submitted
    if st.session_state.get('main_query') and st.session_state.get('query_submitted'):
        query = st.session_state['main_query']
        if query not in st.session_state.query_history:
            st.session_state.query_history.insert(0, query)
            # Keep only last 10 queries
            st.session_state.query_history = st.session_state.query_history[:10]
    
    # Display history
    if st.session_state.query_history:
        for i, past_query in enumerate(st.session_state.query_history[:5]):
            with st.expander(f"Query {i+1}: {past_query[:50]}..."):
                st.write(past_query)
                if st.button(f"Run Again", key=f"rerun_{i}"):
                    st.session_state.query_input = past_query
                    st.experimental_rerun()
    else:
        st.info("No previous queries. Start asking questions about ARGO data!")

def show_query_analytics(vector_store):
    """Show query and database analytics"""
    st.header("📈 System Analytics")
    
    try:
        stats = vector_store.get_profile_statistics()
        
        if stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Profiles", 
                    stats.get('total_profiles', 0),
                    help="Total number of ARGO profiles in the database"
                )
            
            with col2:
                st.metric(
                    "Searchable Profiles", 
                    stats.get('profiles_with_embeddings', 0),
                    help="Profiles with AI embeddings for semantic search"
                )
            
            with col3:
                coverage = stats.get('embedding_coverage', 0)
                st.metric(
                    "Search Coverage", 
                    f"{coverage:.1%}",
                    help="Percentage of profiles available for AI search"
                )
            
            # Performance tips
            with st.expander("💡 Query Tips"):
                st.markdown("""
                **For better results:**
                - Be specific about location (e.g., "North Atlantic", "30°N to 60°N")
                - Include time periods (e.g., "from 2023", "last 6 months")
                - Specify measurement types (temperature, salinity, pressure)
                - Use scientific terminology when appropriate
                
                **Example good queries:**
                - "Temperature profiles deeper than 1000m in the Pacific Ocean"
                - "Salinity anomalies in the Mediterranean Sea from 2022-2023"
                - "Compare winter and summer temperature profiles at 45°N"
                """)
        
    except Exception as e:
        st.warning(f"Could not load analytics: {str(e)}")

if __name__ == "__main__":
    main()