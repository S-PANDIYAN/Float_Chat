import json
import re
import requests
from typing import Dict, List, Optional, Any
from groq import Groq
from src.vector_store import VectorStore
from src.database import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for ARGO data queries using Ollama + Groq"""
    
    def __init__(self, groq_api_key: str, vector_store: VectorStore, ollama_url: str = "http://localhost:11434"):
        self.client = Groq(api_key=groq_api_key)
        self.vector_store = vector_store
        self.ollama_url = ollama_url
        self.model_name = "llama3-8b-8192"  # Using Groq's LLaMA model
        logger.info("RAG Pipeline initialized with Ollama embeddings + Groq LLM")
        
    def process_query(self, user_query: str, filters: Optional[Dict] = None) -> Dict:
        """Process natural language query using RAG pipeline"""
        try:
            # Step 1: Extract intent and parameters from query
            query_analysis = self._analyze_query(user_query)
            
            # Step 2: Retrieve relevant context from vector store
            context = self._retrieve_context(user_query, filters, query_analysis)
            
            # Step 3: Generate SQL query if needed
            sql_query = None
            if query_analysis.get('needs_sql', False):
                sql_query = self._generate_sql_query(user_query, query_analysis, context)
            
            # Step 4: Execute query and get data
            data_results = self._execute_data_query(sql_query, context)
            
            # Step 5: Generate final answer
            answer = self._generate_answer(user_query, context, data_results, query_analysis)
            
            return {
                'answer': answer,
                'sql_query': sql_query,
                'data': data_results,
                'context': context,
                'query_analysis': query_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'answer': f"I apologize, but I encountered an error processing your query: {str(e)}",
                'error': str(e)
            }
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyze user query to extract intent and parameters"""
        analysis_prompt = f"""
        Analyze this ARGO oceanographic data query and extract key information:
        
        Query: "{query}"
        
        Please identify:
        1. Query type (search, comparison, statistics, visualization, specific_data)
        2. Geographic constraints (latitude/longitude ranges)
        3. Temporal constraints (date ranges)
        4. Data types requested (temperature, salinity, pressure, trajectories)
        5. Statistical operations needed (mean, max, min, trends)
        6. Whether SQL query generation is needed
        7. Visualization requirements
        
        Respond in JSON format:
        {{
            "query_type": "search|comparison|statistics|visualization|specific_data",
            "geographic_filters": {{"lat_range": [min, max], "lon_range": [min, max]}},
            "temporal_filters": {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}},
            "data_types": ["temperature", "salinity", "pressure", "trajectories"],
            "operations": ["mean", "max", "min", "count", "trend"],
            "needs_sql": true/false,
            "needs_visualization": true/false,
            "specific_constraints": ["any other specific requirements"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback analysis
                analysis = self._fallback_query_analysis(query)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return self._fallback_query_analysis(query)
    
    def _fallback_query_analysis(self, query: str) -> Dict:
        """Simple fallback query analysis"""
        query_lower = query.lower()
        
        # Basic keyword detection
        needs_sql = any(word in query_lower for word in ['show', 'find', 'get', 'count', 'average'])
        needs_viz = any(word in query_lower for word in ['plot', 'chart', 'map', 'visualize', 'graph'])
        
        data_types = []
        if 'temperature' in query_lower or 'temp' in query_lower:
            data_types.append('temperature')
        if 'salinity' in query_lower or 'salt' in query_lower:
            data_types.append('salinity')
        if 'pressure' in query_lower or 'depth' in query_lower:
            data_types.append('pressure')
        if 'trajectory' in query_lower or 'path' in query_lower:
            data_types.append('trajectories')
        
        return {
            'query_type': 'search',
            'geographic_filters': {},
            'temporal_filters': {},
            'data_types': data_types,
            'operations': [],
            'needs_sql': needs_sql,
            'needs_visualization': needs_viz,
            'specific_constraints': []
        }
    
    def _retrieve_context(self, query: str, filters: Optional[Dict], 
                         analysis: Dict) -> List[Dict]:
        """Retrieve relevant context from vector store"""
        try:
            # Combine user filters with analysis filters
            combined_filters = filters or {}
            
            if analysis.get('geographic_filters'):
                geo_filters = analysis['geographic_filters']
                if 'lat_range' in geo_filters:
                    combined_filters['lat_range'] = geo_filters['lat_range']
                if 'lon_range' in geo_filters:
                    combined_filters['lon_range'] = geo_filters['lon_range']
            
            if analysis.get('temporal_filters'):
                temp_filters = analysis['temporal_filters']
                if 'start_date' in temp_filters and 'end_date' in temp_filters:
                    combined_filters['date_range'] = [
                        temp_filters['start_date'], 
                        temp_filters['end_date']
                    ]
            
            # Perform similarity search
            context = self.vector_store.similarity_search(
                query, 
                limit=10, 
                filters=combined_filters
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _generate_sql_query(self, user_query: str, analysis: Dict, 
                           context: List[Dict]) -> str:
        """Generate SQL query for data retrieval"""
        try:
            # Build context information
            context_info = self._build_context_info(context)
            
            sql_prompt = f"""
            You are an expert in ARGO oceanographic data and SQL. Generate a SQL query to answer this question:
            
            User Query: "{user_query}"
            
            Query Analysis: {json.dumps(analysis, indent=2)}
            
            Database Schema:
            - Table: argo_profiles
            - Columns: id, float_id, cycle_number, profile_date, latitude, longitude, 
                      temperature_data (JSON), salinity_data (JSON), pressure_data (JSON),
                      temp_qc (ARRAY), sal_qc (ARRAY), institution, data_mode, platform_type
            
            Available Context: {context_info}
            
            Generate a PostgreSQL query that:
            1. Filters data based on the user's requirements
            2. Returns relevant columns
            3. Handles JSON data appropriately
            4. Uses proper geographic and temporal constraints
            5. Includes quality control considerations
            
            Return only the SQL query, no explanation:
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": sql_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL query
            sql_query = re.sub(r'^```sql\s*', '', sql_query)
            sql_query = re.sub(r'\s*```$', '', sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return ""
    
    def _execute_data_query(self, sql_query: Optional[str], 
                           context: List[Dict]) -> Optional[Any]:
        """Execute data query and return results"""
        try:
            if sql_query:
                # Execute SQL query
                with self.vector_store.db_manager.session_factory() as session:
                    result = session.execute(sql_query)
                    data = result.fetchall()
                    return data
            else:
                # Return context data if no SQL query
                return context
                
        except Exception as e:
            logger.error(f"Error executing data query: {e}")
            return context  # Fallback to context
    
    def _generate_answer(self, user_query: str, context: List[Dict], 
                        data_results: Any, analysis: Dict) -> str:
        """Generate final answer using retrieved context and data"""
        try:
            # Prepare context summary
            context_summary = self._summarize_context(context)
            data_summary = self._summarize_data_results(data_results)
            
            answer_prompt = f"""
            You are an expert oceanographer analyzing ARGO float data. Answer this question based on the provided context and data:
            
            User Question: "{user_query}"
            
            Query Analysis: {json.dumps(analysis, indent=2)}
            
            Retrieved Context:
            {context_summary}
            
            Data Results:
            {data_summary}
            
            Please provide a clear, informative answer that:
            1. Directly addresses the user's question
            2. Uses the retrieved data and context
            3. Includes relevant statistics or findings
            4. Mentions data quality and limitations if relevant
            5. Suggests follow-up analyses if appropriate
            
            Keep the answer concise but comprehensive, suitable for both scientists and general users.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error generating the response. Please try rephrasing your question."
    
    def _build_context_info(self, context: List[Dict]) -> str:
        """Build context information string"""
        if not context:
            return "No relevant profiles found."
        
        info_parts = [f"Found {len(context)} relevant profiles:"]
        
        for i, profile in enumerate(context[:5]):  # Show first 5
            info_parts.append(
                f"- Profile {i+1}: Float {profile.get('float_id', 'N/A')}, "
                f"Cycle {profile.get('cycle_number', 'N/A')}, "
                f"Date: {profile.get('profile_date', 'N/A')}, "
                f"Location: ({profile.get('latitude', 'N/A'):.2f}, {profile.get('longitude', 'N/A'):.2f})"
            )
        
        if len(context) > 5:
            info_parts.append(f"... and {len(context) - 5} more profiles")
        
        return "\n".join(info_parts)
    
    def _summarize_context(self, context: List[Dict]) -> str:
        """Summarize context for answer generation"""
        if not context:
            return "No relevant data found."
        
        summaries = []
        for profile in context[:3]:  # Use top 3 profiles
            if profile.get('summary'):
                summaries.append(profile['summary'])
        
        return "\n".join(summaries)
    
    def _summarize_data_results(self, data_results: Any) -> str:
        """Summarize data results for answer generation"""
        if not data_results:
            return "No data results available."
        
        if isinstance(data_results, list):
            if len(data_results) == 0:
                return "Query returned no results."
            else:
                return f"Query returned {len(data_results)} records."
        
        return "Data results available for analysis."
    