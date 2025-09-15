# 🌊 ARGO Ocean Data Analysis Platform

A comprehensive platform for analyzing ARGO oceanographic data using AI-powered retrieval-augmented generation (RAG) pipelines and interactive visualizations.

## 🎯 System Overview

This platform implements all your specified requirements:

### ✅ Core Features Implemented

1. **ARGO NetCDF Data Ingestion**
   - Process ARGO NetCDF files with xarray
   - Convert to structured formats (SQL/Parquet)
   - Quality control and data validation
   - Metadata extraction and summary generation

2. **Vector Database Storage**
   - PostgreSQL with pgvector extension
   - Semantic embeddings for profile summaries
   - Efficient similarity search capabilities
   - Metadata and measurement data storage

3. **RAG Pipeline with Model Context Protocol (MCP)**
   - Groq/LLaMA integration for natural language queries
   - Query analysis and intent extraction
   - Automatic SQL generation from natural language
   - Context-aware response generation

4. **Interactive Dashboards**
   - Streamlit-based web interface
   - Profile visualizations (T-S plots, depth profiles)
   - Geographic trajectory mapping with Folium
   - Depth-time series analysis
   - Multi-profile comparisons

## 🏗️ Architecture

```
argo_streamlit_app/
├── app.py                 # Main Streamlit application
├── docker-compose.yml     # PostgreSQL + pgvector setup
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py         # Configuration management
│   ├── database.py       # Database models and operations
│   ├── argo_processor.py # NetCDF file processing
│   ├── vector_store.py   # Vector embeddings and search
│   ├── rag_pipeline.py   # RAG implementation with MCP
│   └── visualizations.py # Plotly/Folium visualizations
└── pages/                # Additional Streamlit pages
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to project
cd argo_streamlit_app

# Copy environment file
cp env_example .env

# Update .env with your API keys
```

### 2. Database Setup

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Verify database is running
docker ps
```

### 3. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

### 4. Run the Application

```bash
# Start Streamlit app
streamlit run app.py
```

## 📊 Usage Examples

### Data Upload
1. Navigate to "Data Upload" page
2. Select ARGO NetCDF files
3. Click "Process Files" to ingest and index data

### Natural Language Queries
```
"Show me temperature profiles in the North Atlantic from 2023"
"Find salinity data between 30°N and 60°N latitude"
"What are the average temperatures at 100m depth?"
"Compare temperature profiles from different seasons"
```

### Visualizations
- **Profile Plots**: T-S profiles with depth
- **Trajectory Maps**: Geographic float paths
- **Depth-Time Series**: Contour plots over time
- **Profile Comparisons**: Multi-profile overlays
- **T-S Diagrams**: Temperature-salinity relationships

## 🔧 Configuration

### Environment Variables (.env)
```
DATABASE_URI=postgresql://postgres:postgres@localhost:5432/vectordb
GROQ_API=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_here (optional)
VECTOR_DIMENSION=384
MAX_DEPTH=2000.0
```

### Database Schema
```sql
-- ARGO profiles table
CREATE TABLE argo_profiles (
    id SERIAL PRIMARY KEY,
    float_id VARCHAR(50),
    cycle_number INTEGER,
    profile_date TIMESTAMP,
    latitude FLOAT,
    longitude FLOAT,
    temperature_data JSON,
    salinity_data JSON,
    pressure_data JSON,
    temp_qc INTEGER[],
    sal_qc INTEGER[],
    embedding vector(384),
    summary TEXT,
    institution VARCHAR(100),
    platform_type VARCHAR(50)
);
```

## 🤖 RAG Pipeline Details

### Query Processing Flow
1. **Intent Analysis**: Extract query type, constraints, and requirements
2. **Context Retrieval**: Vector similarity search with filters
3. **SQL Generation**: Convert natural language to database queries
4. **Data Execution**: Run queries and retrieve results
5. **Answer Generation**: Synthesize response with context

### Model Context Protocol Integration
- Query interpretation using Groq's LLaMA models
- Structured prompt engineering for oceanographic data
- SQL generation with schema awareness
- Multi-modal response generation

## 📈 System Capabilities

### Data Processing
- **File Formats**: NetCDF (ARGO standard)
- **Quality Control**: Automatic flagging and filtering
- **Scalability**: Batch processing support
- **Export**: Parquet format for analytics

### Search & Retrieval
- **Semantic Search**: Vector similarity matching
- **Spatial Filtering**: Geographic bounding boxes
- **Temporal Filtering**: Date range constraints
- **Hybrid Queries**: Combined vector and SQL filtering

### Visualization Types
- **Scientific Plots**: Plotly-based interactive charts
- **Geographic Maps**: Folium with trajectory overlays
- **Statistical Dashboards**: Multi-panel analytics
- **Comparative Analysis**: Side-by-side profile views

## 🔍 Current Implementation Status

### ✅ Completed
- Core architecture and module structure
- Database schema with pgvector integration
- ARGO NetCDF processing pipeline
- Vector embedding and storage system
- RAG pipeline with query analysis
- Streamlit interface framework
- Comprehensive visualization suite

### 🚧 Implementation Notes
- **API Keys**: Update .env with your Groq API key
- **Data**: Upload ARGO NetCDF files through the interface
- **Testing**: Use provided sample queries to test RAG pipeline
- **Scaling**: Current setup handles moderate datasets; scale database for production

### 🎯 Next Steps
1. Add real ARGO NetCDF files for testing
2. Configure API keys in environment
3. Test query pipeline with sample data
4. Customize visualizations for specific use cases
5. Add user authentication for production deployment

## 📚 Technical Stack

- **Backend**: Python, SQLAlchemy, pgvector
- **Frontend**: Streamlit
- **Database**: PostgreSQL with vector extensions
- **AI/ML**: Groq, Sentence Transformers, LangChain
- **Visualization**: Plotly, Folium
- **Data Processing**: xarray, pandas, numpy
- **Deployment**: Docker Compose

## 🤝 Contributing

The platform is designed for extensibility:
- Add new visualization types in `src/visualizations.py`
- Extend RAG capabilities in `src/rag_pipeline.py`
- Add data sources in `src/argo_processor.py`
- Customize UI components in `app.py` and `pages/`

## 🔐 Security Considerations

- Store API keys in environment variables
- Use connection pooling for database access
- Implement input validation for user queries
- Consider rate limiting for API calls

---

**🌊 Ready to explore ARGO ocean data with AI-powered analysis!**
