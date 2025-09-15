# 🌊 ARGO Ocean Data Analytics Platform

A production-ready AI-powered platform for analyzing real ARGO float data from NetCDF files using RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

### 🎯 Core Functionality
- **Real Data Processing**: Works exclusively with actual ARGO NetCDF files (no demo data)
- **AI-Powered Queries**: Natural language questions about oceanographic data
- **Vector Search**: 768-dimensional embeddings using embeddinggemma model
- **Semantic Retrieval**: Find relevant profiles using cosine similarity
- **LLM Responses**: Intelligent answers powered by Groq LLaMA-3.1-8b

### 📊 Data Analytics
- **Profile Statistics**: Total profiles, vectorized data, unique floats
- **Regional Distribution**: Geographic breakdown of ARGO float locations
- **Interactive Visualizations**: Charts and graphs for data exploration
- **Raw Data Access**: Optional display of temperature/salinity measurements

### 🔧 Technical Stack
- **Frontend**: Streamlit with custom CSS styling
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: Local Ollama with embeddinggemma:latest model
- **LLM**: Groq API with LLaMA-3.1-8b-instant
- **Containerization**: Docker Compose for database management

## 🚀 Quick Start

### 1. System Check
Run the configuration checker to verify all components:
```bash
python check_system.py
```

### 2. Start Database
Ensure PostgreSQL with pgvector is running:
```bash
docker-compose up -d
```

### 3. Process Your Data
Load your ARGO NetCDF files into the database:
```bash
python complete_netcdf_processor.py
```

### 4. Launch Application
Start the Streamlit frontend:
```bash
streamlit run app.py
```

### 5. Access Platform
Open your browser to: `http://localhost:8501`

## 📁 Project Structure

```
argo_streamlit_app/
├── app.py                    # Main Streamlit frontend application
├── check_system.py           # System configuration checker
├── complete_netcdf_processor.py  # NetCDF data processor
├── .env                      # Environment configuration
├── docker-compose.yml        # Database container setup
├── requirements.txt          # Python dependencies
├── src/
│   └── database.py          # Database models and operations
└── README.md                # This documentation
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Database Configuration
DATABASE_URI=postgresql://postgres:postgres@localhost:5432/vectordb

# Groq API Configuration
GROQ_API_KEY="your_groq_api_key_here"

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest

# Vector Configuration
VECTOR_DIMENSION=768
```

### Required Services
1. **Docker Desktop**: For PostgreSQL container
2. **Ollama**: For local embedding generation
3. **Groq API**: For LLM responses (free tier available)

## 💡 Usage Examples

### Sample Queries
- "What temperature profiles do we have from the Indian Ocean?"
- "Show me salinity data from January 2023"
- "Where are the ARGO floats located?"
- "What are the depth ranges in our data?"
- "Which floats have the most complete temperature data?"

### Query Features
- **Natural Language**: Ask questions in plain English
- **Contextual Answers**: AI responses based on actual data
- **Profile Retrieval**: See relevant ARGO float profiles
- **Similarity Scores**: Understand result relevance
- **Geographic Context**: Location and regional information

## 📊 Interface Components

### Main Dashboard
- **System Status**: Real-time connection monitoring
- **Database Statistics**: Profile counts and regional distribution
- **Query Interface**: Natural language input with advanced options
- **Results Display**: AI answers with supporting data profiles

### Sidebar Information
- **Connection Status**: Database, embedding model, and LLM status
- **Data Metrics**: Profile counts and geographical distribution
- **Model Configuration**: Current AI model settings

### Advanced Options
- **Result Limits**: Control number of retrieved profiles
- **Raw Data Display**: Toggle detailed oceanographic measurements
- **Similarity Thresholds**: Adjust search sensitivity

## 🔍 How It Works

### 1. Data Ingestion
- NetCDF files are processed and stored in PostgreSQL
- Each profile gets temperature, salinity, and pressure data
- Geographic coordinates and metadata are extracted

### 2. Vector Generation
- Profile summaries are converted to 768-dimensional vectors
- Local embeddinggemma model ensures data privacy
- Vectors are stored using pgvector extension

### 3. Query Processing
- User questions are converted to embeddings
- Vector similarity search finds relevant profiles
- Top matching profiles are retrieved with similarity scores

### 4. AI Response
- Retrieved profiles provide context to Groq LLM
- AI generates comprehensive answers about the data
- Responses include specific measurements and locations

## 🛠️ Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Start PostgreSQL container
docker-compose up -d

# Check container status
docker ps
```

**Ollama Embedding Service Offline**
```bash
# Start Ollama service
ollama serve

# Pull embeddinggemma model
ollama pull embeddinggemma:latest
```

**Groq API Errors**
- Verify API key in .env file
- Check Groq account quota/limits
- Ensure internet connectivity

**No Data Found**
```bash
# Process your NetCDF files
python complete_netcdf_processor.py

# Check database contents
python -c "from src.database import DatabaseManager; print(DatabaseManager().get_database_stats())"
```

## 📈 Performance

### Optimization Features
- **Cached Resources**: Streamlit caching for better performance
- **Batch Processing**: Efficient NetCDF file handling
- **Vector Indexing**: Fast similarity searches with pgvector
- **Connection Pooling**: Optimized database connections

### Scalability
- **Local Processing**: No external API calls for embeddings
- **Flexible Limits**: Configurable result counts
- **Resource Monitoring**: Real-time system status display

## 🔐 Security

### Data Privacy
- **Local Embeddings**: No data sent to external services for vectorization
- **Secure Storage**: PostgreSQL with proper authentication
- **API Key Management**: Environment variable configuration

### Best Practices
- Keep API keys in .env file (not version controlled)
- Use Docker for isolated database environment
- Regular backup of processed data

## 📚 Additional Resources

- [ARGO Float Data Format](https://argo.ucsd.edu/data/data-formats/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Ollama Model Library](https://ollama.ai/library)
- [pgvector Documentation](https://github.com/pgvector/pgvector)

## 🤝 Support

For issues or questions:
1. Run `python check_system.py` for diagnosis
2. Check Docker and Ollama service status
3. Verify .env configuration
4. Review application logs in terminal

---

**Built for real oceanographic research with production-grade AI technology** 🌊🔬