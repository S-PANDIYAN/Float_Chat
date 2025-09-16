# 🌊 ARGO Analytics Platform - Complete Setup Guide

## 🚀 Running the Project from Scratch

This guide will walk you through setting up and running the complete ARGO Ocean Data Analytics Platform from the beginning.

### 📋 Prerequisites

1. **Python 3.8+** with pip
2. **Docker Desktop** for Windows
3. **Ollama** (for local embeddings)
4. **Groq API Key** (free tier available)

### 🛠️ Step-by-Step Setup

#### Step 1: Environment Setup
```bash
# Navigate to project directory
cd C:\Users\Pandiyan\argo_streamlit_app

# Install Python dependencies
pip install -r requirements.txt
```

#### Step 2: Start Docker Desktop
```bash
# Start Docker Desktop (if not running)
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Wait for Docker to start, then launch database
docker-compose up -d
```

#### Step 3: Configure Environment Variables
Ensure your `.env` file has:
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

#### Step 4: Start Ollama Services
```bash
# Start Ollama service
ollama serve

# In another terminal, pull the embedding model
ollama pull embeddinggemma:latest
```

#### Step 5: Process Your ARGO Data
```bash
# Process your NetCDF files and store in database
python complete_netcdf_processor.py
```

#### Step 6: Verify System Health
```bash
# Run system health check
python check_system.py
```

#### Step 7: Launch the Application
```bash
# Start the Streamlit frontend
streamlit run app.py
```

#### Step 8: Access Your Platform
Open your browser to: **http://localhost:8501**

### 🔧 Quick Setup Script

For automated setup, run:
```bash
python setup_from_scratch.py
```

### 📊 What You'll Get

- **78 ARGO Profiles** from your NetCDF data
- **AI-Powered Queries** using natural language
- **Vector Search** with 768-dimensional embeddings
- **Real-time Visualizations** and analytics
- **Professional Interface** with system monitoring

### 🚨 Troubleshooting

**Docker Issues:**
```bash
# Check Docker status
docker ps

# Restart containers
docker-compose down && docker-compose up -d
```

**Database Issues:**
```bash
# Check database connection
python -c "from src.database import DatabaseManager; print(DatabaseManager().get_database_stats())"
```

**Ollama Issues:**
```bash
# Check Ollama status
ollama list

# Test embedding generation
ollama run embeddinggemma "test"
```

**Port Conflicts:**
- Streamlit: http://localhost:8501
- Database: localhost:5432
- Ollama: localhost:11434

### 📝 Sample Queries to Try

Once running, try these queries:
- "What temperature data do we have from the Indian Ocean?"
- "Show me salinity profiles from January 2023"
- "Where are the ARGO floats located?"
- "What are the depth ranges in our data?"

### 🎯 Success Indicators

✅ **System Check Passes:** All components green  
✅ **Database Connected:** 78 profiles loaded  
✅ **Embeddings Working:** 768-dimensional vectors  
✅ **AI Responses:** Groq LLM generating answers  
✅ **Frontend Loading:** Professional interface visible  

---

**Your ARGO Analytics Platform is now ready for oceanographic data analysis!** 🌊