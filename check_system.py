"""
System Configuration Checker for ARGO Analytics Platform
Verifies all components are properly configured before running the main application.
"""

import os
import sys
import requests
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

def check_environment_variables():
    """Check if all required environment variables are set"""
    print("🔧 Checking Environment Variables...")
    
    required_vars = {
        'DATABASE_URI': os.getenv('DATABASE_URI'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'OLLAMA_BASE_URL': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        'OLLAMA_EMBEDDING_MODEL': os.getenv('OLLAMA_EMBEDDING_MODEL', 'embeddinggemma:latest')
    }
    
    all_good = True
    for var, value in required_vars.items():
        if value:
            print(f"  ✅ {var}: {value[:20]}...")
        else:
            print(f"  ❌ {var}: Not set")
            all_good = False
    
    return all_good

def check_database_connection():
    """Check PostgreSQL database connection"""
    print("\n🗄️ Checking Database Connection...")
    
    try:
        from src.database import DatabaseManager
        db = DatabaseManager()
        stats = db.get_database_stats()
        
        print(f"  ✅ Database connected successfully")
        print(f"  📊 Total profiles: {stats['total_profiles']}")
        print(f"  🎯 Profiles with vectors: {stats['profiles_with_vectors']}")
        print(f"  🏷️ Unique floats: {stats['unique_floats']}")
        return True
        
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        print(f"  💡 Try: docker-compose up -d")
        return False

def check_ollama_connection():
    """Check Ollama embedding service"""
    print("\n🤖 Checking Ollama Embedding Service...")
    
    try:
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        model_name = os.getenv('OLLAMA_EMBEDDING_MODEL', 'embeddinggemma:latest')
        
        # Test embedding generation
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=10
        )
        
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            if len(embedding) == 768:
                print(f"  ✅ Ollama connected: {model_name}")
                print(f"  📏 Embedding dimensions: {len(embedding)}")
                return True
            else:
                print(f"  ❌ Wrong embedding dimensions: {len(embedding)} (expected 768)")
                return False
        else:
            print(f"  ❌ Ollama request failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Ollama connection failed: {e}")
        print(f"  💡 Make sure Ollama is running with embeddinggemma model")
        return False

def check_groq_api():
    """Check Groq API connection"""
    print("\n🧠 Checking Groq API...")
    
    try:
        from groq import Groq
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("  ❌ GROQ_API_KEY not set")
            return False
        
        client = Groq(api_key=api_key)
        
        # Test API call
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.1-8b-instant",
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            print("  ✅ Groq API connected successfully")
            print(f"  🎯 Model: llama-3.1-8b-instant")
            return True
        else:
            print("  ❌ Groq API response empty")
            return False
            
    except Exception as e:
        print(f"  ❌ Groq API connection failed: {e}")
        return False

def check_docker_status():
    """Check if Docker containers are running"""
    print("\n🐳 Checking Docker Status...")
    
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            # Check for our specific container name or pgvector image
            if 'argo-pgvector-db' in output or ('pgvector' in output and 'Up' in output):
                print("  ✅ PostgreSQL with pgvector container is running")
                # Also show container details
                lines = output.split('\n')
                for line in lines:
                    if 'pgvector' in line or 'argo-pgvector-db' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            status = 'healthy' if 'healthy' in line else ('Up' if 'Up' in line else 'Unknown')
                            print(f"  📊 Container: {parts[-1]} - Status: {status}")
                return True
            else:
                print("  ⚠️ ARGO PostgreSQL container not found")
                print("  💡 Run: docker-compose up -d")
                return False
        else:
            print("  ❌ Docker command failed")
            return False
            
    except FileNotFoundError:
        print("  ❌ Docker not installed or not in PATH")
        return False
    except Exception as e:
        print(f"  ❌ Docker check failed: {e}")
        return False

def main():
    """Run all system checks"""
    print("🌊 ARGO Analytics Platform - System Check")
    print("=" * 50)
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("Docker Status", check_docker_status),
        ("Database Connection", check_database_connection),
        ("Ollama Embedding Service", check_ollama_connection),
        ("Groq API", check_groq_api)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ {name} check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 System Check Summary:")
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All systems ready! You can run the application with:")
        print("   streamlit run app.py")
    else:
        print("⚠️ Some components need attention before running the application.")
        print("   Fix the failed checks above, then run this script again.")

if __name__ == "__main__":
    main()