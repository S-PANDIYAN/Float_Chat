"""
🌊 ARGO Analytics Platform - Complete Setup From Scratch
Automated script to set up and run the entire platform step by step.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def print_step(step_num, title, description=""):
    """Print formatted step information"""
    print(f"\n{'='*60}")
    print(f"🚀 STEP {step_num}: {title}")
    if description:
        print(f"   {description}")
    print('='*60)

def check_command_exists(command):
    """Check if a command exists in the system"""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def wait_for_service(url, timeout=30, service_name="Service"):
    """Wait for a service to become available"""
    print(f"⏳ Waiting for {service_name} to be ready...")
    for i in range(timeout):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {i+1}/{timeout}...")
        time.sleep(1)
    
    print(f"❌ {service_name} failed to start within {timeout} seconds")
    return False

def main():
    """Run complete setup from scratch"""
    
    print("🌊 ARGO Analytics Platform - Complete Setup")
    print("=" * 60)
    print("This script will set up your entire ARGO analytics platform from scratch.")
    print("Make sure you have your NetCDF files ready!")
    
    # Verify we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: Please run this script from the argo_streamlit_app directory")
        sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Step 1: Check Prerequisites
    print_step(1, "Checking Prerequisites", "Verifying required software is installed")
    
    prerequisites = [
        ("python", "Python"),
        ("docker", "Docker"),
        ("pip", "Pip")
    ]
    
    missing_prereqs = []
    for cmd, name in prerequisites:
        if check_command_exists(cmd):
            print(f"   ✅ {name} is installed")
        else:
            print(f"   ❌ {name} is missing")
            missing_prereqs.append(name)
    
    if missing_prereqs:
        print(f"\n❌ Missing prerequisites: {', '.join(missing_prereqs)}")
        print("Please install them before running this script.")
        sys.exit(1)
    
    # Step 2: Install Python Dependencies
    print_step(2, "Installing Python Dependencies", "Installing required packages from requirements.txt")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Python dependencies installed successfully")
        else:
            print(f"   ⚠️ Some packages may have failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        sys.exit(1)
    
    # Step 3: Start Docker Desktop
    print_step(3, "Starting Docker Desktop", "Launching Docker Desktop and database containers")
    
    try:
        # Try to start Docker Desktop
        subprocess.run([
            "Start-Process", 
            '"C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"'
        ], shell=True, capture_output=True)
        print("   🐳 Docker Desktop startup initiated...")
        
        # Wait a moment for Docker to start
        time.sleep(5)
        
        # Check if Docker is responding
        docker_ready = False
        for i in range(12):  # Wait up to 60 seconds
            try:
                result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
                if result.returncode == 0:
                    docker_ready = True
                    break
            except:
                pass
            print(f"   Waiting for Docker Desktop ({i+1}/12)...")
            time.sleep(5)
        
        if docker_ready:
            print("   ✅ Docker Desktop is ready")
        else:
            print("   ⚠️ Docker Desktop may need more time to start")
            
    except Exception as e:
        print(f"   ⚠️ Could not automatically start Docker Desktop: {e}")
        print("   Please start Docker Desktop manually and press Enter to continue...")
        input()
    
    # Step 4: Start Database Container
    print_step(4, "Starting Database Container", "Launching PostgreSQL with pgvector")
    
    try:
        result = subprocess.run(["docker-compose", "up", "-d"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Database container started successfully")
            time.sleep(3)  # Give container time to initialize
        else:
            print(f"   ❌ Failed to start database: {result.stderr}")
            print("   Please check Docker Desktop is running and try again")
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error starting database: {e}")
        sys.exit(1)
    
    # Step 5: Check Ollama Service
    print_step(5, "Checking Ollama Service", "Verifying embedding model is available")
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma:latest")
    
    # Test Ollama connection
    ollama_ready = False
    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=10
        )
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            if len(embedding) == 768:
                print(f"   ✅ Ollama service ready with {model_name}")
                ollama_ready = True
            else:
                print(f"   ⚠️ Wrong embedding dimensions: {len(embedding)}")
        else:
            print(f"   ⚠️ Ollama responded with status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Ollama service not ready: {e}")
    
    if not ollama_ready:
        print("   💡 Please ensure Ollama is running with embeddinggemma model:")
        print("      ollama serve")
        print("      ollama pull embeddinggemma:latest")
        print("   Press Enter when ready to continue...")
        input()
    
    # Step 6: Verify Environment Configuration
    print_step(6, "Verifying Configuration", "Checking environment variables and API keys")
    
    required_vars = {
        'DATABASE_URI': os.getenv('DATABASE_URI'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'OLLAMA_BASE_URL': os.getenv('OLLAMA_BASE_URL'),
        'OLLAMA_EMBEDDING_MODEL': os.getenv('OLLAMA_EMBEDDING_MODEL')
    }
    
    config_ok = True
    for var, value in required_vars.items():
        if value:
            print(f"   ✅ {var}: {'*' * 10}...{value[-10:] if len(value) > 10 else value}")
        else:
            print(f"   ❌ {var}: Not set")
            config_ok = False
    
    if not config_ok:
        print("   ⚠️ Some environment variables are missing. Check your .env file.")
        print("   Continuing anyway...")
    
    # Step 7: Process ARGO Data (if needed)
    print_step(7, "Processing ARGO Data", "Loading NetCDF files into database")
    
    try:
        # Check if data is already processed
        from src.database import DatabaseManager
        db = DatabaseManager()
        stats = db.get_database_stats()
        
        if stats['total_profiles'] > 0:
            print(f"   ✅ Database already has {stats['total_profiles']} profiles")
            print(f"   📊 {stats['profiles_with_vectors']} profiles with vectors")
            print(f"   🏷️ {stats['unique_floats']} unique floats")
        else:
            print("   📂 No data found. Processing NetCDF files...")
            
            # Look for NetCDF files
            netcdf_files = list(Path().glob("**/*.nc"))
            if netcdf_files:
                print(f"   Found {len(netcdf_files)} NetCDF files")
                print("   🔄 Processing data... This may take a few minutes...")
                
                # Run the complete processor
                result = subprocess.run([sys.executable, "complete_netcdf_processor.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("   ✅ Data processing completed successfully")
                else:
                    print(f"   ⚠️ Data processing had issues: {result.stderr}")
            else:
                print("   ⚠️ No NetCDF files found. Please add your ARGO data files.")
                print("   You can continue and add data later.")
                
    except Exception as e:
        print(f"   ⚠️ Could not process data: {e}")
        print("   You can process data manually later with: python complete_netcdf_processor.py")
    
    # Step 8: Run System Health Check
    print_step(8, "System Health Check", "Verifying all components are working")
    
    try:
        result = subprocess.run([sys.executable, "check_system.py"], 
                              capture_output=True, text=True)
        print("   System Check Results:")
        print(result.stdout)
        
        if "All systems ready!" in result.stdout:
            print("   🎉 All systems are operational!")
        else:
            print("   ⚠️ Some components may need attention")
            
    except Exception as e:
        print(f"   ⚠️ Could not run system check: {e}")
    
    # Step 9: Launch Application
    print_step(9, "Launching Application", "Starting the Streamlit frontend")
    
    print("   🚀 Starting Streamlit application...")
    print("   📱 The app will open at: http://localhost:8501")
    print("   🛑 Press Ctrl+C to stop the application")
    
    try:
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n   👋 Application stopped by user")
    except Exception as e:
        print(f"   ❌ Failed to start application: {e}")
        print("   You can start it manually with: streamlit run app.py")
    
    print("\n" + "="*60)
    print("🎉 ARGO Analytics Platform Setup Complete!")
    print("📚 Check SETUP_GUIDE.md for detailed documentation")
    print("🔧 Run check_system.py anytime to verify system health")
    print("="*60)

if __name__ == "__main__":
    main()