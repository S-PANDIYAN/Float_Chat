"""
Database models and operations for ARGO float data with 768-dimensional vector storage.
Only processes real ARGO NetCDF files - no demo/test data.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

class ArgoProfile(Base):
    '''ARGO profile with 768-dimensional vector storage'''
    __tablename__ = 'argo_profiles'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # ARGO float metadata (from NetCDF files)
    float_id = Column(String(50), nullable=False)
    profile_date = Column(String(20))
    cycle_number = Column(Integer, nullable=False)
    
    # Geographic coordinates
    latitude = Column(Float, nullable=False)  
    longitude = Column(Float, nullable=False)
    
    # Oceanographic data (JSON format for flexibility)
    temperature_data = Column(JSON)
    salinity_data = Column(JSON)
    pressure_data = Column(JSON)
    
    # 768-dimensional vector embedding (embeddinggemma)
    embedding = Column(Vector(768))
    
    # Processed information
    summary = Column(Text)
    region = Column(String(100))
    
    # Quality and metadata
    data_quality = Column(String(20))
    processing_date = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    '''Database manager for ARGO profiles with vector operations'''
    
    def __init__(self, database_uri: Optional[str] = None):
        self.database_uri = database_uri or os.getenv("DATABASE_URI")
        if not self.database_uri:
            raise ValueError("DATABASE_URI not found in environment variables")
        
        self.engine = create_engine(self.database_uri)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("Database connection established")
    
    def store_argo_profile(self, profile_data: Dict, embedding: List[float]) -> int:
        '''Store ARGO profile data with 768-dimensional embedding'''
        session = self.Session()
        try:
            profile = ArgoProfile(
                float_id=profile_data['float_id'],
                profile_date=profile_data.get('profile_date'),
                cycle_number=profile_data.get('cycle_number', 1),
                latitude=profile_data['latitude'],
                longitude=profile_data['longitude'],
                temperature_data=profile_data.get('temperature_data'),
                salinity_data=profile_data.get('salinity_data'),
                pressure_data=profile_data.get('pressure_data'),
                embedding=embedding,  # 768-dimensional vector
                summary=profile_data.get('summary'),
                region=profile_data.get('region'),
                data_quality=profile_data.get('data_quality', 'processed')
            )
            
            session.add(profile)
            session.commit()
            profile_id = profile.id
            logger.info(f"Stored ARGO profile {profile_data['float_id']} with 768-dim vector")
            return profile_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing profile: {e}")
            raise
        finally:
            session.close()
    
    def search_similar_profiles(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        '''Search for similar ARGO profiles using 768-dimensional vector similarity'''
        session = self.Session()
        try:
            # Use pgvector cosine distance for similarity search
            query = session.query(
                ArgoProfile.id,
                ArgoProfile.float_id,
                ArgoProfile.profile_date,
                ArgoProfile.latitude,
                ArgoProfile.longitude,
                ArgoProfile.summary,
                ArgoProfile.region,
                ArgoProfile.embedding.cosine_distance(query_embedding).label('distance')
            ).filter(
                ArgoProfile.embedding.isnot(None)
            ).order_by(
                ArgoProfile.embedding.cosine_distance(query_embedding)
            ).limit(limit)
            
            results = []
            for row in query.all():
                similarity = 1 - row.distance  # Convert distance to similarity
                results.append({
                    'id': row.id,
                    'float_id': row.float_id,
                    'profile_date': row.profile_date,
                    'latitude': row.latitude,
                    'longitude': row.longitude,
                    'summary': row.summary,
                    'region': row.region,
                    'similarity': similarity
                })
            
            logger.info(f"Found {len(results)} similar profiles")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
        finally:
            session.close()
    
    def get_database_stats(self) -> Dict:
        '''Get database statistics'''
        session = self.Session()
        try:
            total_profiles = session.query(ArgoProfile).count()
            profiles_with_vectors = session.query(ArgoProfile).filter(
                ArgoProfile.embedding.isnot(None)
            ).count()
            unique_floats = session.query(ArgoProfile.float_id).distinct().count()
            
            return {
                'total_profiles': total_profiles,
                'profiles_with_vectors': profiles_with_vectors,
                'unique_floats': unique_floats
            }
        finally:
            session.close()

# Convenience functions for external use
def init_database(database_uri: Optional[str] = None) -> DatabaseManager:
    '''Initialize database connection'''
    return DatabaseManager(database_uri)

def store_argo_profile(profile_data: Dict, embedding: List[float], 
                      db: Optional[DatabaseManager] = None) -> int:
    '''Store ARGO profile with embedding'''
    if db is None:
        db = DatabaseManager()
    return db.store_argo_profile(profile_data, embedding)

def search_similar_argo(query_embedding: List[float], limit: int = 5,
                       db: Optional[DatabaseManager] = None) -> List[Dict]:
    '''Search for similar ARGO profiles'''
    if db is None:
        db = DatabaseManager()
    return db.search_similar_profiles(query_embedding, limit)

def get_database_stats(db: Optional[DatabaseManager] = None) -> Dict:
    '''Get database statistics'''
    if db is None:
        db = DatabaseManager()
    return db.get_database_stats()
