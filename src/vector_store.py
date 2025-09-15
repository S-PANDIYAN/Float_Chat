import numpy as np
from typing import List, Dict, Optional, Tuple
import requests
import json
from src.database import DatabaseManager, ArgoProfile
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector storage and retrieval using pgvector with Ollama embeddings"""
    
    def __init__(self, session_factory, ollama_model: str = "embeddinggemma", ollama_url: str = "http://localhost:11434"):
        self.db_manager = DatabaseManager(session_factory)
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.embedding_dim = 768  # Dimension for embeddinggemma (may vary)
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text using Ollama"""
        try:
            # Make request to Ollama embeddings API
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.ollama_model,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                
                if not embedding:
                    raise ValueError("Empty embedding returned from Ollama")
                
                # Update embedding dimension if needed
                if len(embedding) != self.embedding_dim:
                    self.embedding_dim = len(embedding)
                    logger.info(f"Updated embedding dimension to {self.embedding_dim}")
                
                return embedding
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            raise Exception(f"Ollama connection failed: {e}")
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def store_profile_embedding(self, profile_id: int, summary_text: str) -> bool:
        """Generate and store embedding for ARGO profile"""
        try:
            # Generate embedding
            embedding = self.generate_embedding(summary_text)
            
            # Update profile with embedding
            with self.db_manager.session_factory() as session:
                profile = session.query(ArgoProfile).filter(
                    ArgoProfile.id == profile_id
                ).first()
                
                if profile:
                    profile.embedding = embedding
                    profile.summary = summary_text
                    session.commit()
                    return True
                else:
                    logger.warning(f"Profile {profile_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Error storing embedding for profile {profile_id}: {e}")
            return False
    
    def similarity_search(self, query: str, limit: int = 10, 
                         filters: Optional[Dict] = None) -> List[Dict]:
        """Perform semantic similarity search"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Perform vector search
            with self.db_manager.session_factory() as session:
                query_obj = session.query(ArgoProfile)
                
                # Apply filters if provided
                if filters:
                    if 'lat_range' in filters:
                        lat_min, lat_max = filters['lat_range']
                        query_obj = query_obj.filter(
                            ArgoProfile.latitude.between(lat_min, lat_max)
                        )
                    
                    if 'lon_range' in filters:
                        lon_min, lon_max = filters['lon_range']
                        query_obj = query_obj.filter(
                            ArgoProfile.longitude.between(lon_min, lon_max)
                        )
                    
                    if 'date_range' in filters:
                        start_date, end_date = filters['date_range']
                        query_obj = query_obj.filter(
                            ArgoProfile.profile_date.between(start_date, end_date)
                        )
                
                # Order by similarity and limit results
                profiles = query_obj.order_by(
                    ArgoProfile.embedding.cosine_distance(query_embedding)
                ).limit(limit).all()
                
                # Convert to dictionaries
                results = []
                for profile in profiles:
                    result = {
                        'id': profile.id,
                        'float_id': profile.float_id,
                        'cycle_number': profile.cycle_number,
                        'profile_date': profile.profile_date,
                        'latitude': profile.latitude,
                        'longitude': profile.longitude,
                        'summary': profile.summary,
                        'temperature_data': profile.temperature_data,
                        'salinity_data': profile.salinity_data,
                        'pressure_data': profile.pressure_data,
                        'institution': profile.institution,
                        'platform_type': profile.platform_type
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def hybrid_search(self, query: str, sql_filters: Optional[str] = None, 
                     limit: int = 10) -> List[Dict]:
        """Combine vector similarity with SQL filtering"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            with self.db_manager.session_factory() as session:
                # Build base query
                query_obj = session.query(ArgoProfile)
                
                # Apply SQL filters if provided
                if sql_filters:
                    # This would need more sophisticated SQL parsing
                    # For now, just basic filtering
                    pass
                
                # Order by similarity
                profiles = query_obj.order_by(
                    ArgoProfile.embedding.cosine_distance(query_embedding)
                ).limit(limit).all()
                
                return [self._profile_to_dict(p) for p in profiles]
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_similar_profiles(self, profile_id: int, limit: int = 5) -> List[Dict]:
        """Find profiles similar to a given profile"""
        try:
            with self.db_manager.session_factory() as session:
                # Get the reference profile
                ref_profile = session.query(ArgoProfile).filter(
                    ArgoProfile.id == profile_id
                ).first()
                
                if not ref_profile or not ref_profile.embedding:
                    return []
                
                # Find similar profiles
                similar_profiles = session.query(ArgoProfile).filter(
                    ArgoProfile.id != profile_id
                ).order_by(
                    ArgoProfile.embedding.cosine_distance(ref_profile.embedding)
                ).limit(limit).all()
                
                return [self._profile_to_dict(p) for p in similar_profiles]
                
        except Exception as e:
            logger.error(f"Error finding similar profiles: {e}")
            return []
    
    def _profile_to_dict(self, profile: ArgoProfile) -> Dict:
        """Convert ArgoProfile to dictionary"""
        return {
            'id': profile.id,
            'float_id': profile.float_id,
            'cycle_number': profile.cycle_number,
            'profile_date': profile.profile_date,
            'latitude': profile.latitude,
            'longitude': profile.longitude,
            'summary': profile.summary,
            'temperature_data': profile.temperature_data,
            'salinity_data': profile.salinity_data,
            'pressure_data': profile.pressure_data,
            'temp_qc': profile.temp_qc,
            'sal_qc': profile.sal_qc,
            'institution': profile.institution,
            'data_mode': profile.data_mode,
            'platform_type': profile.platform_type
        }
    
    def get_profile_statistics(self) -> Dict:
        """Get statistics about stored profiles"""
        try:
            with self.db_manager.session_factory() as session:
                total_profiles = session.query(ArgoProfile).count()
                
                profiles_with_embeddings = session.query(ArgoProfile).filter(
                    ArgoProfile.embedding.isnot(None)
                ).count()
                
                # Geographic distribution
                lat_stats = session.query(
                    session.query(ArgoProfile.latitude).func.min(),
                    session.query(ArgoProfile.latitude).func.max(),
                    session.query(ArgoProfile.latitude).func.avg()
                ).first()
                
                lon_stats = session.query(
                    session.query(ArgoProfile.longitude).func.min(),
                    session.query(ArgoProfile.longitude).func.max(),
                    session.query(ArgoProfile.longitude).func.avg()
                ).first()
                
                return {
                    'total_profiles': total_profiles,
                    'profiles_with_embeddings': profiles_with_embeddings,
                    'embedding_coverage': profiles_with_embeddings / total_profiles if total_profiles > 0 else 0,
                    'latitude_range': {
                        'min': lat_stats[0],
                        'max': lat_stats[1],
                        'avg': lat_stats[2]
                    },
                    'longitude_range': {
                        'min': lon_stats[0],
                        'max': lon_stats[1],
                        'avg': lon_stats[2]
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting profile statistics: {e}")
            return {}
    
    def reindex_all_profiles(self) -> bool:
        """Regenerate embeddings for all profiles that have summaries"""
        try:
            with self.db_manager.session_factory() as session:
                profiles = session.query(ArgoProfile).filter(
                    ArgoProfile.summary.isnot(None),
                    ArgoProfile.embedding.is_(None)
                ).all()
                
                logger.info(f"Reindexing {len(profiles)} profiles")
                
                for profile in profiles:
                    try:
                        embedding = self.generate_embedding(profile.summary)
                        profile.embedding = embedding
                        session.commit()
                    except Exception as e:
                        logger.error(f"Error reindexing profile {profile.id}: {e}")
                        continue
                
                logger.info("Reindexing completed")
                return True
                
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            return False