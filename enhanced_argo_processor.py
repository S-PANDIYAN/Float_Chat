"""
Enhanced ARGO Float Data Processor with Comprehensive Oceanographic Features
Calculates advanced features for LLM and RAG system enhancement
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import os
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedArgoProcessor:
    """Enhanced ARGO data processor with comprehensive oceanographic feature extraction"""
    
    def __init__(self, database_uri: str = None, ollama_url: str = "http://localhost:11434"):
        self.database_uri = database_uri or os.getenv('DATABASE_URI', 'postgresql://postgres:postgres@localhost:5432/vectordb')
        self.ollama_url = ollama_url
        self.embedding_model = "embeddinggemma"
        
    def extract_comprehensive_features(self, profile_data: Dict) -> Dict:
        """Extract all comprehensive ARGO features for enhanced LLM/RAG analysis"""
        
        features = {}
        
        # Basic profile info
        features['platform_id'] = profile_data.get('float_id', 'unknown')
        features['cycle_number'] = profile_data.get('cycle_number', 0)
        features['latitude'] = float(profile_data.get('latitude', 0))
        features['longitude'] = float(profile_data.get('longitude', 0))
        features['region'] = profile_data.get('region', 'Unknown')
        
        # Temporal features
        profile_date = profile_data.get('profile_date', datetime.now())
        if isinstance(profile_date, str):
            profile_date = pd.to_datetime(profile_date)
        features['timestamp'] = profile_date
        features['month'] = profile_date.month
        features['year'] = profile_date.year
        
        # Extract measurements from summary
        summary = profile_data.get('summary', '')
        temp_data, sal_data, pres_data = self._parse_summary_data(summary)
        
        # Depth Analysis
        if pres_data:
            features.update(self._calculate_depth_features(pres_data))
        else:
            features.update(self._default_depth_features())
            
        # Temperature Analysis
        if temp_data:
            features.update(self._calculate_temperature_features(temp_data))
        else:
            features.update(self._default_temperature_features())
            
        # Salinity Analysis
        if sal_data:
            features.update(self._calculate_salinity_features(sal_data))
        else:
            features.update(self._default_salinity_features())
            
        # Data Quality Analysis
        features.update(self._calculate_quality_features(temp_data, sal_data, pres_data))
        
        # Oceanographic Features
        features.update(self._calculate_oceanographic_features(
            features['latitude'], features['longitude'], 
            temp_data, sal_data, pres_data
        ))
        
        # Metadata
        features.update(self._calculate_metadata_features(profile_data))
        
        return features
    
    def _parse_summary_data(self, summary: str) -> Tuple[List, List, List]:
        """Parse temperature, salinity, and pressure data from summary"""
        temp_data, sal_data, pres_data = [], [], []
        
        try:
            lines = summary.split('\\n')
            for line in lines:
                if 'Temperature:' in line:
                    # Extract temperature range
                    temp_part = line.split('Temperature:')[1].strip()
                    if 'to' in temp_part:
                        temps = temp_part.split('to')
                        temp_data = [float(temps[0].strip().split()[0]), 
                                   float(temps[1].strip().split()[0])]
                elif 'Salinity:' in line:
                    # Extract salinity range  
                    sal_part = line.split('Salinity:')[1].strip()
                    if 'to' in sal_part:
                        sals = sal_part.split('to')
                        sal_data = [float(sals[0].strip().split()[0]),
                                  float(sals[1].strip().split()[0])]
                elif 'Depth:' in line:
                    # Extract depth range
                    depth_part = line.split('Depth:')[1].strip()
                    if 'to' in depth_part:
                        depths = depth_part.split('to')
                        pres_data = [float(depths[0].strip().split()[0]),
                                   float(depths[1].strip().split()[0])]
        except Exception as e:
            logger.warning(f"Error parsing summary data: {e}")
            
        return temp_data, sal_data, pres_data
    
    def _calculate_depth_features(self, pres_data: List) -> Dict:
        """Calculate comprehensive depth-related features"""
        if len(pres_data) >= 2:
            min_depth = min(pres_data)
            max_depth = max(pres_data)
            avg_depth = np.mean(pres_data)
            
            # Estimate mixed layer depth (simplified)
            mixed_layer_depth = min_depth + (max_depth - min_depth) * 0.1
            
            return {
                'min_depth': min_depth,
                'max_depth': max_depth,
                'average_depth': avg_depth,
                'mixed_layer_depth': mixed_layer_depth
            }
        return self._default_depth_features()
    
    def _default_depth_features(self) -> Dict:
        return {
            'min_depth': 0.0,
            'max_depth': 2000.0,
            'average_depth': 1000.0,
            'mixed_layer_depth': 50.0
        }
    
    def _calculate_temperature_features(self, temp_data: List) -> Dict:
        """Calculate comprehensive temperature features"""
        if len(temp_data) >= 2:
            min_temp = min(temp_data)
            max_temp = max(temp_data)
            avg_temp = np.mean(temp_data)
            
            # Determine temperature trend
            trend = 'stable'
            if max_temp - min_temp > 2.0:
                trend = 'decreasing' if temp_data[0] > temp_data[-1] else 'increasing'
                
            # Estimate potential temperature (simplified)
            potential_temp = avg_temp - 0.1  # Simplified calculation
            
            return {
                'average_temperature': avg_temp,
                'min_temperature': min_temp,
                'max_temperature': max_temp,
                'trend_temperature': trend,
                'potential_temperature': potential_temp
            }
        return self._default_temperature_features()
    
    def _default_temperature_features(self) -> Dict:
        return {
            'average_temperature': 15.0,
            'min_temperature': 2.0,
            'max_temperature': 25.0,
            'trend_temperature': 'stable',
            'potential_temperature': 14.9
        }
    
    def _calculate_salinity_features(self, sal_data: List) -> Dict:
        """Calculate comprehensive salinity features"""
        if len(sal_data) >= 2:
            min_sal = min(sal_data)
            max_sal = max(sal_data)
            avg_sal = np.mean(sal_data)
            
            # Determine salinity trend
            trend = 'stable'
            if max_sal - min_sal > 0.5:
                trend = 'decreasing' if sal_data[0] > sal_data[-1] else 'increasing'
                
            return {
                'average_salinity': avg_sal,
                'min_salinity': min_sal,
                'max_salinity': max_sal,
                'trend_salinity': trend
            }
        return self._default_salinity_features()
    
    def _default_salinity_features(self) -> Dict:
        return {
            'average_salinity': 35.0,
            'min_salinity': 34.0,
            'max_salinity': 36.0,
            'trend_salinity': 'stable'
        }
    
    def _calculate_quality_features(self, temp_data: List, sal_data: List, pres_data: List) -> Dict:
        """Calculate data quality metrics"""
        
        # Count missing values (simplified)
        total_expected = 100  # Assume 100 measurements expected
        actual_measurements = len(temp_data) + len(sal_data) + len(pres_data)
        missing_count = max(0, total_expected - actual_measurements)
        
        # Estimate measurement noise (simplified)
        noise = 0.1 if actual_measurements > 50 else 0.2
        
        # Outlier detection (simplified)
        outliers = 0
        if len(temp_data) > 2:
            temp_std = np.std(temp_data)
            outliers += sum(1 for t in temp_data if abs(t - np.mean(temp_data)) > 2 * temp_std)
            
        # Anomaly score (0-1, where 1 is most anomalous)
        anomaly_score = min(1.0, (missing_count / total_expected) * 0.5 + (outliers / max(1, len(temp_data))) * 0.5)
        
        return {
            'missing_values_count': missing_count,
            'measurement_noise': noise,
            'outliers_detected': outliers,
            'anomaly_score': anomaly_score
        }
    
    def _calculate_oceanographic_features(self, lat: float, lon: float, 
                                        temp_data: List, sal_data: List, pres_data: List) -> Dict:
        """Calculate oceanographic and geographic features"""
        
        # Estimate density (simplified)
        if temp_data and sal_data:
            avg_temp = np.mean(temp_data)
            avg_sal = np.mean(sal_data)
            # Simplified density calculation
            density = 1000 + (avg_sal - 35) * 0.8 - (avg_temp - 4) * 0.2
        else:
            density = 1025.0
            
        # Determine pattern type based on depth and temperature
        pattern_type = 'mixed_layer'
        if pres_data and len(pres_data) > 1:
            max_depth = max(pres_data)
            if max_depth > 1000:
                pattern_type = 'deep_water'
            elif temp_data and len(temp_data) > 1 and max(temp_data) - min(temp_data) > 5:
                pattern_type = 'thermocline'
                
        # Ocean feature detection (simplified)
        ocean_feature = 'normal'
        if abs(lat) > 60:
            ocean_feature = 'polar_water'
        elif abs(lat) < 30 and temp_data and np.mean(temp_data) > 25:
            ocean_feature = 'tropical_water'
        elif temp_data and sal_data and np.std(temp_data) > 3 and np.std(sal_data) > 0.5:
            ocean_feature = 'front'
            
        # Estimate distance to coast (simplified)
        distance_to_coast = min(500, abs(lat) * 111 + abs(lon) * 85)  # Rough estimate in km
        
        return {
            'density': density,
            'pattern_type': pattern_type,
            'ocean_feature': ocean_feature,
            'distance_to_coast': distance_to_coast
        }
    
    def _calculate_metadata_features(self, profile_data: Dict) -> Dict:
        """Calculate metadata and usage features"""
        
        return {
            'source_file': profile_data.get('source_file', 'unknown.nc'),
            'data_acquisition_method': 'autonomous_float',
            'standards_compliance': 'ARGO',
            'query_frequency': 0,
            'saved_profile': False,
            'preferred_units': 'metric',
            'nearest_floats': []  # Will be calculated later
        }
    
    def generate_enhanced_embedding(self, features: Dict) -> List[float]:
        """Generate embedding from comprehensive features"""
        
        # Create rich text description
        text_description = self._create_feature_description(features)
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text_description
                },
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json().get('embedding', [])
                if len(embedding) == 768:
                    return embedding
                    
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            
        # Return zero vector if failed
        return [0.0] * 768
    
    def _create_feature_description(self, features: Dict) -> str:
        """Create rich text description from features for embedding"""
        
        desc = f"""ARGO Float Profile Analysis:
Platform: {features.get('platform_id', 'unknown')} Cycle: {features.get('cycle_number', 0)}
Location: {features.get('latitude', 0):.2f}°N, {features.get('longitude', 0):.2f}°E
Region: {features.get('region', 'Unknown')} Date: {features.get('year', 2025)}-{features.get('month', 1):02d}

Depth Profile: {features.get('min_depth', 0):.1f}m to {features.get('max_depth', 2000):.1f}m
Average Depth: {features.get('average_depth', 1000):.1f}m
Mixed Layer Depth: {features.get('mixed_layer_depth', 50):.1f}m

Temperature: {features.get('min_temperature', 2):.1f}°C to {features.get('max_temperature', 25):.1f}°C
Average Temperature: {features.get('average_temperature', 15):.1f}°C
Temperature Trend: {features.get('trend_temperature', 'stable')}
Potential Temperature: {features.get('potential_temperature', 14.9):.1f}°C

Salinity: {features.get('min_salinity', 34):.1f} to {features.get('max_salinity', 36):.1f} PSU
Average Salinity: {features.get('average_salinity', 35):.1f} PSU
Salinity Trend: {features.get('trend_salinity', 'stable')}

Water Properties:
Density: {features.get('density', 1025):.1f} kg/m³
Pattern Type: {features.get('pattern_type', 'mixed_layer')}
Ocean Feature: {features.get('ocean_feature', 'normal')}
Distance to Coast: {features.get('distance_to_coast', 100):.0f} km

Data Quality:
Missing Values: {features.get('missing_values_count', 0)}
Measurement Noise: {features.get('measurement_noise', 0.1):.2f}
Outliers: {features.get('outliers_detected', 0)}
Anomaly Score: {features.get('anomaly_score', 0):.3f}

Standards: {features.get('standards_compliance', 'ARGO')}
Acquisition: {features.get('data_acquisition_method', 'autonomous_float')}"""

        return desc
    
    def process_and_store_enhanced_profile(self, profile_data: Dict) -> bool:
        """Process profile with enhanced features and store in database"""
        
        try:
            # Extract comprehensive features
            features = self.extract_comprehensive_features(profile_data)
            
            # Generate enhanced embedding
            embedding = self.generate_enhanced_embedding(features)
            
            # Store in database
            engine = create_engine(self.database_uri)
            
            # Prepare update SQL with all enhanced features
            update_sql = """
            UPDATE argo_profiles SET
                min_depth = :min_depth,
                max_depth = :max_depth,
                average_depth = :average_depth,
                mixed_layer_depth = :mixed_layer_depth,
                average_temperature = :average_temperature,
                min_temperature = :min_temperature,
                max_temperature = :max_temperature,
                trend_temperature = :trend_temperature,
                potential_temperature = :potential_temperature,
                average_salinity = :average_salinity,
                min_salinity = :min_salinity,
                max_salinity = :max_salinity,
                trend_salinity = :trend_salinity,
                missing_values_count = :missing_values_count,
                measurement_noise = :measurement_noise,
                outliers_detected = :outliers_detected,
                anomaly_score = :anomaly_score,
                density = :density,
                pattern_type = :pattern_type,
                ocean_feature = :ocean_feature,
                distance_to_coast = :distance_to_coast,
                month = :month,
                year = :year,
                timestamp = :timestamp,
                source_file = :source_file,
                data_acquisition_method = :data_acquisition_method,
                standards_compliance = :standards_compliance,
                query_frequency = :query_frequency,
                saved_profile = :saved_profile,
                preferred_units = :preferred_units,
                profile_embedding = :embedding
            WHERE float_id = :float_id AND cycle_number = :cycle_number
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(update_sql), {
                    **features,
                    'embedding': embedding,
                    'float_id': features['platform_id'],
                })
                conn.commit()
                
            logger.info(f"✅ Enhanced profile {features['platform_id']} cycle {features['cycle_number']}")
            return result.rowcount > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to process enhanced profile: {e}")
            return False

def enhance_existing_profiles():
    """Enhance all existing profiles with comprehensive features"""
    
    processor = EnhancedArgoProcessor()
    engine = create_engine(processor.database_uri)
    
    print("🚀 Starting comprehensive ARGO profile enhancement...")
    
    try:
        # Get all existing profiles
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT float_id, cycle_number, latitude, longitude, region, 
                       profile_date, summary
                FROM argo_profiles
                ORDER BY float_id, cycle_number
            """))
            
            profiles = result.fetchall()
            
        print(f"📊 Found {len(profiles)} profiles to enhance")
        
        enhanced_count = 0
        for profile in profiles:
            profile_data = {
                'float_id': profile[0],
                'cycle_number': profile[1], 
                'latitude': profile[2],
                'longitude': profile[3],
                'region': profile[4],
                'profile_date': profile[5],
                'summary': profile[6]
            }
            
            if processor.process_and_store_enhanced_profile(profile_data):
                enhanced_count += 1
                
            if enhanced_count % 10 == 0:
                print(f"   ✅ Enhanced {enhanced_count}/{len(profiles)} profiles")
                
        print(f"🎉 Successfully enhanced {enhanced_count} profiles with comprehensive features!")
        return True
        
    except Exception as e:
        print(f"❌ Enhancement failed: {e}")
        return False

if __name__ == "__main__":
    enhance_existing_profiles()