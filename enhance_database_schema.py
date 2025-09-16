"""
Enhanced ARGO Float Features Database Migration
Adds comprehensive oceanographic analysis features
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def upgrade_database_schema():
    """Add enhanced ARGO float features to the database"""
    
    # Database connection
    db_url = os.getenv('DATABASE_URI', 'postgresql://postgres:postgres@localhost:5432/vectordb')
    engine = create_engine(db_url)
    
    # Enhanced schema with comprehensive ARGO features
    migration_sql = """
    -- Add comprehensive ARGO float analysis features
    
    -- Depth Analysis Features
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS min_depth FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS max_depth FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS average_depth FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS mixed_layer_depth FLOAT;
    
    -- Temperature Analysis Features
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS average_temperature FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS min_temperature FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS max_temperature FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS trend_temperature VARCHAR(20); -- 'increasing', 'decreasing', 'stable'
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS potential_temperature FLOAT;
    
    -- Salinity Analysis Features
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS average_salinity FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS min_salinity FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS max_salinity FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS trend_salinity VARCHAR(20); -- 'increasing', 'decreasing', 'stable'
    
    -- Data Quality Features
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS missing_values_count INTEGER DEFAULT 0;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS measurement_noise FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS outliers_detected INTEGER DEFAULT 0;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS anomaly_score FLOAT;
    
    -- Oceanographic Features
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS density FLOAT;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS pattern_type VARCHAR(50); -- 'thermocline', 'mixed_layer', 'deep_water'
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS ocean_feature VARCHAR(100); -- 'upwelling', 'eddy', 'front', 'normal'
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS distance_to_coast FLOAT; -- km
    
    -- Geographic Context
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS month INTEGER;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS year INTEGER;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP;
    
    -- Metadata and Usage
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS source_file VARCHAR(255);
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS data_acquisition_method VARCHAR(100);
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS standards_compliance VARCHAR(50); -- 'ARGO', 'QC_PASSED', 'DELAYED_MODE'
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS query_frequency INTEGER DEFAULT 0;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS saved_profile BOOLEAN DEFAULT FALSE;
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS preferred_units VARCHAR(20) DEFAULT 'metric';
    
    -- Nearest Neighbors (JSON array of float IDs)
    ALTER TABLE argo_profiles ADD COLUMN IF NOT EXISTS nearest_floats JSONB;
    
    -- Create indexes for enhanced querying
    CREATE INDEX IF NOT EXISTS idx_argo_temperature_trend ON argo_profiles(trend_temperature);
    CREATE INDEX IF NOT EXISTS idx_argo_salinity_trend ON argo_profiles(trend_salinity);
    CREATE INDEX IF NOT EXISTS idx_argo_pattern_type ON argo_profiles(pattern_type);
    CREATE INDEX IF NOT EXISTS idx_argo_ocean_feature ON argo_profiles(ocean_feature);
    CREATE INDEX IF NOT EXISTS idx_argo_anomaly_score ON argo_profiles(anomaly_score);
    CREATE INDEX IF NOT EXISTS idx_argo_year_month ON argo_profiles(year, month);
    CREATE INDEX IF NOT EXISTS idx_argo_depth_range ON argo_profiles(min_depth, max_depth);
    CREATE INDEX IF NOT EXISTS idx_argo_temp_range ON argo_profiles(min_temperature, max_temperature);
    CREATE INDEX IF NOT EXISTS idx_argo_sal_range ON argo_profiles(min_salinity, max_salinity);
    
    -- Update existing records with default values where appropriate
    UPDATE argo_profiles SET 
        standards_compliance = 'ARGO',
        data_acquisition_method = 'autonomous_float',
        preferred_units = 'metric',
        query_frequency = 0,
        saved_profile = FALSE
    WHERE standards_compliance IS NULL;
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(migration_sql))
            conn.commit()
            print("✅ Database schema enhanced successfully!")
            print("📊 Added comprehensive ARGO float analysis features:")
            print("   • Depth analysis (min/max/average/mixed layer)")
            print("   • Temperature trends and statistics")
            print("   • Salinity trends and statistics") 
            print("   • Data quality metrics (noise, outliers, anomalies)")
            print("   • Oceanographic features and patterns")
            print("   • Geographic and temporal context")
            print("   • Metadata and usage tracking")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Enhanced ARGO Features Migration...")
    success = upgrade_database_schema()
    if success:
        print("🎉 Migration completed successfully!")
    else:
        print("💥 Migration failed!")