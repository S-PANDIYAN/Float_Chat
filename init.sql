-- Initialize database with pgvector extension and create indexes
-- This script runs automatically when the container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better performance
-- Note: These will be created by SQLAlchemy, but we can optimize them here

-- Example of creating additional indexes for common queries
-- Uncomment and modify as needed based on your query patterns

-- CREATE INDEX IF NOT EXISTS idx_argo_profiles_float_id ON argo_profiles(float_id);
-- CREATE INDEX IF NOT EXISTS idx_argo_profiles_date ON argo_profiles(profile_date);
-- CREATE INDEX IF NOT EXISTS idx_argo_profiles_location ON argo_profiles(latitude, longitude);
-- CREATE INDEX IF NOT EXISTS idx_argo_profiles_embedding ON argo_profiles USING ivfflat (embedding vector_cosine_ops);

-- Set recommended PostgreSQL settings for vector operations
-- ALTER SYSTEM SET shared_preload_libraries = 'vector';
-- ALTER SYSTEM SET max_connections = 200;

-- Create a user for the application (optional, for better security)
-- CREATE USER argo_user WITH PASSWORD 'argo_password';
-- GRANT ALL PRIVILEGES ON DATABASE vectordb TO argo_user;