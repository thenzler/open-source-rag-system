-- Initial Database Setup for RAG System
-- This script creates the database schema and initial data

-- Create database if it doesn't exist (run this as superuser)
-- CREATE DATABASE ragdb;
-- CREATE USER raguser WITH ENCRYPTED PASSWORD 'your_password_here';
-- GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;

-- Connect to ragdb database
\c ragdb;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE processing_status AS ENUM (
    'pending',
    'processing', 
    'completed',
    'failed',
    'cancelled'
);

CREATE TYPE user_role AS ENUM (
    'user',
    'moderator',
    'admin',
    'super_admin'
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    
    -- Account status
    is_active BOOLEAN DEFAULT true NOT NULL,
    is_verified BOOLEAN DEFAULT false NOT NULL,
    is_superuser BOOLEAN DEFAULT false NOT NULL,
    
    -- Roles and permissions
    roles TEXT[] DEFAULT ARRAY['user'] NOT NULL,
    permissions JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE,
    email_verified_at TIMESTAMP WITH TIME ZONE,
    
    -- User preferences
    preferences JSONB DEFAULT '{}',
    
    -- API usage tracking
    api_quota_limit INTEGER DEFAULT 1000,
    api_quota_used INTEGER DEFAULT 0,
    api_quota_reset_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Organizations table
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    
    -- Organization details
    description TEXT,
    website VARCHAR(255),
    logo_url VARCHAR(500),
    
    -- Subscription and limits
    plan_type VARCHAR(50) DEFAULT 'free',
    document_limit INTEGER DEFAULT 100,
    storage_limit_gb INTEGER DEFAULT 1,
    query_limit_monthly INTEGER DEFAULT 1000,
    
    -- Usage tracking
    document_count INTEGER DEFAULT 0,
    storage_used_bytes BIGINT DEFAULT 0,
    query_count_current_month INTEGER DEFAULT 0,
    
    -- Settings
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Organization members table
CREATE TABLE organization_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    
    -- Membership details
    role VARCHAR(50) DEFAULT 'member',
    permissions JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    
    -- Timestamps
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    UNIQUE(organization_id, user_id)
);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    
    -- File information
    mime_type VARCHAR(100),
    file_size BIGINT,
    checksum VARCHAR(64),
    
    -- Processing status
    status VARCHAR(50) DEFAULT 'pending' NOT NULL,
    processing_progress INTEGER DEFAULT 0,
    processing_message TEXT,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_error TEXT,
    
    -- Timestamps
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- User and organization
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    
    -- Document metadata and tags
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    category VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    
    -- Content statistics
    total_pages INTEGER,
    total_chunks INTEGER DEFAULT 0,
    total_characters INTEGER DEFAULT 0,
    total_words INTEGER DEFAULT 0,
    
    -- Privacy and access control
    is_public BOOLEAN DEFAULT false,
    access_level VARCHAR(50) DEFAULT 'private',
    encryption_key_id VARCHAR(255),
    
    -- Content analysis
    content_hash VARCHAR(64),
    similarity_threshold REAL DEFAULT 0.8
);

-- Document chunks table
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE NOT NULL,
    
    -- Chunk information
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    
    -- Position information
    start_char INTEGER,
    end_char INTEGER,
    page_number INTEGER,
    section_title VARCHAR(500),
    
    -- Chunk statistics
    char_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    sentence_count INTEGER,
    
    -- Vector storage reference
    vector_id VARCHAR(255),
    embedding_model VARCHAR(255),
    embedding_created_at TIMESTAMP WITH TIME ZONE,
    
    -- Chunk metadata
    chunk_metadata JSONB DEFAULT '{}',
    chunk_type VARCHAR(50) DEFAULT 'text',
    
    -- Quality scoring
    quality_score REAL,
    relevance_score REAL,
    
    -- Processing information
    processing_method VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Query logs table
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Query information
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),
    query_type VARCHAR(50) DEFAULT 'semantic',
    
    -- Query parameters
    top_k INTEGER DEFAULT 5,
    min_score REAL DEFAULT 0.0,
    filters JSONB DEFAULT '{}',
    
    -- Results
    response_text TEXT,
    source_documents JSONB DEFAULT '[]',
    result_count INTEGER DEFAULT 0,
    confidence_score REAL,
    
    -- Performance metrics
    processing_time_ms INTEGER,
    embedding_time_ms INTEGER,
    search_time_ms INTEGER,
    llm_time_ms INTEGER,
    
    -- Context
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    
    -- Feedback and evaluation
    user_rating INTEGER,
    user_feedback TEXT,
    was_helpful BOOLEAN,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Query results table
CREATE TABLE query_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_log_id UUID REFERENCES query_logs(id) ON DELETE CASCADE NOT NULL,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE NOT NULL,
    chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE NOT NULL,
    
    -- Relevance scoring
    relevance_score REAL NOT NULL,
    rerank_score REAL,
    final_score REAL NOT NULL,
    
    -- Position in results
    rank INTEGER NOT NULL,
    
    -- Used in response
    used_in_response BOOLEAN DEFAULT false
);

-- Audit logs table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Event details
    event_type VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    result VARCHAR(50) NOT NULL,
    
    -- Actor information
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    username VARCHAR(255),
    
    -- Target resource
    resource_type VARCHAR(100),
    resource_id UUID,
    resource_name VARCHAR(255),
    
    -- Request context
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    request_id VARCHAR(255),
    
    -- Event details
    details JSONB DEFAULT '{}',
    error_message TEXT,
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- System metrics table
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    
    -- Metric value
    value REAL NOT NULL,
    unit VARCHAR(20),
    
    -- Labels/dimensions
    labels JSONB DEFAULT '{}',
    
    -- Service/component
    service_name VARCHAR(100),
    instance_id VARCHAR(255),
    
    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- API keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    
    -- Key details
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(20) NOT NULL,
    
    -- Permissions and limits
    scopes TEXT[] DEFAULT ARRAY[]::TEXT[],
    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_hour INTEGER DEFAULT 1000,
    rate_limit_per_day INTEGER DEFAULT 10000,
    
    -- Usage tracking
    last_used_at TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Document templates table
CREATE TABLE document_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Template configuration
    file_patterns TEXT[] DEFAULT ARRAY[]::TEXT[],
    mime_types TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Processing settings
    chunking_strategy VARCHAR(100) DEFAULT 'recursive',
    chunk_size INTEGER DEFAULT 512,
    chunk_overlap INTEGER DEFAULT 50,
    
    -- Extraction settings
    extraction_settings JSONB DEFAULT '{}',
    
    -- Metadata extraction
    metadata_extractors TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Quality filters
    min_chunk_length INTEGER DEFAULT 100,
    max_chunk_length INTEGER DEFAULT 2000,
    quality_threshold REAL DEFAULT 0.5,
    
    -- Template metadata
    is_default BOOLEAN DEFAULT false,
    is_system BOOLEAN DEFAULT false,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create indexes for performance
-- User indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_active ON users(is_active);

-- Organization indexes  
CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_is_active ON organizations(is_active);

-- Organization member indexes
CREATE INDEX idx_org_members_organization_id ON organization_members(organization_id);
CREATE INDEX idx_org_members_user_id ON organization_members(user_id);

-- Document indexes
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_organization_id ON documents(organization_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_uploaded_at ON documents(uploaded_at);
CREATE INDEX idx_documents_checksum ON documents(checksum);
CREATE INDEX idx_documents_user_status ON documents(user_id, status);
CREATE INDEX idx_documents_category_status ON documents(category, status);

-- Full-text search on documents
CREATE INDEX idx_documents_filename_fts ON documents USING gin(to_tsvector('english', filename));
CREATE INDEX idx_documents_tags_gin ON documents USING gin(tags);

-- Document chunk indexes
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_vector_id ON document_chunks(vector_id);
CREATE INDEX idx_chunks_page_number ON document_chunks(page_number);
CREATE INDEX idx_chunks_content_hash ON document_chunks(content_hash);
CREATE INDEX idx_chunks_document_index ON document_chunks(document_id, chunk_index);

-- Full-text search on content
CREATE INDEX idx_chunks_content_fts ON document_chunks USING gin(to_tsvector('english', content));

-- Query log indexes
CREATE INDEX idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX idx_query_logs_query_hash ON query_logs(query_hash);
CREATE INDEX idx_query_logs_session_id ON query_logs(session_id);
CREATE INDEX idx_query_logs_user_created ON query_logs(user_id, created_at);

-- Partial index for recent queries
CREATE INDEX idx_query_logs_recent ON query_logs(user_id, created_at) 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days';

-- Query result indexes
CREATE INDEX idx_query_results_query_log_id ON query_results(query_log_id);
CREATE INDEX idx_query_results_document_id ON query_results(document_id);
CREATE INDEX idx_query_results_chunk_id ON query_results(chunk_id);
CREATE INDEX idx_query_results_final_score ON query_results(final_score);

-- Audit log indexes
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_user_timestamp ON audit_logs(user_id, timestamp);

-- System metrics indexes
CREATE INDEX idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
CREATE INDEX idx_system_metrics_service_timestamp ON system_metrics(service_name, timestamp);

-- API key indexes
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organization_members_updated_at BEFORE UPDATE ON organization_members 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api_keys 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_templates_updated_at BEFORE UPDATE ON document_templates 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for generating API key prefixes
CREATE OR REPLACE FUNCTION generate_api_key_prefix(key_name TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN 'rsk_' || substring(md5(key_name || extract(epoch from now())::text), 1, 8);
END;
$$ LANGUAGE plpgsql;

-- Create views for common queries
CREATE VIEW active_documents AS
SELECT 
    d.*,
    u.username,
    u.email as user_email,
    o.name as organization_name
FROM documents d
JOIN users u ON d.user_id = u.id
LEFT JOIN organizations o ON d.organization_id = o.id
WHERE d.status = 'completed' AND u.is_active = true;

CREATE VIEW document_statistics AS
SELECT 
    user_id,
    COUNT(*) as total_documents,
    SUM(file_size) as total_size_bytes,
    SUM(total_chunks) as total_chunks,
    AVG(total_chunks) as avg_chunks_per_doc,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_documents,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_documents
FROM documents
GROUP BY user_id;

CREATE VIEW query_analytics AS
SELECT 
    user_id,
    DATE(created_at) as query_date,
    COUNT(*) as query_count,
    AVG(processing_time_ms) as avg_processing_time_ms,
    AVG(confidence_score) as avg_confidence_score,
    COUNT(CASE WHEN was_helpful = true THEN 1 END) as helpful_count,
    COUNT(CASE WHEN was_helpful = false THEN 1 END) as not_helpful_count
FROM query_logs
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY user_id, DATE(created_at);

-- Create row-level security policies
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;

-- Users can only access their own documents
CREATE POLICY documents_user_policy ON documents
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Users can only access chunks from their documents
CREATE POLICY chunks_user_policy ON document_chunks
    FOR ALL TO authenticated_users
    USING (
        document_id IN (
            SELECT id FROM documents 
            WHERE user_id = current_setting('app.current_user_id')::uuid
        )
    );

-- Users can only access their own query logs
CREATE POLICY query_logs_user_policy ON query_logs
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Create a role for the application
CREATE ROLE authenticated_users;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated_users;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated_users;

-- Grant raguser the authenticated_users role
GRANT authenticated_users TO raguser;

-- Insert initial data
-- Create system admin user (password should be changed immediately)
INSERT INTO users (
    id,
    username, 
    email, 
    hashed_password, 
    full_name, 
    is_active, 
    is_verified, 
    is_superuser,
    roles
) VALUES (
    uuid_generate_v4(),
    'admin',
    'admin@localhost',
    '$2b$12$LQv3c1yqBwWFcGgNVHy$O2EMvYumJn8C8UKkCDDvHo6FYnDFU.9.O',  -- password: admin123
    'System Administrator',
    true,
    true,
    true,
    ARRAY['super_admin', 'admin', 'user']
);

-- Create default document template
INSERT INTO document_templates (
    name,
    description,
    file_patterns,
    mime_types,
    is_default,
    is_system
) VALUES (
    'Default Template',
    'Default template for document processing',
    ARRAY['*.pdf', '*.docx', '*.txt'],
    ARRAY['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'],
    true,
    true
);

-- Create sample organization
INSERT INTO organizations (
    name,
    slug,
    description,
    plan_type
) VALUES (
    'Default Organization',
    'default-org',
    'Default organization for new users',
    'free'
);

COMMIT;

-- Final message
\echo 'Database initialization completed successfully!'
\echo 'Default admin user created with username: admin, password: admin123'
\echo 'Please change the admin password immediately after first login!'
