import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [stats, setStats] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [activeTab, setActiveTab] = useState('query');

  // Fetch documents
  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/documents`);
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  // Fetch system stats
  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/analytics/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  // Upload document
  const uploadDocument = async () => {
    if (!selectedFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/documents`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      alert('Document uploaded successfully!');
      setSelectedFile(null);
      fetchDocuments();
    } catch (error) {
      console.error('Error uploading document:', error);
      alert('Error uploading document');
    } finally {
      setUploading(false);
    }
  };

  // Search documents
  const searchDocuments = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/query`, {
        query: query,
        top_k: 5,
        min_score: 0.0
      });
      setResults(response.data.results || []);
    } catch (error) {
      console.error('Error searching documents:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Delete document
  const deleteDocument = async (documentId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;

    try {
      await axios.delete(`${API_BASE_URL}/api/v1/documents/${documentId}`);
      alert('Document deleted successfully!');
      fetchDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
      alert('Error deleting document');
    }
  };

  useEffect(() => {
    fetchDocuments();
    fetchStats();
  }, []);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      searchDocuments();
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString();
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔍 RAG System</h1>
        <p>Retrieval-Augmented Generation Document Search</p>
      </header>

      <div className="container">
        {/* Navigation */}
        <div className="nav-tabs">
          <button 
            className={activeTab === 'query' ? 'active' : ''} 
            onClick={() => setActiveTab('query')}
          >
            Search Documents
          </button>
          <button 
            className={activeTab === 'documents' ? 'active' : ''} 
            onClick={() => setActiveTab('documents')}
          >
            Manage Documents
          </button>
          <button 
            className={activeTab === 'stats' ? 'active' : ''} 
            onClick={() => setActiveTab('stats')}
          >
            Statistics
          </button>
        </div>

        {/* Search Tab */}
        {activeTab === 'query' && (
          <div className="tab-content">
            <div className="search-section">
              <h2>Search Documents</h2>
              <div className="search-box">
                <input
                  type="text"
                  placeholder="Enter your question or search query..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="search-input"
                />
                <button onClick={searchDocuments} disabled={loading} className="search-button">
                  {loading ? 'Searching...' : 'Search'}
                </button>
              </div>
            </div>

            {/* Search Results */}
            {results.length > 0 && (
              <div className="results-section">
                <h3>Search Results</h3>
                {results.map((result, index) => (
                  <div key={index} className="result-item">
                    <div className="result-header">
                      <span className="result-score">Score: {result.score.toFixed(3)}</span>
                      <span className="result-source">Source: {result.source_document}</span>
                    </div>
                    <div className="result-content">
                      <p>{result.content}</p>
                    </div>
                    <div className="result-metadata">
                      <small>Document ID: {result.document_id}</small>
                      {result.metadata && (
                        <small> | Page: {result.metadata.page || 'N/A'}</small>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <div className="tab-content">
            <div className="upload-section">
              <h2>Upload Document</h2>
              <div className="upload-box">
                <input
                  type="file"
                  accept=".pdf,.docx,.doc,.txt,.csv,.xlsx,.xml"
                  onChange={handleFileChange}
                  className="file-input"
                />
                <button
                  onClick={uploadDocument}
                  disabled={!selectedFile || uploading}
                  className="upload-button"
                >
                  {uploading ? 'Uploading...' : 'Upload Document'}
                </button>
              </div>
              {selectedFile && (
                <div className="selected-file">
                  <p>Selected: {selectedFile.name} ({formatFileSize(selectedFile.size)})</p>
                </div>
              )}
            </div>

            <div className="documents-section">
              <h2>Documents</h2>
              {documents.length === 0 ? (
                <p>No documents uploaded yet.</p>
              ) : (
                <div className="documents-grid">
                  {documents.map((doc) => (
                    <div key={doc.id} className="document-card">
                      <div className="document-header">
                        <h4>{doc.original_filename}</h4>
                        <span className={`status ${doc.status}`}>{doc.status}</span>
                      </div>
                      <div className="document-details">
                        <p>Type: {doc.file_type}</p>
                        <p>Size: {formatFileSize(doc.file_size)}</p>
                        <p>Uploaded: {formatDate(doc.upload_date)}</p>
                        {doc.chunks_count > 0 && <p>Chunks: {doc.chunks_count}</p>}
                      </div>
                      <div className="document-actions">
                        <button
                          onClick={() => deleteDocument(doc.id)}
                          className="delete-button"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Stats Tab */}
        {activeTab === 'stats' && (
          <div className="tab-content">
            <div className="stats-section">
              <h2>System Statistics</h2>
              {stats ? (
                <div className="stats-grid">
                  <div className="stat-card">
                    <h3>Documents</h3>
                    <p className="stat-number">{stats.total_documents}</p>
                  </div>
                  <div className="stat-card">
                    <h3>Queries</h3>
                    <p className="stat-number">{stats.total_queries}</p>
                  </div>
                  <div className="stat-card">
                    <h3>Avg Response Time</h3>
                    <p className="stat-number">{stats.avg_response_time?.toFixed(2)}s</p>
                  </div>
                  <div className="stat-card">
                    <h3>Storage Used</h3>
                    <p className="stat-number">{formatFileSize(stats.storage_used || 0)}</p>
                  </div>
                </div>
              ) : (
                <p>Loading statistics...</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
