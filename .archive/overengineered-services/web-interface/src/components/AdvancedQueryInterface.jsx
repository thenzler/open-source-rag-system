import React, { useState, useEffect, useCallback } from 'react';
import { 
  Search, 
  FileText, 
  Brain, 
  Zap, 
  TrendingUp, 
  Shield, 
  Settings,
  Download,
  Filter,
  BarChart3,
  Activity,
  Users,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  RefreshCw,
  Eye,
  MessageSquare,
  Lightbulb
} from 'lucide-react';

const AdvancedQueryInterface = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [queryExpansion, setQueryExpansion] = useState(null);
  const [filters, setFilters] = useState({
    categories: [],
    contentTypes: [],
    dateRange: { start: '', end: '' },
    similarityThreshold: 0.5
  });
  const [advancedMode, setAdvancedMode] = useState(false);
  const [streamingMode, setStreamingMode] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);
  const [clusters, setClusters] = useState([]);

  // Debounced search suggestions
  const [suggestionTimeout, setSuggestionTimeout] = useState(null);

  const fetchSuggestions = useCallback(async (searchQuery) => {
    if (searchQuery.length < 2) return;
    
    try {
      const response = await fetch(`/api/v1/advanced/query/suggestions?query=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setSuggestions(data.suggestions || []);
    } catch (error) {
      console.error('Failed to fetch suggestions:', error);
    }
  }, []);

  const handleQueryChange = (e) => {
    const value = e.target.value;
    setQuery(value);
    
    if (suggestionTimeout) {
      clearTimeout(suggestionTimeout);
    }
    
    const timeout = setTimeout(() => {
      fetchSuggestions(value);
    }, 300);
    
    setSuggestionTimeout(timeout);
  };

  const executeQuery = async (queryText = query) => {
    setLoading(true);
    setResults([]);
    setConfidence(0);
    setProcessingTime(0);
    setClusters([]);
    
    try {
      const requestBody = {
        query: queryText,
        enable_expansion: advancedMode,
        enable_clustering: advancedMode,
        similarity_threshold: filters.similarityThreshold,
        categories: filters.categories,
        content_types: filters.contentTypes,
        date_range: filters.dateRange.start && filters.dateRange.end ? filters.dateRange : null,
        top_k: 15
      };

      if (streamingMode) {
        await executeStreamingQuery(requestBody);
      } else {
        await executeStandardQuery(requestBody);
      }
    } catch (error) {
      console.error('Query failed:', error);
      setResults([{ error: 'Query failed. Please try again.' }]);
    } finally {
      setLoading(false);
    }
  };

  const executeStandardQuery = async (requestBody) => {
    const response = await fetch('/api/v1/advanced/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    setResults(data.results || []);
    setConfidence(data.confidence_score || 0);
    setProcessingTime(data.processing_time || 0);
    setQueryExpansion(data.query_expansion || null);
    setClusters(data.semantic_clusters || []);
    setSuggestions(data.suggestions || []);
  };

  const executeStreamingQuery = async (requestBody) => {
    const response = await fetch('/api/v1/advanced/query/streaming', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        query: requestBody.query,
        stream_results: true,
        batch_size: 5,
        delay_ms: 100
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedResults = [];

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'results') {
                accumulatedResults = [...accumulatedResults, ...data.results];
                setResults([...accumulatedResults]);
              } else if (data.type === 'complete') {
                setConfidence(data.confidence_score || 0);
                setProcessingTime(data.processing_time || 0);
              }
            } catch (error) {
              console.error('Failed to parse streaming data:', error);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion.text);
    setSuggestions([]);
    executeQuery(suggestion.text);
  };

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
  };

  const submitFeedback = async (resultId, feedback) => {
    try {
      await fetch('/api/v1/advanced/query/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          query_id: resultId,
          ...feedback
        })
      });
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center gap-3">
          <Brain className="text-blue-600" size={32} />
          Advanced RAG Query Interface
        </h1>
        <p className="text-gray-600">
          Intelligent document search with semantic understanding and advanced filtering
        </p>
      </div>

      {/* Query Input Section */}
      <div className="mb-8">
        <div className="relative">
          <div className="relative">
            <Search className="absolute left-3 top-3 text-gray-400" size={20} />
            <input
              type="text"
              value={query}
              onChange={handleQueryChange}
              placeholder="Ask anything about your documents..."
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
              onKeyPress={(e) => e.key === 'Enter' && executeQuery()}
            />
          </div>
          
          {/* Suggestions Dropdown */}
          {suggestions.length > 0 && (
            <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {suggestions.map((suggestion, index) => (
                <div
                  key={index}
                  className="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center gap-2"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  <Lightbulb size={16} className="text-yellow-500" />
                  <span className="text-sm">{suggestion.text}</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    suggestion.type === 'history' ? 'bg-blue-100 text-blue-800' :
                    suggestion.type === 'expansion' ? 'bg-green-100 text-green-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {suggestion.type}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Query Options */}
        <div className="flex items-center gap-4 mt-4">
          <button
            onClick={() => executeQuery()}
            disabled={loading || !query.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <RefreshCw size={16} className="animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Search size={16} />
                Search
              </>
            )}
          </button>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={advancedMode}
              onChange={(e) => setAdvancedMode(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm text-gray-700">Advanced Mode</span>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={streamingMode}
              onChange={(e) => setStreamingMode(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm text-gray-700">Streaming Results</span>
          </label>

          <button
            onClick={() => setAdvancedMode(!advancedMode)}
            className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-800"
          >
            <Filter size={16} />
            Filters
          </button>
        </div>

        {/* Advanced Filters */}
        {advancedMode && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Settings size={16} />
              Advanced Filters
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Similarity Threshold */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Similarity Threshold: {filters.similarityThreshold}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={filters.similarityThreshold}
                  onChange={(e) => handleFilterChange('similarityThreshold', parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Categories */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Categories
                </label>
                <select
                  multiple
                  value={filters.categories}
                  onChange={(e) => handleFilterChange('categories', Array.from(e.target.selectedOptions, option => option.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="research">Research</option>
                  <option value="technology">Technology</option>
                  <option value="business">Business</option>
                  <option value="academic">Academic</option>
                </select>
              </div>

              {/* Content Types */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Content Types
                </label>
                <select
                  multiple
                  value={filters.contentTypes}
                  onChange={(e) => handleFilterChange('contentTypes', Array.from(e.target.selectedOptions, option => option.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="pdf">PDF</option>
                  <option value="docx">Word Document</option>
                  <option value="txt">Text File</option>
                  <option value="xlsx">Excel</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Query Stats */}
      {(confidence > 0 || processingTime > 0) && (
        <div className="mb-6 p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <TrendingUp size={16} className="text-blue-600" />
              <span className="text-sm font-medium">Confidence: {(confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock size={16} className="text-blue-600" />
              <span className="text-sm font-medium">Processing Time: {processingTime.toFixed(3)}s</span>
            </div>
            <div className="flex items-center gap-2">
              <FileText size={16} className="text-blue-600" />
              <span className="text-sm font-medium">Results: {results.length}</span>
            </div>
          </div>
        </div>
      )}

      {/* Query Expansion */}
      {queryExpansion && queryExpansion.expanded_queries.length > 0 && (
        <div className="mb-6 p-4 bg-green-50 rounded-lg">
          <h3 className="font-semibold mb-2 flex items-center gap-2">
            <Zap size={16} className="text-green-600" />
            Query Expansion
          </h3>
          <div className="flex flex-wrap gap-2">
            {queryExpansion.expanded_queries.map((expandedQuery, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm cursor-pointer hover:bg-green-200"
                onClick={() => executeQuery(expandedQuery)}
              >
                {expandedQuery}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Semantic Clusters */}
      {clusters.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <BarChart3 size={16} className="text-purple-600" />
            Semantic Clusters
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {clusters.map((cluster, index) => (
              <div key={index} className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-medium text-purple-900 mb-2">{cluster.topic}</h4>
                <p className="text-sm text-purple-700 mb-2">
                  {cluster.size} documents • Avg Score: {cluster.avg_score.toFixed(2)}
                </p>
                <div className="flex flex-wrap gap-1">
                  {cluster.results.slice(0, 3).map((result, idx) => (
                    <span key={idx} className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">
                      {result.title?.substring(0, 20) || 'Document'}...
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      <div className="space-y-6">
        {results.length > 0 && (
          <h3 className="font-semibold text-lg flex items-center gap-2">
            <FileText size={20} className="text-blue-600" />
            Search Results
          </h3>
        )}

        {results.map((result, index) => (
          <ResultCard
            key={index}
            result={result}
            onFeedback={(feedback) => submitFeedback(result.id, feedback)}
          />
        ))}
      </div>

      {/* Suggestions */}
      {suggestions.length > 0 && results.length > 0 && (
        <div className="mt-8 p-4 bg-yellow-50 rounded-lg">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Lightbulb size={16} className="text-yellow-600" />
            Related Suggestions
          </h3>
          <div className="flex flex-wrap gap-2">
            {suggestions.slice(0, 5).map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm hover:bg-yellow-200"
              >
                {suggestion.text}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <RefreshCw size={32} className="animate-spin text-blue-600 mx-auto mb-4" />
            <p className="text-gray-600">Processing your query...</p>
          </div>
        </div>
      )}

      {/* No Results */}
      {!loading && results.length === 0 && query && (
        <div className="text-center py-12">
          <MessageSquare size={48} className="text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No results found for your query.</p>
          <p className="text-sm text-gray-500 mt-2">
            Try adjusting your search terms or filters.
          </p>
        </div>
      )}
    </div>
  );
};

const ResultCard = ({ result, onFeedback }) => {
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedback, setFeedback] = useState({
    rating: 5,
    relevance: 5,
    comments: ''
  });

  const handleFeedbackSubmit = () => {
    onFeedback(feedback);
    setShowFeedback(false);
    setFeedback({ rating: 5, relevance: 5, comments: '' });
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h4 className="font-semibold text-lg text-gray-900 mb-1">
            {result.title || 'Document'}
          </h4>
          <p className="text-sm text-gray-500 mb-2">
            {result.document_type} • {result.created_at ? new Date(result.created_at).toLocaleDateString() : 'Unknown date'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(result.score)}`}>
            {(result.score * 100).toFixed(1)}%
          </span>
          <button
            onClick={() => setShowFeedback(!showFeedback)}
            className="text-gray-400 hover:text-gray-600"
          >
            <MessageSquare size={16} />
          </button>
        </div>
      </div>

      <div className="mb-4">
        <p className="text-gray-700 leading-relaxed">
          {result.content}
        </p>
      </div>

      <div className="flex items-center justify-between text-sm text-gray-500">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <FileText size={14} />
            {result.document_id}
          </span>
          {result.chunk_id && (
            <span className="flex items-center gap-1">
              <Eye size={14} />
              Chunk {result.chunk_id}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span>Page {result.page_number || 1}</span>
          <Download size={14} className="cursor-pointer hover:text-gray-700" />
        </div>
      </div>

      {/* Feedback Section */}
      {showFeedback && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h5 className="font-medium mb-3">Rate this result</h5>
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Overall Rating
              </label>
              <select
                value={feedback.rating}
                onChange={(e) => setFeedback({...feedback, rating: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                {[1, 2, 3, 4, 5].map(rating => (
                  <option key={rating} value={rating}>{rating} Stars</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Relevance
              </label>
              <select
                value={feedback.relevance}
                onChange={(e) => setFeedback({...feedback, relevance: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                {[1, 2, 3, 4, 5].map(relevance => (
                  <option key={relevance} value={relevance}>{relevance} - {
                    relevance === 1 ? 'Not Relevant' :
                    relevance === 2 ? 'Slightly Relevant' :
                    relevance === 3 ? 'Moderately Relevant' :
                    relevance === 4 ? 'Very Relevant' :
                    'Extremely Relevant'
                  }</option>
                ))}
              </select>
            </div>
          </div>
          <textarea
            value={feedback.comments}
            onChange={(e) => setFeedback({...feedback, comments: e.target.value})}
            placeholder="Additional comments (optional)"
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-md mb-3"
          />
          <div className="flex gap-2">
            <button
              onClick={handleFeedbackSubmit}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Submit Feedback
            </button>
            <button
              onClick={() => setShowFeedback(false)}
              className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedQueryInterface;