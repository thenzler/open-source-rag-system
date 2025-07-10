/**
 * RAG Widget API Client
 * JavaScript API for interacting with the RAG system
 * 
 * Usage:
 * const ragApi = new RagApi('your-api-key', 'https://your-api-url.com');
 * const response = await ragApi.chat('Hello, world!');
 */

class RagApi {
    constructor(apiKey, baseUrl = 'http://localhost:8001') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
        this.chatHistory = [];
        this.requestQueue = [];
        this.isProcessing = false;
        this.retryAttempts = 3;
        this.retryDelay = 1000;
        this.timeout = 30000; // 30 seconds
        
        // Event listeners
        this.listeners = {
            'message': [],
            'error': [],
            'typing': [],
            'connect': [],
            'disconnect': []
        };
    }

    /**
     * Send a chat message to the RAG system
     * @param {string} query - The user's message
     * @param {Object} options - Additional options
     * @returns {Promise<Object>} - Response from the API
     */
    async chat(query, options = {}) {
        if (!query || typeof query !== 'string') {
            throw new Error('Query must be a non-empty string');
        }

        const requestData = {
            query: query.trim(),
            chat_history: options.includeHistory !== false ? this.chatHistory : [],
            max_tokens: options.maxTokens || 1000,
            temperature: options.temperature || 0.7,
            context_limit: options.contextLimit || 5,
            ...options.additionalParams
        };

        try {
            this.emit('typing', true);
            
            const response = await this.makeRequest('/api/chat', {
                method: 'POST',
                body: JSON.stringify(requestData)
            });

            // Add to chat history
            if (options.includeHistory !== false) {
                this.chatHistory.push({
                    query: query,
                    response: response.response,
                    timestamp: new Date().toISOString()
                });

                // Limit history size
                if (this.chatHistory.length > 50) {
                    this.chatHistory = this.chatHistory.slice(-50);
                }
            }

            this.emit('message', {
                query: query,
                response: response.response,
                context: response.context,
                confidence: response.confidence
            });

            return response;
        } catch (error) {
            this.emit('error', error);
            throw error;
        } finally {
            this.emit('typing', false);
        }
    }

    /**
     * Upload a document to the RAG system
     * @param {File|string} document - File object or text content
     * @param {Object} options - Upload options
     * @returns {Promise<Object>} - Upload response
     */
    async uploadDocument(document, options = {}) {
        const formData = new FormData();
        
        if (document instanceof File) {
            formData.append('file', document);
        } else if (typeof document === 'string') {
            const blob = new Blob([document], { type: 'text/plain' });
            formData.append('file', blob, options.filename || 'document.txt');
        } else {
            throw new Error('Document must be a File object or string');
        }

        if (options.metadata) {
            formData.append('metadata', JSON.stringify(options.metadata));
        }

        try {
            return await this.makeRequest('/api/upload', {
                method: 'POST',
                body: formData
            });
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Get available documents
     * @returns {Promise<Array>} - List of documents
     */
    async getDocuments() {
        try {
            return await this.makeRequest('/api/documents');
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Delete a document
     * @param {string} documentId - Document ID to delete
     * @returns {Promise<Object>} - Deletion response
     */
    async deleteDocument(documentId) {
        if (!documentId) {
            throw new Error('Document ID is required');
        }

        try {
            return await this.makeRequest(`/api/documents/${documentId}`, {
                method: 'DELETE'
            });
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Search documents
     * @param {string} query - Search query
     * @param {Object} options - Search options
     * @returns {Promise<Object>} - Search results
     */
    async searchDocuments(query, options = {}) {
        if (!query) {
            throw new Error('Search query is required');
        }

        const params = new URLSearchParams({
            q: query,
            limit: options.limit || 10,
            offset: options.offset || 0,
            ...options.filters
        });

        try {
            return await this.makeRequest(`/api/search?${params}`);
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Get system status
     * @returns {Promise<Object>} - System status
     */
    async getStatus() {
        try {
            return await this.makeRequest('/api/status');
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Clear chat history
     */
    clearHistory() {
        this.chatHistory = [];
        this.emit('message', { type: 'history_cleared' });
    }

    /**
     * Get chat history
     * @returns {Array} - Chat history
     */
    getHistory() {
        return [...this.chatHistory];
    }

    /**
     * Set chat history
     * @param {Array} history - Chat history to set
     */
    setHistory(history) {
        if (!Array.isArray(history)) {
            throw new Error('History must be an array');
        }
        this.chatHistory = [...history];
    }

    /**
     * Make HTTP request with retry logic
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<Object>} - Response data
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const requestOptions = {
            method: options.method || 'GET',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
                ...options.headers
            },
            body: options.body,
            signal: AbortSignal.timeout(this.timeout)
        };

        // Don't set Content-Type for FormData
        if (options.body instanceof FormData) {
            delete requestOptions.headers['Content-Type'];
        }

        let lastError;
        
        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                const response = await fetch(url, requestOptions);
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                
                if (attempt > 1) {
                    this.emit('connect');
                }
                
                return data;
            } catch (error) {
                lastError = error;
                
                if (attempt === this.retryAttempts) {
                    this.emit('disconnect');
                    break;
                }

                // Don't retry on certain errors
                if (error.message.includes('401') || error.message.includes('403')) {
                    break;
                }

                // Wait before retry
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * attempt));
            }
        }

        throw lastError;
    }

    /**
     * Add event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    off(event, callback) {
        if (!this.listeners[event]) return;
        
        const index = this.listeners[event].indexOf(callback);
        if (index > -1) {
            this.listeners[event].splice(index, 1);
        }
    }

    /**
     * Emit event
     * @param {string} event - Event name
     * @param {*} data - Event data
     */
    emit(event, data) {
        if (!this.listeners[event]) return;
        
        this.listeners[event].forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Error in event listener:', error);
            }
        });
    }

    /**
     * Check if API is available
     * @returns {Promise<boolean>} - True if API is available
     */
    async isAvailable() {
        try {
            await this.getStatus();
            return true;
        } catch (error) {
            return false;
        }
    }

    /**
     * Get API configuration
     * @returns {Object} - API configuration
     */
    getConfig() {
        return {
            apiKey: this.apiKey ? '***' : null,
            baseUrl: this.baseUrl,
            retryAttempts: this.retryAttempts,
            retryDelay: this.retryDelay,
            timeout: this.timeout
        };
    }

    /**
     * Update API configuration
     * @param {Object} config - New configuration
     */
    updateConfig(config) {
        if (config.apiKey !== undefined) this.apiKey = config.apiKey;
        if (config.baseUrl !== undefined) this.baseUrl = config.baseUrl.replace(/\/$/, '');
        if (config.retryAttempts !== undefined) this.retryAttempts = config.retryAttempts;
        if (config.retryDelay !== undefined) this.retryDelay = config.retryDelay;
        if (config.timeout !== undefined) this.timeout = config.timeout;
    }
}

/**
 * RAG Widget Controller
 * Higher-level controller for the widget functionality
 */
class RagWidget {
    constructor(apiKey, options = {}) {
        this.api = new RagApi(apiKey, options.apiUrl);
        this.element = null;
        this.isOpen = false;
        this.options = {
            theme: 'default',
            position: 'bottom-right',
            title: 'RAG Assistant',
            welcomeMessage: 'Hello! How can I help you today?',
            placeholder: 'Type your message...',
            showTypingIndicator: true,
            showNotifications: true,
            persistHistory: true,
            maxHistorySize: 50,
            ...options
        };

        this.setupEventListeners();
        this.loadHistory();
    }

    /**
     * Initialize the widget
     * @param {HTMLElement} container - Container element
     */
    init(container) {
        this.element = container;
        this.render();
        this.attachEventListeners();
    }

    /**
     * Render the widget
     */
    render() {
        // Implementation would go here
        // This is a placeholder for the actual rendering logic
        console.log('Rendering widget with options:', this.options);
    }

    /**
     * Open the widget
     */
    open() {
        this.isOpen = true;
        this.element?.classList.add('rag-widget-open');
        this.api.emit('widget_open');
    }

    /**
     * Close the widget
     */
    close() {
        this.isOpen = false;
        this.element?.classList.remove('rag-widget-open');
        this.api.emit('widget_close');
    }

    /**
     * Toggle the widget
     */
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    /**
     * Send a message
     * @param {string} message - Message to send
     */
    async sendMessage(message) {
        try {
            const response = await this.api.chat(message);
            this.displayMessage(message, 'user');
            this.displayMessage(response.response, 'assistant');
            this.saveHistory();
        } catch (error) {
            this.displayError(error.message);
        }
    }

    /**
     * Display a message in the widget
     * @param {string} message - Message to display
     * @param {string} type - Message type ('user' or 'assistant')
     */
    displayMessage(message, type) {
        // Implementation would go here
        console.log(`${type}: ${message}`);
    }

    /**
     * Display an error message
     * @param {string} error - Error message
     */
    displayError(error) {
        // Implementation would go here
        console.error('Widget error:', error);
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        this.api.on('typing', (isTyping) => {
            // Show/hide typing indicator
            this.element?.classList.toggle('rag-typing', isTyping);
        });

        this.api.on('error', (error) => {
            this.displayError(error.message);
        });

        this.api.on('message', (data) => {
            if (this.options.showNotifications && !this.isOpen) {
                this.showNotification();
            }
        });
    }

    /**
     * Attach DOM event listeners
     */
    attachEventListeners() {
        if (!this.element) return;

        // Add click handlers, input handlers, etc.
        // Implementation would go here
    }

    /**
     * Show notification
     */
    showNotification() {
        // Implementation would go here
        console.log('Showing notification');
    }

    /**
     * Save chat history to localStorage
     */
    saveHistory() {
        if (!this.options.persistHistory) return;

        try {
            const history = this.api.getHistory();
            localStorage.setItem(`rag_widget_history_${this.api.apiKey}`, JSON.stringify(history));
        } catch (error) {
            console.error('Failed to save history:', error);
        }
    }

    /**
     * Load chat history from localStorage
     */
    loadHistory() {
        if (!this.options.persistHistory) return;

        try {
            const saved = localStorage.getItem(`rag_widget_history_${this.api.apiKey}`);
            if (saved) {
                const history = JSON.parse(saved);
                this.api.setHistory(history);
            }
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }

    /**
     * Clear chat history
     */
    clearHistory() {
        this.api.clearHistory();
        if (this.options.persistHistory) {
            localStorage.removeItem(`rag_widget_history_${this.api.apiKey}`);
        }
    }

    /**
     * Update widget options
     * @param {Object} newOptions - New options
     */
    updateOptions(newOptions) {
        this.options = { ...this.options, ...newOptions };
        this.render();
    }

    /**
     * Destroy the widget
     */
    destroy() {
        this.element?.remove();
        this.api.listeners = {};
        this.element = null;
    }
}

// Export for use in different environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RagApi, RagWidget };
} else if (typeof window !== 'undefined') {
    window.RagApi = RagApi;
    window.RagWidget = RagWidget;
}