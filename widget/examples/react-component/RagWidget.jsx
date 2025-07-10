import React, { useState, useEffect, useRef } from 'react';
import './RagWidget.css';

/**
 * RAG Chat Widget React Component
 * 
 * Props:
 * - apiKey (string, required): Your RAG API key
 * - apiUrl (string): API base URL (default: http://localhost:8001)
 * - theme (string): Widget theme (default, dark, blue, green, purple, orange, red)
 * - position (string): Widget position (bottom-right, bottom-left, top-right, top-left)
 * - primaryColor (string): Primary color hex code
 * - secondaryColor (string): Secondary color hex code
 * - title (string): Widget title
 * - welcomeMessage (string): Welcome message
 * - placeholder (string): Input placeholder text
 * - zIndex (number): CSS z-index value
 * - onMessage (function): Callback when message is sent
 * - onResponse (function): Callback when response is received
 * - onError (function): Callback when error occurs
 * - onOpen (function): Callback when widget opens
 * - onClose (function): Callback when widget closes
 */

const RagWidget = ({
  apiKey,
  apiUrl = 'http://localhost:8001',
  theme = 'default',
  position = 'bottom-right',
  primaryColor = '#667eea',
  secondaryColor = '#764ba2',
  title = 'AI Assistant',
  welcomeMessage = 'Hello! How can I help you today?',
  placeholder = 'Type your message...',
  zIndex = 999999,
  onMessage,
  onResponse,
  onError,
  onOpen,
  onClose
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [hasNewMessage, setHasNewMessage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  // Initialize with welcome message
  useEffect(() => {
    setMessages([
      {
        id: 1,
        type: 'assistant',
        content: welcomeMessage,
        timestamp: new Date()
      }
    ]);
  }, [welcomeMessage]);
  
  // Scroll to bottom when new messages are added
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Focus input when widget opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);
  
  // Load chat history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem(`rag-widget-history-${apiKey}`);
    if (savedHistory) {
      try {
        const history = JSON.parse(savedHistory);
        if (history.length > 0) {
          setMessages(history);
        }
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }
  }, [apiKey]);
  
  // Save chat history to localStorage
  useEffect(() => {
    if (messages.length > 1) { // Don't save just the welcome message
      localStorage.setItem(`rag-widget-history-${apiKey}`, JSON.stringify(messages));
    }
  }, [messages, apiKey]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  const toggleWidget = () => {
    const newIsOpen = !isOpen;
    setIsOpen(newIsOpen);
    setHasNewMessage(false);
    
    if (newIsOpen) {
      onOpen?.();
    } else {
      onClose?.();
    }
  };
  
  const sendMessage = async (message) => {
    if (!message.trim() || isLoading) return;
    
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setIsTyping(true);
    
    onMessage?.(message);
    
    try {
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          query: message,
          chat_history: messages.slice(-10) // Send last 10 messages for context
        })
      });
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }
      
      const data = await response.json();
      
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: data.response,
        timestamp: new Date(),
        confidence: data.confidence,
        sources: data.context
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      onResponse?.(data);
      
      // Show notification if widget is closed
      if (!isOpen) {
        setHasNewMessage(true);
      }
      
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again later.',
        timestamp: new Date(),
        error: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
      onError?.(error);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputValue);
    }
  };
  
  const clearHistory = () => {
    setMessages([
      {
        id: 1,
        type: 'assistant',
        content: welcomeMessage,
        timestamp: new Date()
      }
    ]);
    localStorage.removeItem(`rag-widget-history-${apiKey}`);
  };
  
  // CSS custom properties for theming
  const cssVariables = {
    '--rag-primary-color': primaryColor,
    '--rag-secondary-color': secondaryColor,
    '--rag-z-index': zIndex
  };
  
  return (
    <div 
      className={`rag-widget ${theme} ${position}`}
      style={cssVariables}
    >
      {/* Chat bubble */}
      <div 
        className={`rag-chat-bubble ${isOpen ? 'hidden' : ''}`}
        onClick={toggleWidget}
      >
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 9h12v2H6V9zm8 5H6v-2h8v2zm4-6H6V6h12v2z"/>
        </svg>
        {hasNewMessage && (
          <div className="rag-notification-badge">1</div>
        )}
      </div>
      
      {/* Chat window */}
      <div className={`rag-chat-window ${isOpen ? 'open' : ''}`}>
        {/* Header */}
        <div className="rag-chat-header">
          <h3>{title}</h3>
          <button 
            onClick={toggleWidget}
            className="rag-close-button"
            aria-label="Close chat"
          >
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </div>
        
        {/* Messages */}
        <div className="rag-chat-messages">
          {messages.map(message => (
            <div 
              key={message.id}
              className={`rag-message ${message.type} ${message.error ? 'error' : ''}`}
            >
              <div className="rag-message-content">
                {message.content}
                {message.confidence && (
                  <div className="rag-message-confidence">
                    Confidence: {Math.round(message.confidence * 100)}%
                  </div>
                )}
                {message.sources && message.sources.length > 0 && (
                  <div className="rag-message-sources">
                    <details>
                      <summary>Sources ({message.sources.length})</summary>
                      <ul>
                        {message.sources.map((source, index) => (
                          <li key={index}>
                            <strong>{source.source_document}</strong>
                            <p>{source.content.substring(0, 100)}...</p>
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}
              </div>
              <div className="rag-message-time">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          
          {/* Typing indicator */}
          {isTyping && (
            <div className="rag-typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input */}
        <div className="rag-chat-input-container">
          <div className="rag-chat-controls">
            <button 
              onClick={clearHistory}
              className="rag-clear-button"
              title="Clear chat history"
            >
              🗑️
            </button>
          </div>
          <div className="rag-chat-input-wrapper">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={placeholder}
              className="rag-chat-input"
              disabled={isLoading}
            />
            <button
              onClick={() => sendMessage(inputValue)}
              className="rag-send-button"
              disabled={isLoading || !inputValue.trim()}
            >
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RagWidget;