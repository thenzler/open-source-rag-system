/**
 * RAG Chat Widget Loader
 * Easy integration script for embedding the RAG chat widget on any website
 * 
 * Usage:
 * <script src="https://yoursite.com/widget-loader.js" 
 *         data-api-key="YOUR_API_KEY"
 *         data-api-url="https://your-api.com"
 *         data-theme="default"
 *         data-position="bottom-right">
 * </script>
 */

(function() {
    'use strict';

    // Get the current script element
    const currentScript = document.currentScript || (function() {
        const scripts = document.getElementsByTagName('script');
        return scripts[scripts.length - 1];
    })();

    // Extract configuration from data attributes
    const config = {
        apiKey: currentScript.getAttribute('data-api-key') || '',
        apiUrl: currentScript.getAttribute('data-api-url') || 'http://localhost:8000',
        theme: currentScript.getAttribute('data-theme') || 'default',
        position: currentScript.getAttribute('data-position') || 'bottom-right',
        primaryColor: currentScript.getAttribute('data-primary-color') || '#667eea',
        secondaryColor: currentScript.getAttribute('data-secondary-color') || '#764ba2',
        welcomeMessage: currentScript.getAttribute('data-welcome-message') || 'Hello! I\'m your RAG assistant. How can I help you today?',
        title: currentScript.getAttribute('data-title') || 'RAG Assistant',
        placeholder: currentScript.getAttribute('data-placeholder') || 'Type your message...',
        autoLoad: currentScript.getAttribute('data-auto-load') !== 'false',
        zIndex: currentScript.getAttribute('data-z-index') || '999999'
    };

    // Widget HTML template
    const widgetHTML = `
        <style>
            /* Reset and base styles */
            .rag-widget-frame * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            /* Widget container */
            .rag-widget-frame {
                position: fixed;
                bottom: 20px;
                ${config.position === 'bottom-left' ? 'left: 20px;' : 'right: 20px;'}
                z-index: ${config.zIndex};
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }

            /* Chat bubble button */
            .rag-chat-bubble {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(135deg, ${config.primaryColor} 0%, ${config.secondaryColor} 100%);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                position: relative;
            }

            .rag-chat-bubble:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            }

            .rag-chat-bubble svg {
                width: 30px;
                height: 30px;
                fill: white;
            }

            /* Notification badge */
            .rag-notification-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                width: 20px;
                height: 20px;
                background: #ff4757;
                border-radius: 50%;
                display: none;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                color: white;
                font-weight: bold;
            }

            /* Chat window */
            .rag-chat-window {
                position: absolute;
                bottom: 80px;
                ${config.position === 'bottom-left' ? 'left: 0;' : 'right: 0;'}
                width: 380px;
                height: 600px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
                display: none;
                flex-direction: column;
                overflow: hidden;
                animation: ragSlideUp 0.3s ease;
            }

            @keyframes ragSlideUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Chat header */
            .rag-chat-header {
                background: linear-gradient(135deg, ${config.primaryColor} 0%, ${config.secondaryColor} 100%);
                color: white;
                padding: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .rag-chat-title {
                font-size: 18px;
                font-weight: 600;
            }

            .rag-close-button {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                padding: 5px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 4px;
                transition: background 0.2s;
            }

            .rag-close-button:hover {
                background: rgba(255, 255, 255, 0.2);
            }

            .rag-close-button svg {
                width: 20px;
                height: 20px;
                fill: currentColor;
            }

            /* Chat messages area */
            .rag-chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }

            /* Message styles */
            .rag-message {
                margin-bottom: 16px;
                display: flex;
                animation: ragFadeIn 0.3s ease;
            }

            @keyframes ragFadeIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .rag-message.user {
                justify-content: flex-end;
            }

            .rag-message-content {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 18px;
                word-wrap: break-word;
            }

            .rag-message.user .rag-message-content {
                background: linear-gradient(135deg, ${config.primaryColor} 0%, ${config.secondaryColor} 100%);
                color: white;
                border-bottom-right-radius: 4px;
            }

            .rag-message.assistant .rag-message-content {
                background: white;
                color: #333;
                border: 1px solid #e0e0e0;
                border-bottom-left-radius: 4px;
            }

            /* Typing indicator */
            .rag-typing-indicator {
                display: none;
                padding: 12px 16px;
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 18px;
                border-bottom-left-radius: 4px;
                width: fit-content;
                margin-bottom: 16px;
            }

            .rag-typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #999;
                margin: 0 2px;
                animation: ragTyping 1.4s infinite;
            }

            .rag-typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }

            .rag-typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes ragTyping {
                0%, 60%, 100% {
                    transform: translateY(0);
                }
                30% {
                    transform: translateY(-10px);
                }
            }

            /* Chat input area */
            .rag-chat-input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #e0e0e0;
            }

            .rag-chat-input-wrapper {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .rag-chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #e0e0e0;
                border-radius: 24px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.2s;
                font-family: inherit;
            }

            .rag-chat-input:focus {
                border-color: ${config.primaryColor};
            }

            .rag-send-button {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, ${config.primaryColor} 0%, ${config.secondaryColor} 100%);
                border: none;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }

            .rag-send-button:hover {
                transform: scale(1.1);
            }

            .rag-send-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: scale(1);
            }

            .rag-send-button svg {
                width: 20px;
                height: 20px;
                fill: currentColor;
            }

            /* Mobile responsive */
            @media (max-width: 480px) {
                .rag-chat-window {
                    width: 100vw;
                    height: 100vh;
                    bottom: 0;
                    ${config.position === 'bottom-left' ? 'left: 0;' : 'right: 0;'}
                    border-radius: 0;
                    max-width: 100vw;
                }

                .rag-widget-frame {
                    bottom: 10px;
                    ${config.position === 'bottom-left' ? 'left: 10px;' : 'right: 10px;'}
                }

                .rag-chat-bubble {
                    width: 56px;
                    height: 56px;
                }
            }

            /* Custom scrollbar */
            .rag-chat-messages::-webkit-scrollbar {
                width: 6px;
            }

            .rag-chat-messages::-webkit-scrollbar-track {
                background: #f1f1f1;
            }

            .rag-chat-messages::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 3px;
            }

            .rag-chat-messages::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>

        <div class="rag-widget-frame" id="ragWidgetFrame">
            <!-- Chat bubble -->
            <div class="rag-chat-bubble" id="ragChatBubble">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 9h12v2H6V9zm8 5H6v-2h8v2zm4-6H6V6h12v2z"/>
                </svg>
                <div class="rag-notification-badge" id="ragNotificationBadge">1</div>
            </div>

            <!-- Chat window -->
            <div class="rag-chat-window" id="ragChatWindow">
                <div class="rag-chat-header">
                    <h3 class="rag-chat-title">${config.title}</h3>
                    <button class="rag-close-button" id="ragCloseButton">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                        </svg>
                    </button>
                </div>

                <div class="rag-chat-messages" id="ragChatMessages">
                    <!-- Welcome message -->
                    <div class="rag-message assistant">
                        <div class="rag-message-content">
                            ${config.welcomeMessage}
                        </div>
                    </div>
                    
                    <!-- Typing indicator -->
                    <div class="rag-typing-indicator" id="ragTypingIndicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>

                <div class="rag-chat-input-container">
                    <div class="rag-chat-input-wrapper">
                        <input 
                            type="text" 
                            class="rag-chat-input" 
                            id="ragChatInput"
                            placeholder="${config.placeholder}"
                            autocomplete="off"
                        />
                        <button class="rag-send-button" id="ragSendButton">
                            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Create and inject widget
    function createWidget() {
        // Create container div
        const widgetContainer = document.createElement('div');
        widgetContainer.innerHTML = widgetHTML;
        
        // Append to body
        document.body.appendChild(widgetContainer);

        // Initialize widget functionality
        initializeWidget();
    }

    // Initialize widget functionality
    function initializeWidget() {
        // DOM elements
        const chatBubble = document.getElementById('ragChatBubble');
        const chatWindow = document.getElementById('ragChatWindow');
        const closeButton = document.getElementById('ragCloseButton');
        const chatMessages = document.getElementById('ragChatMessages');
        const chatInput = document.getElementById('ragChatInput');
        const sendButton = document.getElementById('ragSendButton');
        const typingIndicator = document.getElementById('ragTypingIndicator');
        const notificationBadge = document.getElementById('ragNotificationBadge');

        // State
        let isOpen = false;
        let messageHistory = [];

        // Event listeners
        chatBubble.addEventListener('click', toggleChat);
        closeButton.addEventListener('click', closeChat);
        chatInput.addEventListener('keypress', handleInputKeypress);
        sendButton.addEventListener('click', sendMessage);

        // Load chat history from localStorage
        loadChatHistory();

        // Toggle chat window
        function toggleChat() {
            isOpen = !isOpen;
            if (isOpen) {
                chatWindow.style.display = 'flex';
                chatBubble.style.display = 'none';
                chatInput.focus();
                notificationBadge.style.display = 'none';
                scrollToBottom();
            } else {
                closeChat();
            }
        }

        // Close chat window
        function closeChat() {
            isOpen = false;
            chatWindow.style.display = 'none';
            chatBubble.style.display = 'flex';
        }

        // Handle input keypress
        function handleInputKeypress(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        // Send message
        async function sendMessage() {
            const message = chatInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            chatInput.value = '';
            sendButton.disabled = true;

            // Show typing indicator
            showTypingIndicator();

            try {
                // Try optimized endpoint first
                let response;
                let data;
                
                try {
                    console.log('Trying optimized endpoint...');
                    response = await fetch(`${config.apiUrl}/api/v1/query/optimized`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            query: message,
                            context_limit: 3,
                            max_tokens: 200
                        })
                    });
                    
                    if (response.ok) {
                        data = await response.json();
                        console.log('Optimized endpoint response:', data);
                    } else {
                        throw new Error(`Optimized endpoint failed: ${response.status}`);
                    }
                } catch (optimizedError) {
                    console.log('Optimized endpoint failed, trying fallback...');
                    
                    // Fallback to chat endpoint
                    response = await fetch(`${config.apiUrl}/api/chat`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            query: message,
                            chat_history: messageHistory.slice(-5) // Last 5 messages
                        })
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error('Fallback API Error:', response.status, errorText);
                        throw new Error(`Both endpoints failed. Last error: ${response.status} - ${errorText}`);
                    }
                    
                    data = await response.json();
                    console.log('Fallback endpoint response:', data);
                }
                
                // Hide typing indicator
                hideTypingIndicator();
                
                // Check if we have a valid response
                if (!data || (!data.response && !data.answer)) {
                    console.error('Invalid response structure:', data);
                    throw new Error('Invalid response format from API');
                }
                
                // Handle both response formats
                const responseText = data.response || data.answer || 'No response available';
                const sources = data.context || data.sources || [];
                
                // Add assistant message with sources
                addMessage(responseText, 'assistant', sources);

                // Save to history
                messageHistory.push({
                    query: message,
                    response: responseText,
                    sources: sources
                });
                saveChatHistory();

            } catch (error) {
                console.error('Error details:', error);
                hideTypingIndicator();
                
                // Show more specific error message
                let errorMessage = 'Sorry, I encountered an error. ';
                if (error.message.includes('fetch')) {
                    errorMessage += 'Cannot connect to the server. Please check if the API is running.';
                } else if (error.message.includes('400')) {
                    errorMessage += 'Invalid request format.';
                } else if (error.message.includes('404')) {
                    errorMessage += 'API endpoint not found.';
                } else if (error.message.includes('500')) {
                    errorMessage += 'Server error occurred.';
                } else {
                    errorMessage += 'Please try again later.';
                }
                
                addMessage(errorMessage, 'assistant');
            } finally {
                sendButton.disabled = false;
                chatInput.focus();
            }
        }

        // Add message to chat
        function addMessage(content, type, sources = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `rag-message ${type}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'rag-message-content';
            contentDiv.innerHTML = content.replace(/\n/g, '<br>');
            
            messageDiv.appendChild(contentDiv);
            
            // Add sources if available (for assistant messages)
            if (sources && sources.length > 0 && type === 'assistant') {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'rag-message-sources';
                sourcesDiv.style.cssText = `
                    margin-top: 8px;
                    padding: 8px;
                    background: #f8f9fa;
                    border-radius: 6px;
                    font-size: 11px;
                    color: #6c757d;
                    border-left: 3px solid ${config.primaryColor};
                `;
                
                const sourcesTitle = document.createElement('div');
                sourcesTitle.style.cssText = 'font-weight: 600; margin-bottom: 4px;';
                sourcesTitle.textContent = '📚 Sources:';
                sourcesDiv.appendChild(sourcesTitle);
                
                sources.slice(0, 2).forEach((source, index) => {
                    const sourceItem = document.createElement('div');
                    sourceItem.style.cssText = 'margin-bottom: 2px;';
                    sourceItem.textContent = `• ${source.source_document}`;
                    sourcesDiv.appendChild(sourceItem);
                });
                
                if (sources.length > 2) {
                    const moreDiv = document.createElement('div');
                    moreDiv.style.cssText = 'font-style: italic;';
                    moreDiv.textContent = `+ ${sources.length - 2} more sources`;
                    sourcesDiv.appendChild(moreDiv);
                }
                
                messageDiv.appendChild(sourcesDiv);
            }
            
            chatMessages.insertBefore(messageDiv, typingIndicator);
            scrollToBottom();

            // Show notification if chat is closed
            if (!isOpen && type === 'assistant') {
                notificationBadge.style.display = 'flex';
                notificationBadge.textContent = '1';
            }
        }

        // Show typing indicator
        function showTypingIndicator() {
            typingIndicator.style.display = 'block';
            scrollToBottom();
        }

        // Hide typing indicator
        function hideTypingIndicator() {
            typingIndicator.style.display = 'none';
        }

        // Scroll to bottom of messages
        function scrollToBottom() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Save chat history to localStorage
        function saveChatHistory() {
            const key = `ragChatHistory_${config.apiKey}`;
            localStorage.setItem(key, JSON.stringify(messageHistory));
        }

        // Load chat history from localStorage
        function loadChatHistory() {
            const key = `ragChatHistory_${config.apiKey}`;
            const saved = localStorage.getItem(key);
            if (saved) {
                try {
                    messageHistory = JSON.parse(saved);
                } catch (e) {
                    console.error('Failed to load chat history:', e);
                    messageHistory = [];
                }
            }
        }

        // Expose API for external control
        window.ragWidget = {
            open: () => {
                isOpen = false;
                toggleChat();
            },
            close: closeChat,
            sendMessage: (message) => {
                chatInput.value = message;
                sendMessage();
            },
            clear: () => {
                messageHistory = [];
                const messages = chatMessages.querySelectorAll('.rag-message');
                messages.forEach(msg => {
                    if (!msg.querySelector('.rag-message-content').textContent.includes(config.welcomeMessage)) {
                        msg.remove();
                    }
                });
                saveChatHistory();
            },
            config: config
        };
    }

    // Load widget when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createWidget);
    } else {
        createWidget();
    }

})();