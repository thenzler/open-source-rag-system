/**
 * WordPress RAG Widget Loader
 * This file loads the RAG widget with WordPress-specific configuration
 */

(function() {
    'use strict';

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initWidget);
    } else {
        initWidget();
    }

    function initWidget() {
        // Check if configuration is available
        if (typeof ragWidgetConfig === 'undefined') {
            console.error('RAG Widget: Configuration not found');
            return;
        }

        // Validate required configuration
        if (!ragWidgetConfig.apiKey || !ragWidgetConfig.apiUrl) {
            console.error('RAG Widget: API key and URL are required');
            return;
        }

        // Create widget script element
        const script = document.createElement('script');
        script.src = ragWidgetConfig.apiUrl + '/widget-loader.js';
        
        // Set data attributes from WordPress configuration
        script.setAttribute('data-api-key', ragWidgetConfig.apiKey);
        script.setAttribute('data-api-url', ragWidgetConfig.apiUrl);
        script.setAttribute('data-theme', ragWidgetConfig.theme || 'default');
        script.setAttribute('data-position', ragWidgetConfig.position || 'bottom-right');
        script.setAttribute('data-primary-color', ragWidgetConfig.primaryColor || '#667eea');
        script.setAttribute('data-secondary-color', ragWidgetConfig.secondaryColor || '#764ba2');
        script.setAttribute('data-title', ragWidgetConfig.title || 'AI Assistant');
        script.setAttribute('data-welcome-message', ragWidgetConfig.welcomeMessage || 'Hello! How can I help you today?');
        script.setAttribute('data-placeholder', ragWidgetConfig.placeholder || 'Type your message...');
        script.setAttribute('data-z-index', ragWidgetConfig.zIndex || '999999');

        // Handle script loading
        script.onload = function() {
            console.log('RAG Widget loaded successfully');
            
            // WordPress-specific event handling
            if (typeof jQuery !== 'undefined') {
                jQuery(document).trigger('rag-widget-loaded');
            }
        };

        script.onerror = function() {
            console.error('RAG Widget: Failed to load widget script from ' + ragWidgetConfig.apiUrl);
        };

        // Append script to head
        document.head.appendChild(script);
    }
})();