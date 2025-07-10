# 🎯 RAG Widget Integration Guide

Complete guide for integrating the RAG chat widget into any website or application.

## 🚀 Quick Start

### Basic Integration (1 Line of Code)

Add this single line to your HTML:

```html
<script src="widget-loader.js" data-api-key="YOUR_API_KEY" data-api-url="http://localhost:8001"></script>
```

That's it! The widget will appear as a chat bubble in the bottom-right corner.

### Advanced Configuration

```html
<script src="widget-loader.js" 
        data-api-key="your-api-key"
        data-api-url="http://localhost:8001"
        data-theme="dark"
        data-position="bottom-right"
        data-title="Support Assistant"
        data-welcome="Hi! How can I help you today?"
        data-primary-color="#007bff"
        data-secondary-color="#6c757d">
</script>
```

## 📋 Configuration Reference

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data-api-key` | Your API authentication key | `"demo-key-123"` |
| `data-api-url` | URL of your RAG API server | `"http://localhost:8001"` |

### Optional Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `data-theme` | `default` | `default`, `dark`, `blue`, `green`, `purple`, `orange`, `red` | Widget color theme |
| `data-position` | `bottom-right` | `bottom-right`, `bottom-left`, `top-right`, `top-left` | Widget position |
| `data-title` | `"Assistant"` | Any string | Chat header title |
| `data-welcome` | `"Hi! How can..."` | Any string | Welcome message |
| `data-primary-color` | Theme-based | Hex color | Primary color override |
| `data-secondary-color` | Theme-based | Hex color | Secondary color override |

## 🎨 Themes & Styling

### Built-in Themes

#### Default Theme
```html
<script src="widget-loader.js" data-theme="default"></script>
```
- **Colors**: Blue/purple gradient
- **Style**: Modern, friendly
- **Use case**: General purpose

#### Dark Theme
```html
<script src="widget-loader.js" data-theme="dark"></script>
```
- **Colors**: Dark grey/black
- **Style**: Professional, sleek
- **Use case**: Corporate websites

#### Green Theme
```html
<script src="widget-loader.js" data-theme="green"></script>
```
- **Colors**: Green accent
- **Style**: Success, support
- **Use case**: Support pages

#### Blue Theme
```html
<script src="widget-loader.js" data-theme="blue"></script>
```
- **Colors**: Corporate blue
- **Style**: Trust, professional
- **Use case**: Business websites

#### Purple Theme
```html
<script src="widget-loader.js" data-theme="purple"></script>
```
- **Colors**: Creative purple
- **Style**: Modern, creative
- **Use case**: Creative agencies

#### Orange Theme
```html
<script src="widget-loader.js" data-theme="orange"></script>
```
- **Colors**: Energetic orange
- **Style**: Bold, energetic
- **Use case**: Marketing sites

#### Red Theme
```html
<script src="widget-loader.js" data-theme="red"></script>
```
- **Colors**: Urgent red
- **Style**: Important, urgent
- **Use case**: Alert systems

### Custom Colors

Override any theme with custom colors:

```html
<script src="widget-loader.js"
        data-theme="dark"
        data-primary-color="#ff6b6b"
        data-secondary-color="#4ecdc4">
</script>
```

### CSS Customization

Target widget elements with custom CSS:

```css
/* Widget container */
.rag-widget-container {
    z-index: 9999 !important;
}

/* Chat bubble */
.rag-chat-bubble {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}

/* Chat window */
.rag-chat-window {
    border-radius: 20px !important;
    border: 2px solid #667eea !important;
}

/* Messages */
.rag-message.user {
    background: #007bff !important;
}

.rag-message.assistant {
    background: #f8f9fa !important;
    border-left: 4px solid #28a745 !important;
}

/* Source citations */
.rag-message-sources {
    background: #e9ecef !important;
    border-left: 3px solid #fd7e14 !important;
}
```

## 🔧 Platform Integration Examples

### WordPress

#### Method 1: Theme Integration
Add to your theme's `functions.php`:

```php
function add_rag_widget() {
    $api_key = get_option('rag_api_key', 'demo-key-123');
    $api_url = get_option('rag_api_url', 'http://localhost:8001');
    ?>
    <script src="<?php echo get_template_directory_uri(); ?>/widget-loader.js" 
            data-api-key="<?php echo esc_attr($api_key); ?>"
            data-api-url="<?php echo esc_url($api_url); ?>"
            data-theme="default"
            data-title="Support Assistant">
    </script>
    <?php
}
add_action('wp_footer', 'add_rag_widget');
```

#### Method 2: Plugin
Use the included WordPress plugin in `examples/wordpress-plugin/`.

### React Integration

```jsx
import { useEffect } from 'react';

const RAGWidget = ({ apiKey, apiUrl, theme = 'default' }) => {
    useEffect(() => {
        // Remove existing widget
        const existingScript = document.querySelector('script[data-rag-widget]');
        if (existingScript) {
            existingScript.remove();
        }

        // Add new widget
        const script = document.createElement('script');
        script.src = '/widget-loader.js';
        script.setAttribute('data-rag-widget', 'true');
        script.setAttribute('data-api-key', apiKey);
        script.setAttribute('data-api-url', apiUrl);
        script.setAttribute('data-theme', theme);
        document.body.appendChild(script);
        
        return () => {
            const scriptToRemove = document.querySelector('script[data-rag-widget]');
            if (scriptToRemove) {
                scriptToRemove.remove();
            }
            // Clean up widget DOM
            const widget = document.querySelector('.rag-widget-container');
            if (widget) {
                widget.remove();
            }
        };
    }, [apiKey, apiUrl, theme]);
    
    return null; // Widget is added to body
};

// Usage
function App() {
    return (
        <div>
            <h1>My App</h1>
            <RAGWidget 
                apiKey="your-api-key" 
                apiUrl="http://localhost:8001"
                theme="dark" 
            />
        </div>
    );
}
```

### Vue.js Integration

```vue
<template>
  <div>
    <!-- Your app content -->
  </div>
</template>

<script>
export default {
  name: 'App',
  mounted() {
    this.loadRAGWidget();
  },
  beforeDestroy() {
    this.removeRAGWidget();
  },
  methods: {
    loadRAGWidget() {
      const script = document.createElement('script');
      script.src = '/widget-loader.js';
      script.setAttribute('data-api-key', 'your-api-key');
      script.setAttribute('data-api-url', 'http://localhost:8001');
      script.setAttribute('data-theme', 'blue');
      document.body.appendChild(script);
    },
    removeRAGWidget() {
      const widget = document.querySelector('.rag-widget-container');
      if (widget) {
        widget.remove();
      }
    }
  }
}
</script>
```

### Angular Integration

```typescript
import { Component, OnInit, OnDestroy } from '@angular/core';

@Component({
  selector: 'app-root',
  template: `
    <div>
      <!-- Your app content -->
    </div>
  `
})
export class AppComponent implements OnInit, OnDestroy {
  
  ngOnInit() {
    this.loadRAGWidget();
  }
  
  ngOnDestroy() {
    this.removeRAGWidget();
  }
  
  private loadRAGWidget() {
    const script = document.createElement('script');
    script.src = '/assets/widget-loader.js';
    script.setAttribute('data-api-key', 'your-api-key');
    script.setAttribute('data-api-url', 'http://localhost:8001');
    script.setAttribute('data-theme', 'purple');
    document.body.appendChild(script);
  }
  
  private removeRAGWidget() {
    const widget = document.querySelector('.rag-widget-container');
    if (widget) {
      widget.remove();
    }
  }
}
```

### Static HTML

```html
<!DOCTYPE html>
<html>
<head>
    <title>My Website</title>
</head>
<body>
    <h1>Welcome to My Website</h1>
    <p>Content goes here...</p>
    
    <!-- RAG Widget - place before closing body tag -->
    <script src="widget-loader.js" 
            data-api-key="demo-key-123"
            data-api-url="http://localhost:8001"
            data-theme="default"
            data-position="bottom-right"
            data-title="Help Assistant"
            data-welcome="Hi! I can help you find information. What would you like to know?">
    </script>
</body>
</html>
```

## 📱 Mobile Optimization

The widget is fully responsive and mobile-optimized:

### Features
- **Touch-friendly**: Large touch targets for mobile
- **Responsive sizing**: Adapts to screen size
- **Mobile gestures**: Swipe to close on mobile
- **Keyboard handling**: Proper mobile keyboard support

### Mobile-specific CSS
```css
@media (max-width: 768px) {
    .rag-chat-window {
        width: 95vw !important;
        height: 80vh !important;
        bottom: 10px !important;
        right: 2.5vw !important;
    }
    
    .rag-chat-bubble {
        bottom: 20px !important;
        right: 20px !important;
    }
}
```

## 🔌 API Integration

### Expected Response Format

The widget expects this response format from your API:

```json
{
  "response": "Here's the answer to your question...",
  "query": "What was the user's question?",
  "context": [
    {
      "source_document": "document.pdf",
      "content": "Relevant text from the document...",
      "similarity_score": 0.95
    }
  ],
  "confidence": 0.87,
  "processing_time": 2.3
}
```

### Error Handling

The widget handles various error scenarios gracefully:

```json
// API Error Response
{
  "detail": "Error message describing what went wrong"
}
```

Common error types:
- **Network errors**: Connection timeout, server unavailable
- **API errors**: 400, 404, 500 status codes
- **Rate limiting**: 429 status code
- **Validation errors**: Invalid request format

### Custom Error Messages

```javascript
// The widget will show specific error messages based on the error type
const errorMessages = {
    'fetch': 'Cannot connect to the server. Please check if the API is running.',
    '400': 'Invalid request format.',
    '404': 'API endpoint not found.',
    '429': 'Too many requests. Please wait a moment.',
    '500': 'Server error occurred.'
};
```

## 🛠️ JavaScript API

### Programmatic Control

Access the widget programmatically:

```javascript
// Open widget
window.ragWidget.open();

// Close widget
window.ragWidget.close();

// Send message programmatically
window.ragWidget.sendMessage("Tell me about renewable energy");

// Check if widget is open
const isOpen = window.ragWidget.isOpen();

// Get chat history
const history = window.ragWidget.getHistory();

// Clear chat history
window.ragWidget.clearHistory();
```

### Event Listeners

Listen to widget events:

```javascript
// Widget loaded
window.addEventListener('ragWidgetLoaded', () => {
    console.log('RAG widget is ready');
});

// Message sent
window.addEventListener('ragMessageSent', (event) => {
    console.log('User message:', event.detail.message);
});

// Response received
window.addEventListener('ragResponseReceived', (event) => {
    console.log('Assistant response:', event.detail.response);
});

// Widget opened/closed
window.addEventListener('ragWidgetToggled', (event) => {
    console.log('Widget is now:', event.detail.isOpen ? 'open' : 'closed');
});
```

### Custom Initialization

```javascript
// Wait for DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Custom widget configuration
    window.ragWidgetConfig = {
        apiKey: 'your-dynamic-key',
        apiUrl: 'https://your-domain.com/api',
        theme: 'custom',
        primaryColor: '#ff6b6b',
        secondaryColor: '#4ecdc4',
        onReady: () => {
            console.log('Widget ready!');
        },
        onError: (error) => {
            console.error('Widget error:', error);
        }
    };
    
    // Load widget script
    const script = document.createElement('script');
    script.src = 'widget-loader.js';
    document.body.appendChild(script);
});
```

## 🔒 Security Best Practices

### API Key Management

```javascript
// DON'T: Expose real API keys in client-side code
const apiKey = "sk-real-secret-key-12345"; // ❌ Visible to users

// DO: Use public API keys or implement server-side proxy
const apiKey = "pk-public-key-67890"; // ✅ Safe for client-side

// BEST: Server-side proxy endpoint
const apiUrl = "/api/proxy/rag"; // ✅ Hides real API details
```

### Content Security Policy (CSP)

Add CSP headers to allow widget loading:

```html
<meta http-equiv="Content-Security-Policy" 
      content="script-src 'self' 'unsafe-inline'; 
               connect-src 'self' http://localhost:8001;
               style-src 'self' 'unsafe-inline';">
```

### HTTPS in Production

Always use HTTPS for production deployments:

```html
<!-- Production -->
<script src="https://yourcdn.com/widget-loader.js" 
        data-api-url="https://api.yourdomain.com">
</script>
```

## 📊 Performance Optimization

### Lazy Loading

The widget loads asynchronously by default:

```javascript
// Widget loads after page content
window.addEventListener('load', () => {
    // Widget initialization happens here
});
```

### Caching

Enable browser caching for widget assets:

```nginx
# Nginx configuration
location /widget/ {
    expires 1d;
    add_header Cache-Control "public, immutable";
}
```

### Bundle Size

Widget bundle sizes:
- **widget-loader.js**: ~15KB (gzipped: ~5KB)
- **widget-styles.css**: ~8KB (gzipped: ~2KB)
- **Total overhead**: ~23KB (~7KB gzipped)

### Performance Monitoring

```javascript
// Monitor widget performance
window.addEventListener('ragResponseReceived', (event) => {
    const responseTime = event.detail.processing_time;
    console.log(`Response time: ${responseTime}s`);
    
    // Send to analytics
    if (window.gtag) {
        gtag('event', 'rag_response_time', {
            value: Math.round(responseTime * 1000),
            custom_parameter: 'widget_performance'
        });
    }
});
```

## 🧪 Testing & Debugging

### Debug Mode

Enable detailed logging:

```html
<script src="widget-loader.js" 
        data-debug="true"
        data-api-key="test-key">
</script>
```

Debug output includes:
- API request/response details
- Widget lifecycle events
- Error stack traces
- Performance metrics

### Test Environment

Use the test server for development:

```bash
# Start test server
python test_widget_server.py

# Test endpoints
curl http://localhost:3000/widget/test-standalone.html
```

### Browser Console Testing

Test widget functionality in browser console:

```javascript
// Send test message
window.ragWidget.sendMessage("Test question");

// Check widget state
console.log('Widget open:', window.ragWidget.isOpen());
console.log('Chat history:', window.ragWidget.getHistory());

// Test API endpoint directly
fetch('http://localhost:8001/api/v1/query/optimized', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: 'Test query'})
})
.then(r => r.json())
.then(console.log);
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Widget Not Appearing
**Symptoms**: No chat bubble visible
**Causes**: 
- JavaScript errors
- Incorrect file paths
- CSS conflicts

**Solutions**:
```javascript
// Check for errors
console.error('Check browser console for errors');

// Verify file paths
console.log('Widget script loaded:', !!window.ragWidget);

// Check CSS conflicts
document.querySelector('.rag-widget-container').style.zIndex = '999999';
```

#### 2. API Connection Errors
**Symptoms**: "Cannot connect to server" messages
**Causes**:
- Wrong API URL
- Server not running
- CORS issues

**Solutions**:
```bash
# Verify API server
curl http://localhost:8001/api/status

# Check CORS headers
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS \
     http://localhost:8001/api/v1/query/optimized
```

#### 3. CORS Errors
**Symptoms**: "Cross-Origin Request Blocked" in console
**Solutions**:
- Use test server: `python test_widget_server.py`
- Configure CORS on API server
- Use server-side proxy

#### 4. Widget Styling Issues
**Symptoms**: Widget looks broken or overlaps content
**Solutions**:
```css
/* Fix z-index conflicts */
.rag-widget-container {
    z-index: 2147483647 !important;
}

/* Fix positioning issues */
.rag-chat-window {
    position: fixed !important;
    bottom: 80px !important;
    right: 20px !important;
}
```

#### 5. Mobile Issues
**Symptoms**: Widget not responsive on mobile
**Solutions**:
```css
@media (max-width: 768px) {
    .rag-chat-window {
        width: calc(100vw - 20px) !important;
        height: calc(100vh - 100px) !important;
        bottom: 10px !important;
        right: 10px !important;
        left: 10px !important;
    }
}
```

### Debug Tools

#### Network Tab Analysis
1. Open browser DevTools (F12)
2. Go to Network tab
3. Trigger widget interaction
4. Check for failed requests

#### Console Debugging
```javascript
// Enable verbose logging
localStorage.setItem('ragWidgetDebug', 'true');

// Check widget status
console.log('Widget config:', window.ragWidgetConfig);
console.log('Widget instance:', window.ragWidget);
```

#### API Testing
```html
<!-- Standalone API test -->
<script>
async function testAPI() {
    try {
        const response = await fetch('http://localhost:8001/api/status');
        const data = await response.json();
        console.log('API Status:', data);
    } catch (error) {
        console.error('API Error:', error);
    }
}
testAPI();
</script>
```

## 📈 Analytics & Monitoring

### Usage Analytics

Track widget usage:

```javascript
// Track widget interactions
window.addEventListener('ragMessageSent', (event) => {
    // Google Analytics 4
    if (window.gtag) {
        gtag('event', 'rag_query', {
            query_length: event.detail.message.length,
            custom_parameter: 'widget_usage'
        });
    }
    
    // Custom analytics
    fetch('/analytics/widget-usage', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            event: 'message_sent',
            query: event.detail.message,
            timestamp: Date.now()
        })
    });
});
```

### Performance Metrics

Monitor widget performance:

```javascript
// Response time tracking
window.addEventListener('ragResponseReceived', (event) => {
    const metrics = {
        response_time: event.detail.processing_time,
        query_length: event.detail.query?.length || 0,
        source_count: event.detail.context?.length || 0,
        confidence: event.detail.confidence || 0
    };
    
    // Send to monitoring service
    fetch('/metrics/widget-performance', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(metrics)
    });
});
```

### Error Monitoring

Track widget errors:

```javascript
window.addEventListener('ragError', (event) => {
    // Sentry error tracking
    if (window.Sentry) {
        Sentry.captureException(new Error(event.detail.message), {
            tags: {
                component: 'rag_widget',
                error_type: event.detail.type
            }
        });
    }
    
    // Custom error logging
    console.error('RAG Widget Error:', event.detail);
});
```

## 🔄 Updates & Maintenance

### Version Updates

Check for widget updates:

```javascript
// Check widget version
console.log('Widget version:', window.ragWidget?.version);

// Update notification
if (window.ragWidget?.version !== 'latest') {
    console.log('Widget update available');
}
```

### Backward Compatibility

The widget maintains backward compatibility:

- API changes are versioned
- Old configurations continue to work
- Graceful degradation for unsupported features

### Update Process

1. **Backup current widget files**
2. **Download new widget version**
3. **Test in development environment**
4. **Deploy to production**
5. **Verify functionality**

## 📞 Support

### Getting Help

- **Documentation**: This guide and API docs
- **Examples**: Check `examples/` folder
- **Issues**: Create GitHub issues
- **Testing**: Use `test-standalone.html`

### Reporting Bugs

When reporting issues, include:

- Browser version and OS
- Widget configuration
- Console error messages
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests

We welcome feature requests! Please include:

- Use case description
- Proposed implementation
- Benefits and impact
- Willingness to contribute

---

## 📝 Summary

The RAG widget provides a powerful, flexible way to add AI-powered chat to any website. With just one line of code, you can give your users instant access to your document knowledge base.

**Key benefits:**
- ✅ **Easy integration** - One line of code
- ✅ **Highly customizable** - Themes, colors, positioning
- ✅ **Mobile responsive** - Works on all devices
- ✅ **Performance optimized** - Fast, lightweight
- ✅ **Production ready** - Error handling, caching, security

**Get started now:**
```html
<script src="widget-loader.js" data-api-key="YOUR_KEY" data-api-url="YOUR_API"></script>
```

Happy coding! 🚀