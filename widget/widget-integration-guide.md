# RAG Chat Widget Integration Guide

## Overview

The RAG Chat Widget is a customizable, embeddable chat interface that allows users to interact with your RAG (Retrieval-Augmented Generation) system on any website. It provides a seamless chat experience with support for various customization options and easy integration.

## Features

- **Easy Integration**: One-line script tag integration
- **Customizable Appearance**: Multiple themes, colors, and positioning options
- **Mobile Responsive**: Works perfectly on all devices
- **Persistent Chat History**: Saves conversation history across sessions
- **Real-time Typing Indicators**: Shows when the assistant is responding
- **Notification System**: Alerts users of new messages when chat is closed
- **Cross-Origin Support**: Works on any domain with proper CORS configuration
- **Accessibility**: Full keyboard navigation and screen reader support
- **No Dependencies**: Pure JavaScript, no external libraries required

## Quick Start

### 1. Basic Integration

Add this single script tag to your website:

```html
<script src="https://yoursite.com/widget-loader.js" 
        data-api-key="YOUR_API_KEY">
</script>
```

### 2. Full Configuration

```html
<script src="https://yoursite.com/widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-api-url="https://your-rag-api.com"
        data-theme="default"
        data-position="bottom-right"
        data-primary-color="#667eea"
        data-secondary-color="#764ba2"
        data-title="AI Assistant"
        data-welcome-message="Hello! How can I help you today?"
        data-placeholder="Type your message..."
        data-z-index="999999">
</script>
```

## Configuration Options

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data-api-key` | Your API key for authentication | `"sk-1234567890abcdef"` |

### Optional Parameters

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `data-api-url` | `http://localhost:8001` | Base URL of your RAG API | Any valid URL |
| `data-theme` | `default` | Widget theme | `default`, `dark`, `blue`, `green`, `purple`, `orange`, `red` |
| `data-position` | `bottom-right` | Widget position | `bottom-right`, `bottom-left`, `top-right`, `top-left` |
| `data-primary-color` | `#667eea` | Primary color (hex) | Any hex color |
| `data-secondary-color` | `#764ba2` | Secondary color (hex) | Any hex color |
| `data-title` | `RAG Assistant` | Chat window title | Any string |
| `data-welcome-message` | `Hello! I'm your RAG assistant...` | Welcome message | Any string |
| `data-placeholder` | `Type your message...` | Input placeholder | Any string |
| `data-auto-load` | `true` | Auto-load widget | `true`, `false` |
| `data-z-index` | `999999` | CSS z-index | Any number |

## Advanced Integration

### Using the JavaScript API

```javascript
// Initialize the API client
const ragApi = new RagApi('YOUR_API_KEY', 'https://your-rag-api.com');

// Send a message
const response = await ragApi.chat('Hello, world!');
console.log(response.response);

// Listen for events
ragApi.on('message', (data) => {
    console.log('New message:', data);
});

ragApi.on('error', (error) => {
    console.error('Error:', error);
});

// Upload a document
const file = document.getElementById('fileInput').files[0];
const uploadResponse = await ragApi.uploadDocument(file);

// Search documents
const searchResults = await ragApi.searchDocuments('search query');
```

### Widget Controller

```javascript
// Initialize widget controller
const widget = new RagWidget('YOUR_API_KEY', {
    apiUrl: 'https://your-rag-api.com',
    theme: 'dark',
    position: 'bottom-left',
    title: 'Custom Assistant'
});

// Control widget programmatically
widget.open();
widget.close();
widget.toggle();
widget.sendMessage('Hello from code!');
widget.clearHistory();
```

## Customization

### CSS Customization

Include the custom CSS file for advanced styling:

```html
<link rel="stylesheet" href="https://yoursite.com/widget-styles.css">
```

Override CSS variables for easy theming:

```css
:root {
    --rag-primary-color: #your-color;
    --rag-secondary-color: #your-color;
    --rag-background-color: #your-color;
    --rag-text-color: #your-color;
}
```

### Custom Themes

Create custom themes by setting data attributes:

```html
<script src="widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-theme="custom"
        data-primary-color="#ff6b6b"
        data-secondary-color="#4ecdc4">
</script>
```

## API Endpoints

Your RAG API should implement these endpoints:

### Chat Endpoint
```http
POST /api/chat
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
    "query": "User's message",
    "chat_history": [
        {"query": "Previous question", "response": "Previous answer"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
}
```

Response:
```json
{
    "response": "Assistant's response",
    "context": ["relevant", "context", "chunks"],
    "confidence": 0.95
}
```

### Document Upload Endpoint
```http
POST /api/upload
Content-Type: multipart/form-data
Authorization: Bearer YOUR_API_KEY

file: [FILE_DATA]
metadata: {"title": "Document Title"}
```

### Status Endpoint
```http
GET /api/status
Authorization: Bearer YOUR_API_KEY
```

Response:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600
}
```

## CORS Configuration

Update your API's CORS settings to allow the widget to make requests:

```python
# FastAPI example
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

## Events

The widget emits various events that you can listen to:

```javascript
// Access the widget instance
const widget = window.ragWidget;

// Listen for widget events
widget.on('open', () => console.log('Widget opened'));
widget.on('close', () => console.log('Widget closed'));
widget.on('message', (data) => console.log('Message sent:', data));
widget.on('response', (data) => console.log('Response received:', data));
widget.on('error', (error) => console.error('Error:', error));
```

## Mobile Optimization

The widget is fully responsive and includes mobile-specific optimizations:

- Touch-friendly interface
- Optimized for small screens
- Full-screen mode on mobile devices
- Gesture support for opening/closing

## Accessibility

The widget includes full accessibility support:

- Keyboard navigation
- Screen reader compatibility
- Focus management
- ARIA labels and roles
- High contrast mode support

## Security

### API Key Security
- Never expose your API key in client-side code
- Use environment variables or secure configuration
- Implement rate limiting on your API
- Consider using JWT tokens for enhanced security

### Content Security Policy
Add the widget domain to your CSP:

```html
<meta http-equiv="Content-Security-Policy" 
      content="script-src 'self' https://yoursite.com; 
               connect-src 'self' https://your-rag-api.com;">
```

## Troubleshooting

### Common Issues

1. **Widget not loading**
   - Check if the script URL is correct
   - Verify API key is valid
   - Check browser console for errors

2. **API requests failing**
   - Verify CORS is configured correctly
   - Check API endpoint URLs
   - Ensure API key has proper permissions

3. **Styling issues**
   - Check for CSS conflicts
   - Verify z-index values
   - Use browser developer tools to debug

### Debug Mode

Enable debug mode by adding:

```html
<script>
window.ragWidgetDebug = true;
</script>
```

## Performance Optimization

### Lazy Loading

Load the widget only when needed:

```html
<script>
// Load widget when user scrolls or clicks
function loadWidget() {
    const script = document.createElement('script');
    script.src = 'https://yoursite.com/widget-loader.js';
    script.setAttribute('data-api-key', 'YOUR_API_KEY');
    document.head.appendChild(script);
}

// Load on user interaction
document.addEventListener('scroll', loadWidget, { once: true });
document.addEventListener('click', loadWidget, { once: true });
</script>
```

### Caching

Implement proper caching headers for widget files:

```http
Cache-Control: public, max-age=3600
ETag: "widget-v1.0.0"
```

## Examples

### E-commerce Site
```html
<script src="https://yoursite.com/widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-theme="blue"
        data-title="Shopping Assistant"
        data-welcome-message="Hi! I can help you find products and answer questions."
        data-placeholder="Ask about products, orders, or returns...">
</script>
```

### Documentation Site
```html
<script src="https://yoursite.com/widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-theme="green"
        data-title="Documentation Helper"
        data-welcome-message="I can help you find information in our docs!"
        data-placeholder="Search documentation...">
</script>
```

### SaaS Application
```html
<script src="https://yoursite.com/widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-theme="purple"
        data-position="bottom-left"
        data-title="Support Assistant"
        data-welcome-message="Need help? I'm here to assist you!"
        data-placeholder="Describe your issue...">
</script>
```

## Support

For technical support and questions:

1. Check the troubleshooting section
2. Review the API documentation
3. Test with the provided examples
4. Contact support with detailed error information

## License

This widget is open source and available under the MIT License.

## Changelog

### Version 1.0.0
- Initial release
- Basic chat functionality
- Multiple themes
- Mobile responsive design
- Accessibility support
- API integration