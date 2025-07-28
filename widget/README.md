# RAG Chat Widget

A complete embeddable chat widget solution for RAG (Retrieval-Augmented Generation) systems. Easy to integrate, highly customizable, and works on any website.

## 🚀 Quick Start

Add the widget to any website with just one line of code:

```html
<script src="https://yoursite.com/widget-loader.js" data-api-key="YOUR_API_KEY"></script>
```

## 📁 Files Overview

### Core Widget Files
- **`widget-loader.js`** - Main loader script for easy integration
- **`chat-widget.html`** - Standalone HTML widget with embedded CSS/JS
- **`widget-styles.css`** - Separate CSS file for advanced customization
- **`widget-api.js`** - JavaScript API client for programmatic control
- **`widget-integration-guide.md`** - Comprehensive integration documentation

### Example Integrations
- **`examples/vanilla-example.html`** - Simple HTML integration example
- **`examples/wordpress-plugin/`** - Complete WordPress plugin
- **`examples/react-component/`** - React component implementation

## ✨ Features

- **🎯 Easy Integration**: Single script tag setup
- **🎨 Customizable**: Multiple themes, colors, and positioning options
- **📱 Mobile Responsive**: Works perfectly on all devices
- **💾 Persistent History**: Saves conversations across sessions
- **🔔 Notifications**: Alerts when new messages arrive
- **🌐 Cross-Origin Support**: Works on any domain with proper CORS
- **♿ Accessible**: Full keyboard navigation and screen reader support
- **🚀 No Dependencies**: Pure JavaScript, no external libraries

## 🛠️ Installation

### Method 1: Direct Script Include (Recommended)

```html
<script src="https://yoursite.com/widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-api-url="https://your-rag-api.com">
</script>
```

### Method 2: Self-Hosted

1. Download the widget files to your server
2. Include the script with your configuration:

```html
<script src="/path/to/widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-api-url="https://your-rag-api.com">
</script>
```

### Method 3: WordPress Plugin

1. Install the WordPress plugin from `examples/wordpress-plugin/`
2. Configure through WordPress admin panel
3. Widget automatically appears on your site

## ⚙️ Configuration Options

### Basic Configuration

```html
<script src="widget-loader.js" 
        data-api-key="YOUR_API_KEY"
        data-api-url="https://your-api.com"
        data-theme="default"
        data-position="bottom-right">
</script>
```

### All Configuration Options

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `data-api-key` | **Required** | Your RAG API key | Any string |
| `data-api-url` | `http://localhost:8001` | RAG API base URL | Any valid URL |
| `data-theme` | `default` | Widget theme | `default`, `dark`, `blue`, `green`, `purple`, `orange`, `red` |
| `data-position` | `bottom-right` | Widget position | `bottom-right`, `bottom-left`, `top-right`, `top-left` |
| `data-primary-color` | `#667eea` | Primary color | Any hex color |
| `data-secondary-color` | `#764ba2` | Secondary color | Any hex color |
| `data-title` | `RAG Assistant` | Widget title | Any string |
| `data-welcome-message` | `Hello! I'm your RAG assistant...` | Welcome message | Any string |
| `data-placeholder` | `Type your message...` | Input placeholder | Any string |
| `data-z-index` | `999999` | CSS z-index | Any number |

## 🎨 Customization

### Themes

```html
<!-- Built-in themes -->
<script src="widget-loader.js" data-theme="dark" data-api-key="YOUR_KEY"></script>
<script src="widget-loader.js" data-theme="blue" data-api-key="YOUR_KEY"></script>
<script src="widget-loader.js" data-theme="green" data-api-key="YOUR_KEY"></script>
```

### Custom Colors

```html
<script src="widget-loader.js" 
        data-primary-color="#FF6B6B"
        data-secondary-color="#4ECDC4"
        data-api-key="YOUR_KEY">
</script>
```

### CSS Customization

Include the CSS file for advanced styling:

```html
<link rel="stylesheet" href="widget-styles.css">
```

Override CSS variables:

```css
:root {
  --rag-primary-color: #your-color;
  --rag-secondary-color: #your-color;
  --rag-border-radius: 8px;
  --rag-font-family: 'Your Font', sans-serif;
}
```

## 🔧 API Integration

### Required API Endpoints

Your RAG API should implement these endpoints:

#### Chat Endpoint
```http
POST /api/chat
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "query": "User's message",
  "chat_history": [
    {"query": "Previous question", "response": "Previous answer"}
  ]
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

#### Status Endpoint
```http
GET /api/status
Authorization: Bearer YOUR_API_KEY
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### API Server Setup

Update your API server to support CORS:

```python
# FastAPI example
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

## 🎮 JavaScript API

Control the widget programmatically:

```javascript
// Access widget instance
const widget = window.ragWidget;

// Control widget
widget.open();                    // Open widget
widget.close();                   // Close widget
widget.sendMessage('Hello!');     // Send message
widget.clear();                   // Clear history

// Listen for events
widget.on('message', (data) => {
  console.log('Message sent:', data);
});

widget.on('response', (data) => {
  console.log('Response received:', data);
});

widget.on('error', (error) => {
  console.error('Error:', error);
});
```

## 📱 Mobile Optimization

The widget automatically adapts to mobile devices:

- Touch-friendly interface
- Responsive design
- Full-screen mode on small screens
- Optimized for mobile interactions

## ♿ Accessibility

Full accessibility support included:

- Keyboard navigation (Tab, Enter, Escape)
- Screen reader compatibility
- ARIA labels and roles
- Focus management
- High contrast mode support

## 🔐 Security

### Best Practices

1. **API Key Security**: Never expose your API key in client-side code
2. **CORS Configuration**: Properly configure CORS on your API server
3. **Rate Limiting**: Implement rate limiting on your API endpoints
4. **Input Validation**: Validate all user inputs on the server side

### Content Security Policy

Add the widget domain to your CSP:

```html
<meta http-equiv="Content-Security-Policy" 
      content="script-src 'self' https://yoursite.com; 
               connect-src 'self' https://your-rag-api.com;">
```

## 🚀 Performance

### Optimization Tips

1. **Lazy Loading**: Load the widget only when needed
2. **Caching**: Implement proper caching headers
3. **CDN**: Use a CDN for widget files
4. **Minification**: Minify CSS and JavaScript files

### Lazy Loading Example

```javascript
// Load widget on user interaction
function loadWidget() {
  const script = document.createElement('script');
  script.src = 'widget-loader.js';
  script.setAttribute('data-api-key', 'YOUR_KEY');
  document.head.appendChild(script);
}

// Trigger loading on scroll or click
document.addEventListener('scroll', loadWidget, { once: true });
```

## 🐛 Troubleshooting

### Common Issues

1. **Widget not loading**
   - Check script URL and API key
   - Verify CORS configuration
   - Check browser console for errors

2. **API requests failing**
   - Verify API endpoint URLs
   - Check API key permissions
   - Ensure CORS is properly configured

3. **Styling issues**
   - Check for CSS conflicts
   - Verify z-index values
   - Use browser dev tools to debug

### Debug Mode

Enable debug logging:

```html
<script>
window.ragWidgetDebug = true;
</script>
```

## 📝 Examples

### E-commerce Site
```html
<script src="widget-loader.js" 
        data-api-key="YOUR_KEY"
        data-theme="blue"
        data-title="Shopping Assistant"
        data-welcome-message="Hi! I can help you find products!"
        data-placeholder="Ask about products, orders, or returns...">
</script>
```

### Documentation Site
```html
<script src="widget-loader.js" 
        data-api-key="YOUR_KEY"
        data-theme="green"
        data-title="Documentation Helper"
        data-welcome-message="I can help you find information!"
        data-placeholder="Search documentation...">
</script>
```

### SaaS Application
```html
<script src="widget-loader.js" 
        data-api-key="YOUR_KEY"
        data-theme="purple"
        data-position="bottom-left"
        data-title="Support Assistant"
        data-welcome-message="Need help? I'm here to assist!"
        data-placeholder="Describe your issue...">
</script>
```

## 🔗 Integration Guides

- **[Vanilla HTML](examples/vanilla-example.html)** - Simple HTML integration
- **[WordPress Plugin](examples/wordpress-plugin/)** - Complete WordPress plugin
- **[React Component](examples/react-component/)** - React component implementation
- **[Integration Guide](widget-integration-guide.md)** - Comprehensive documentation

## 📊 Analytics

Track widget usage with built-in events:

```javascript
// Track widget interactions
window.ragWidget.on('message', (data) => {
  // Track user messages
  gtag('event', 'widget_message', {
    event_category: 'widget',
    event_label: data.query
  });
});

window.ragWidget.on('response', (data) => {
  // Track assistant responses
  gtag('event', 'widget_response', {
    event_category: 'widget',
    event_label: data.response
  });
});
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:

1. Check the [Integration Guide](widget-integration-guide.md)
2. Review the [Troubleshooting](#-troubleshooting) section
3. Check existing GitHub issues
4. Create a new issue with detailed information

## 🎯 Roadmap

### Upcoming Features

- [ ] **Multi-language Support**: Internationalization and localization
- [ ] **Voice Integration**: Voice messages and responses
- [ ] **File Upload**: Allow users to upload files in chat
- [ ] **Conversation Export**: Export chat history
- [ ] **Advanced Analytics**: Detailed usage statistics
- [ ] **A/B Testing**: Built-in A/B testing capabilities
- [ ] **Offline Mode**: Basic functionality without internet
- [ ] **Custom Animations**: More animation options
- [ ] **Widget Templates**: Pre-built widget templates
- [ ] **Advanced Theming**: Visual theme builder

### Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Mobile optimizations and accessibility improvements
- **v1.2.0** - WordPress plugin and React component
- **v1.3.0** - Advanced theming and customization options

---

Made with ❤️ for the RAG community