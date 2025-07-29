# TechCorp Solutions Demo Website

A professional demo website showcasing RAG (Retrieval-Augmented Generation) widget integration for testing and demonstration purposes.

## 🎯 Purpose

This demo website demonstrates how to integrate the RAG chat widget into a real-world website, showing different configurations and use cases across multiple pages.

## 📁 Files

- `index.html` - Homepage with API documentation and general widget
- `about.html` - Company information with corporate-themed widget
- `contact.html` - Contact page with support-themed widget
- `styles.css` - Complete styling for the website and widget integration
- `script.js` - Interactive functionality and widget helpers
- `README.md` - This documentation file

## 🚀 Quick Start

### 1. Start the RAG API Server

First, make sure your RAG API is running:

```bash
# From the project root directory
python simple_api.py
```

The API should be running on `http://localhost:8000`

### 2. Upload Some Documents

Use the existing frontend to upload documents to your RAG system:

```bash
# Open the simple frontend
python -m http.server 8080
# Then visit http://localhost:8080/simple_frontend.html
```

Upload some PDF or text documents so the RAG system has content to work with.

### 3. Test the Demo Website

Open the demo website in your browser:

```bash
# From the example-website directory
python -m http.server 3000
# Then visit http://localhost:3000
```

Or simply open `index.html` directly in your browser.

## 🎨 Widget Configurations

### Homepage (index.html)
- **Theme**: Default (blue/purple gradient)
- **Position**: Bottom-right
- **Title**: "Documentation Assistant"
- **Purpose**: General API documentation help

### About Page (about.html)
- **Theme**: Dark (professional corporate)
- **Position**: Bottom-left
- **Title**: "Company Info Assistant"
- **Purpose**: Company and team information

### Contact Page (contact.html)
- **Theme**: Green (support theme)
- **Position**: Bottom-right
- **Title**: "Support Assistant"
- **Purpose**: Customer support and inquiries

## 💬 Test Questions

Try these questions to test the RAG widget functionality:

### Technical Questions (Homepage)
```
How do I integrate the API with React?
What authentication do I need?
Show me a code example
What are the system requirements?
How fast are the responses?
I'm having connection issues
```

### Company Questions (About Page)
```
Tell me about your company
What's your pricing?
Who's on your team?
What are your company values?
How many companies use your service?
```

### Support Questions (Contact Page)
```
How do I get support?
What are your business hours?
Do you offer enterprise pricing?
How do I schedule training?
I need technical help
```

## 🔧 Customization

### Changing Widget Configuration

Edit the widget script tags in each HTML file:

```html
<script src="../widget/widget-loader.js" 
        data-api-key="your-api-key-here"
        data-api-url="http://localhost:8000"
        data-theme="default"
        data-position="bottom-right"
        data-title="Your Custom Title"
        data-welcome="Your custom welcome message">
</script>
```

### Available Themes
- `default` - Blue/purple gradient
- `dark` - Professional dark theme
- `green` - Support/success theme
- `blue` - Blue corporate theme
- `purple` - Purple creative theme
- `orange` - Orange energetic theme
- `red` - Red urgent/important theme

### Available Positions
- `bottom-right` (default)
- `bottom-left`
- `top-right`
- `top-left`

### Custom Colors

You can also use custom colors:

```html
data-primary-color="#your-color"
data-secondary-color="#your-secondary-color"
```

## 🛠️ Integration with Real API

To connect to your actual RAG API:

1. **Update API Configuration**:
   - Replace `data-api-key="demo-key-123"` with your real API key
   - Ensure `data-api-url` points to your RAG API server

2. **Add CORS Support**:
   Make sure your `simple_api.py` has CORS enabled for the domain hosting this website.

3. **Upload Documents**:
   Use the RAG system's document upload feature to add content that users can query.

## 📱 Mobile Responsiveness

The demo website is fully responsive and works on:
- Desktop browsers
- Tablets
- Mobile phones
- Different screen orientations

## 🎯 Features Demonstrated

### Website Features
- ✅ Professional responsive design
- ✅ Multi-page navigation
- ✅ Interactive contact forms
- ✅ Smooth animations
- ✅ Mobile-first approach

### Widget Features
- ✅ Multiple theme configurations
- ✅ Different positioning options
- ✅ Context-aware suggestions
- ✅ Real-time chat interface
- ✅ Persistent chat history
- ✅ Mobile optimization

## 🔍 Testing Checklist

- [ ] All three pages load correctly
- [ ] Navigation works between pages
- [ ] Widget appears on all pages
- [ ] Widget themes are different on each page
- [ ] Chat functionality works
- [ ] Contact form submission works
- [ ] Mobile responsiveness works
- [ ] RAG API responses are received
- [ ] Sources are displayed with answers

## 🚧 Known Limitations

This is a demo website for testing purposes:

1. **Contact Form**: Currently shows success message but doesn't actually send emails
2. **API Integration**: Uses demo API key - replace with real credentials
3. **Content**: Uses placeholder content - customize for your use case
4. **Analytics**: No analytics tracking implemented
5. **SEO**: Basic SEO - would need optimization for production

## 📞 Support

If you encounter issues:

1. **Check API Status**: Ensure your RAG API is running on `http://localhost:8000`
2. **Check Browser Console**: Look for JavaScript errors or network issues
3. **Check CORS**: Ensure the API allows requests from your domain
4. **Test Widget Separately**: Try the widget examples in the `widget/` directory

## 🔄 Next Steps

To make this production-ready:

1. **Replace Demo Content**: Update with your actual company information
2. **Add Real API Keys**: Configure proper authentication
3. **Implement Backend**: Add real contact form processing
4. **Add Analytics**: Implement tracking and monitoring
5. **Optimize Performance**: Add caching and CDN integration
6. **Add SEO**: Implement proper meta tags and structured data
7. **Security**: Add proper input validation and sanitization

## 📚 Additional Resources

- [RAG Widget Documentation](../widget/widget-integration-guide.md)
- [API Documentation](../docs/API.md)
- [Widget Examples](../widget/examples/)
- [Main Project README](../README.md)

---

**Happy Testing!** 🎉

This demo website provides a realistic testing environment for the RAG widget integration. Use it to test different configurations, themes, and integration scenarios before deploying to production websites.