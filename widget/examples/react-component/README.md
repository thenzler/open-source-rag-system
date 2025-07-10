# RAG Widget React Component

A React component for integrating the RAG Chat Widget into React applications.

## Installation

1. Copy the `RagWidget.jsx` and `RagWidget.css` files to your React project
2. Import the component in your React application

```jsx
import RagWidget from './components/RagWidget';
```

## Basic Usage

```jsx
import React from 'react';
import RagWidget from './components/RagWidget';

function App() {
  return (
    <div className="App">
      <h1>My Application</h1>
      <p>Your app content here...</p>
      
      <RagWidget
        apiKey="your-api-key-here"
        apiUrl="http://localhost:8001"
        title="Help Assistant"
        welcomeMessage="Hi! How can I help you today?"
      />
    </div>
  );
}

export default App;
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `apiKey` | string | **Required** | Your RAG API key |
| `apiUrl` | string | `'http://localhost:8001'` | Base URL of your RAG API |
| `theme` | string | `'default'` | Widget theme (`default`, `dark`, `blue`, `green`, `purple`, `orange`, `red`) |
| `position` | string | `'bottom-right'` | Widget position (`bottom-right`, `bottom-left`, `top-right`, `top-left`) |
| `primaryColor` | string | `'#667eea'` | Primary color (hex code) |
| `secondaryColor` | string | `'#764ba2'` | Secondary color (hex code) |
| `title` | string | `'AI Assistant'` | Widget title |
| `welcomeMessage` | string | `'Hello! How can I help you today?'` | Initial welcome message |
| `placeholder` | string | `'Type your message...'` | Input placeholder text |
| `zIndex` | number | `999999` | CSS z-index value |
| `onMessage` | function | `undefined` | Callback when user sends a message |
| `onResponse` | function | `undefined` | Callback when assistant responds |
| `onError` | function | `undefined` | Callback when an error occurs |
| `onOpen` | function | `undefined` | Callback when widget opens |
| `onClose` | function | `undefined` | Callback when widget closes |

## Advanced Usage

```jsx
import React, { useState } from 'react';
import RagWidget from './components/RagWidget';

function App() {
  const [messages, setMessages] = useState([]);

  const handleMessage = (message) => {
    console.log('User sent:', message);
    setMessages(prev => [...prev, { type: 'user', content: message }]);
  };

  const handleResponse = (response) => {
    console.log('Assistant responded:', response);
    setMessages(prev => [...prev, { type: 'assistant', content: response.response }]);
  };

  const handleError = (error) => {
    console.error('Widget error:', error);
    // Handle error (show notification, etc.)
  };

  return (
    <div className="App">
      <h1>Advanced RAG Widget Example</h1>
      
      <div className="chat-log">
        <h2>Chat Log</h2>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            <strong>{msg.type}:</strong> {msg.content}
          </div>
        ))}
      </div>

      <RagWidget
        apiKey="your-api-key-here"
        apiUrl="https://your-rag-api.com"
        theme="blue"
        position="bottom-left"
        primaryColor="#2196F3"
        secondaryColor="#1976D2"
        title="Smart Assistant"
        welcomeMessage="Welcome! I'm here to help you with any questions about our documentation."
        placeholder="Ask me anything..."
        onMessage={handleMessage}
        onResponse={handleResponse}
        onError={handleError}
        onOpen={() => console.log('Widget opened')}
        onClose={() => console.log('Widget closed')}
      />
    </div>
  );
}

export default App;
```

## Theming

The component supports several built-in themes:

```jsx
// Built-in themes
<RagWidget theme="default" />
<RagWidget theme="dark" />
<RagWidget theme="blue" />
<RagWidget theme="green" />
<RagWidget theme="purple" />
<RagWidget theme="orange" />
<RagWidget theme="red" />

// Custom colors
<RagWidget 
  primaryColor="#FF6B6B"
  secondaryColor="#4ECDC4"
/>
```

## Customization

You can customize the widget's appearance by modifying the CSS variables or overriding the CSS classes:

```css
/* Custom styles */
.rag-widget.custom-theme {
  --rag-primary-color: #your-color;
  --rag-secondary-color: #your-color;
}

.rag-widget.custom-theme .rag-chat-window {
  border-radius: 20px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
}
```

## Event Handling

The component provides several callback props for handling events:

```jsx
<RagWidget
  onMessage={(message) => {
    // Called when user sends a message
    console.log('User message:', message);
  }}
  onResponse={(response) => {
    // Called when assistant responds
    console.log('Assistant response:', response);
    console.log('Confidence:', response.confidence);
    console.log('Sources:', response.context);
  }}
  onError={(error) => {
    // Called when an error occurs
    console.error('Error:', error);
  }}
  onOpen={() => {
    // Called when widget opens
    console.log('Widget opened');
  }}
  onClose={() => {
    // Called when widget closes
    console.log('Widget closed');
  }}
/>
```

## State Management

The component manages its own internal state, including:

- Messages history
- Input value
- Loading states
- Widget open/closed state

Chat history is automatically saved to `localStorage` and restored on page reload.

## Accessibility

The component includes accessibility features:

- Keyboard navigation
- ARIA labels
- Focus management
- Screen reader support

## Mobile Support

The widget is fully responsive and includes mobile-specific optimizations:

- Touch-friendly interface
- Responsive sizing
- Mobile-specific positioning

## Dependencies

The component has no external dependencies beyond React. It uses:

- React hooks (useState, useEffect, useRef)
- CSS for styling
- localStorage for persistence

## Browser Support

The component works in all modern browsers that support:

- CSS Grid and Flexbox
- Fetch API
- localStorage
- CSS custom properties

## TypeScript Support

If you're using TypeScript, you can create a type definition file:

```typescript
// RagWidget.d.ts
import React from 'react';

interface RagWidgetProps {
  apiKey: string;
  apiUrl?: string;
  theme?: 'default' | 'dark' | 'blue' | 'green' | 'purple' | 'orange' | 'red';
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  primaryColor?: string;
  secondaryColor?: string;
  title?: string;
  welcomeMessage?: string;
  placeholder?: string;
  zIndex?: number;
  onMessage?: (message: string) => void;
  onResponse?: (response: any) => void;
  onError?: (error: Error) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

declare const RagWidget: React.FC<RagWidgetProps>;
export default RagWidget;
```

## Examples

See the `examples` directory for more usage examples:

- Basic integration
- Custom theming
- Event handling
- State management
- TypeScript usage