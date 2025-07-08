# RAG System Frontend

A modern React TypeScript frontend for the Open Source RAG System.

## 🚀 Features

### **Core Functionality**
- **Document Management**: Upload, view, search, and manage documents with real-time processing status
- **Advanced Search**: Semantic search with advanced filtering, re-ranking, and query expansion
- **Analytics Dashboard**: Comprehensive system metrics, usage analytics, and performance monitoring
- **User Management**: Profile management, settings, and authentication

### **UI/UX Features**
- **Modern Design**: Clean, responsive interface with dark/light theme support
- **Real-time Updates**: WebSocket integration for live document processing updates
- **Drag & Drop**: Intuitive file upload with progress tracking
- **Advanced Filtering**: Multi-faceted search and filtering capabilities
- **Mobile Responsive**: Optimized for all device sizes

### **Technical Features**
- **TypeScript**: Full type safety and modern development experience
- **State Management**: Zustand for global state, React Query for server state
- **Component Library**: Custom UI components with consistent design system
- **Performance**: Optimized with code splitting, lazy loading, and caching

## 🛠️ Technology Stack

- **React 18** - Modern React with concurrent features
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **React Query** - Server state management
- **Zustand** - Client state management
- **React Router** - Client-side routing
- **React Dropzone** - File upload handling
- **Recharts** - Data visualization
- **Radix UI** - Accessible component primitives
- **Framer Motion** - Animation library

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## 🏗️ Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # Base UI components (Button, Input, etc.)
│   ├── layout/         # Layout components (Header, Sidebar)
│   ├── documents/      # Document-related components
│   ├── search/         # Search interface components
│   ├── analytics/      # Analytics and charts
│   └── settings/       # Settings pages
├── hooks/              # Custom React hooks
├── lib/                # Utilities and API client
├── pages/              # Main application pages
├── store/              # Global state management
├── types/              # TypeScript type definitions
└── App.tsx             # Main application component
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=ws://localhost:8000/ws
```

### API Integration

The frontend integrates with the FastAPI backend through:

- **REST API**: All CRUD operations and queries
- **WebSocket**: Real-time updates for document processing
- **File Upload**: Multipart form data for document uploads

## 📱 Key Components

### Document Management
- **DocumentsPage**: Main document library with grid/list view
- **DocumentCard**: Individual document display with actions
- **DocumentUploadModal**: Drag & drop file upload interface
- **DocumentFilters**: Advanced filtering and search

### Search Interface
- **SearchPage**: Main search interface with query builder
- **SearchResults**: Formatted search results with source attribution
- **AdvancedSearchModal**: Advanced query configuration
- **SearchHistory**: Recent searches and saved queries

### Analytics
- **AnalyticsPage**: System metrics and usage analytics
- **AnalyticsChart**: Configurable charts (line, area, bar)
- **MetricCard**: Key performance indicators
- **DateRangePicker**: Time range selection

### Settings
- **SettingsPage**: Tabbed settings interface
- **UserSettings**: Profile and account management
- **SystemSettings**: System configuration display
- **SecuritySettings**: Password and security options
- **NotificationSettings**: Notification preferences

## 🎨 Design System

### Color Palette
- **Primary**: Blue (#3b82f6)
- **Secondary**: Gray (#6b7280)
- **Success**: Green (#10b981)
- **Warning**: Yellow (#f59e0b)
- **Error**: Red (#ef4444)

### Typography
- **Font**: Inter (Google Fonts)
- **Sizes**: 2xs (10px) to 4xl (36px)
- **Weights**: 400 (normal), 500 (medium), 600 (semibold), 700 (bold)

### Components
All components follow a consistent design pattern:
- Responsive design principles
- Dark/light theme support
- Accessibility features (ARIA labels, keyboard navigation)
- Loading and error states

## 🔄 State Management

### Global State (Zustand)
- User authentication state
- Theme preferences
- UI state (sidebar, notifications)

### Server State (React Query)
- Document data caching
- Automatic refetching
- Optimistic updates
- Error handling

## 🚀 Performance Optimizations

- **Code Splitting**: Route-based lazy loading
- **Image Optimization**: Responsive images with lazy loading
- **Bundle Analysis**: Webpack bundle analyzer integration
- **Caching**: Aggressive caching for static assets
- **Compression**: Gzip compression for production builds

## 🧪 Testing

```bash
# Run unit tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

## 🐳 Docker Deployment

The frontend is containerized with multi-stage build:

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **HTTPS Support**: TLS encryption for production
- **CSRF Protection**: Cross-site request forgery prevention
- **Input Validation**: Client-side validation with server-side verification
- **Content Security Policy**: XSS protection headers

## 📈 Analytics Integration

- **Usage Tracking**: User interaction analytics
- **Performance Monitoring**: Core Web Vitals tracking
- **Error Tracking**: Automatic error reporting
- **Custom Events**: Business-specific metrics

## 🌐 Internationalization

Ready for i18n with:
- Extractable text strings
- Date/time formatting
- Number formatting
- RTL layout support

## 🤝 Contributing

1. Follow the existing code style
2. Write tests for new features
3. Update documentation
4. Use semantic commit messages

## 📝 License

MIT License - see [LICENSE](../LICENSE) for details.
