import { render, screen } from '@testing-library/react';
import App from './App';

// Mock react-router-dom
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  BrowserRouter: ({ children }) => <div>{children}</div>,
  Routes: ({ children }) => <div>{children}</div>,
  Route: ({ element }) => element,
  Link: ({ children, to }) => <a href={to}>{children}</a>,
  useLocation: () => ({ pathname: '/' })
}));

// Mock @tanstack/react-query
jest.mock('@tanstack/react-query', () => ({
  QueryClient: jest.fn(() => ({})),
  QueryClientProvider: ({ children }) => children,
  useQuery: () => ({
    data: {
      total_documents: 0,
      total_queries: 0,
      processing_queue_size: 0,
      queries_last_hour: 0
    },
    isLoading: false
  })
}));

// Mock react-hot-toast
jest.mock('react-hot-toast', () => ({
  Toaster: () => <div data-testid="toaster" />
}));

// Mock heroicons
jest.mock('@heroicons/react/24/outline', () => ({
  HomeIcon: () => <div data-testid="home-icon" />,
  DocumentTextIcon: () => <div data-testid="document-icon" />,
  MagnifyingGlassIcon: () => <div data-testid="search-icon" />,
  ChartBarIcon: () => <div data-testid="chart-icon" />,
  CogIcon: () => <div data-testid="cog-icon" />,
  BellIcon: () => <div data-testid="bell-icon" />,
  UserCircleIcon: () => <div data-testid="user-icon" />,
  ClockIcon: () => <div data-testid="clock-icon" />
}));

test('renders RAG System header', () => {
  render(<App />);
  const headerElement = screen.getByText(/RAG System/i);
  expect(headerElement).toBeInTheDocument();
});

test('renders dashboard by default', () => {
  render(<App />);
  const dashboardElement = screen.getByText(/Dashboard/i);
  expect(dashboardElement).toBeInTheDocument();
});

test('renders navigation sidebar', () => {
  render(<App />);
  const documentsLink = screen.getByText(/Documents/i);
  const searchLink = screen.getByText(/Search/i);
  const analyticsLink = screen.getByText(/Analytics/i);
  const settingsLink = screen.getByText(/Settings/i);
  
  expect(documentsLink).toBeInTheDocument();
  expect(searchLink).toBeInTheDocument();
  expect(analyticsLink).toBeInTheDocument();
  expect(settingsLink).toBeInTheDocument();
});
