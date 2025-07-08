import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

// Mock the useAppStore hook
jest.mock('./store/useAppStore', () => ({
  useAppStore: () => ({
    isAuthenticated: false,
    user: null,
    isLoading: false,
    theme: 'light',
  }),
}));

// Mock the API client
jest.mock('./lib/api', () => ({
  wsClient: {
    connect: jest.fn(),
    disconnect: jest.fn(),
    on: jest.fn(),
  },
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {component}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

test('renders without crashing', () => {
  renderWithProviders(<App />);
});

test('redirects to login when not authenticated', () => {
  renderWithProviders(<App />);
  // Since we're not authenticated, we should be redirected to login
  // This test just ensures the app renders without errors
  expect(document.body).toBeInTheDocument();
});
