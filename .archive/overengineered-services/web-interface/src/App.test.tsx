import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

// Mock the hooks and components that require external dependencies
jest.mock('./hooks/useToast', () => ({
  useToast: () => ({ toast: jest.fn() })
}));

jest.mock('./lib/api', () => ({
  wsClient: {
    connect: jest.fn(),
    disconnect: jest.fn(),
    on: jest.fn(),
  }
}));

jest.mock('./store/useAppStore', () => ({
  useAppStore: (selector: any) => {
    const mockState = {
      isAuthenticated: false,
      user: null,
      isLoading: false,
      theme: 'light'
    };
    return selector ? selector(mockState) : mockState;
  }
}));

const renderWithProviders = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {ui}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

test('renders app without crashing', () => {
  renderWithProviders(<App />);
  expect(document.body).toBeInTheDocument();
});
