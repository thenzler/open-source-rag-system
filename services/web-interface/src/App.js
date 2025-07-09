import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import Dashboard from './pages/Dashboard';
import Documents from './pages/Documents';
import Search from './pages/Search';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

import './App.css';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Header />
          <div className="flex">
            <Sidebar />
            <main className="flex-1 p-6">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/documents" element={<Documents />} />
                <Route path="/search" element={<Search />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
          <Toaster position="top-right" />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
