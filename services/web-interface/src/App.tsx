import React, { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAppStore } from './store/useAppStore';
import { wsClient } from './lib/api';
import { Layout } from './components/layout/Layout';
import { ProtectedRoute } from './components/auth/ProtectedRoute';
import { LoginPage } from './pages/auth/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { SearchPage } from './pages/SearchPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { SettingsPage } from './pages/SettingsPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { LoadingSpinner } from './components/ui/loading-spinner';
import { useToast } from './hooks/useToast';

function App() {
  const { isAuthenticated, user, isLoading, theme } = useAppStore();
  const { toast } = useToast();

  useEffect(() => {
    // Initialize WebSocket connection when authenticated
    if (isAuthenticated && user) {
      wsClient.connect();

      // Listen for document processing updates
      wsClient.on('document_processing', (data) => {
        toast({
          title: 'Document Update',
          description: `${data.document_id}: ${data.status}`,
          variant: data.status === 'completed' ? 'success' : 
                   data.status === 'failed' ? 'destructive' : 'info',
        });
      });

      // Listen for system notifications
      wsClient.on('system_notification', (data) => {
        toast({
          title: data.title,
          description: data.message,
          variant: data.type || 'info',
        });
      });

      return () => {
        wsClient.disconnect();
      };
    }
  }, [isAuthenticated, user, toast]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className={`app ${theme}`}>
      <Routes>
        {/* Public routes */}
        <Route
          path="/login"
          element={
            isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />
          }
        />

        {/* Protected routes */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<DashboardPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="search" element={<SearchPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>

        {/* 404 page */}
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </div>
  );
}

export default App;
