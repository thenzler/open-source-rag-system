import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart3,
  TrendingUp,
  FileText,
  Search,
  Clock,
  Database,
  Users,
  Download,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { analyticsApi } from '../lib/api';
import { AnalyticsChart } from '../components/analytics/AnalyticsChart';
import { MetricCard } from '../components/analytics/MetricCard';
import { DateRangePicker } from '../components/analytics/DateRangePicker';
import { LoadingSpinner } from '../components/ui/loading-spinner';
import { formatNumber, formatBytes, formatDuration } from '../lib/utils';

export function AnalyticsPage() {
  const [dateRange, setDateRange] = useState({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0],
  });
  
  // Fetch analytics data
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['analytics', 'stats'],
    queryFn: analyticsApi.getStats,
  });
  
  const { data: queryAnalytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics', 'queries', dateRange],
    queryFn: () => analyticsApi.getQueryAnalytics({
      start_date: dateRange.start,
      end_date: dateRange.end,
      granularity: 'day',
    }),
  });
  
  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }
  
  const metrics = [
    {
      title: 'Total Documents',
      value: formatNumber(stats?.documents.total || 0),
      change: '+12%',
      trend: 'up' as const,
      icon: FileText,
      color: 'blue',
    },
    {
      title: 'Total Queries',
      value: formatNumber(stats?.queries.total_this_week || 0),
      change: '+23%',
      trend: 'up' as const,
      icon: Search,
      color: 'green',
    },
    {
      title: 'Avg Response Time',
      value: formatDuration(stats?.queries.average_response_time_ms || 0),
      change: '-8%',
      trend: 'down' as const,
      icon: Clock,
      color: 'purple',
    },
    {
      title: 'Storage Used',
      value: formatBytes(stats?.storage.documents_size_bytes || 0),
      change: '+5%',
      trend: 'up' as const,
      icon: Database,
      color: 'orange',
    },
  ];
  
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Analytics
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Insights into system performance and usage patterns
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <DateRangePicker
            value={dateRange}
            onChange={setDateRange}
          />
          <Button
            variant="outline"
            className="flex items-center space-x-2"
          >
            <Download className="h-4 w-4" />
            <span>Export</span>
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => (
          <MetricCard
            key={index}
            title={metric.title}
            value={metric.value}
            change={metric.change}
            trend={metric.trend}
            icon={metric.icon}
            color={metric.color}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Query Volume Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Search className="h-5 w-5" />
              <span>Query Volume</span>
            </CardTitle>
            <CardDescription>
              Number of queries over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            {analyticsLoading ? (
              <div className="flex items-center justify-center h-64">
                <LoadingSpinner size="md" />
              </div>
            ) : (
              <AnalyticsChart
                data={queryAnalytics?.metrics.map(m => ({
                  date: m.date,
                  value: m.query_count,
                })) || []}
                type="line"
                color="#3b82f6"
              />
            )}
          </CardContent>
        </Card>

        {/* Response Time Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="h-5 w-5" />
              <span>Response Times</span>
            </CardTitle>
            <CardDescription>
              Average response time in milliseconds
            </CardDescription>
          </CardHeader>
          <CardContent>
            {analyticsLoading ? (
              <div className="flex items-center justify-center h-64">
                <LoadingSpinner size="md" />
              </div>
            ) : (
              <AnalyticsChart
                data={queryAnalytics?.metrics.map(m => ({
                  date: m.date,
                  value: m.average_response_time_ms,
                })) || []}
                type="line"
                color="#8b5cf6"
              />
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Top Queries */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Top Queries</span>
            </CardTitle>
            <CardDescription>
              Most frequently searched terms
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {stats?.queries.most_common_topics?.slice(0, 5).map((topic, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {topic}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${(5 - index) * 20}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500 min-w-[2rem]">
                      {Math.floor(Math.random() * 100) + 1}
                    </span>
                  </div>
                </div>
              )) || (
                <div className="text-center py-8 text-gray-500">
                  No query data available
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Document Categories */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="h-5 w-5" />
              <span>Document Categories</span>
            </CardTitle>
            <CardDescription>
              Distribution by document type
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { name: 'PDF Documents', count: 45, color: 'bg-red-500' },
                { name: 'Word Documents', count: 32, color: 'bg-blue-500' },
                { name: 'Excel Files', count: 18, color: 'bg-green-500' },
                { name: 'Text Files', count: 12, color: 'bg-gray-500' },
                { name: 'XML Files', count: 8, color: 'bg-purple-500' },
              ].map((category, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${category.color}`} />
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {category.name}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500">
                    {category.count}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>System Health</span>
            </CardTitle>
            <CardDescription>
              Current system performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  CPU Usage
                </span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '35%' }} />
                  </div>
                  <span className="text-xs text-gray-500 min-w-[2rem]">35%</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  Memory Usage
                </span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '68%' }} />
                  </div>
                  <span className="text-xs text-gray-500 min-w-[2rem]">68%</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  Disk Usage
                </span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-blue-500 h-2 rounded-full" style={{ width: '42%' }} />
                  </div>
                  <span className="text-xs text-gray-500 min-w-[2rem]">42%</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  Network I/O
                </span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-purple-500 h-2 rounded-full" style={{ width: '22%' }} />
                  </div>
                  <span className="text-xs text-gray-500 min-w-[2rem]">22%</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                <FileText className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(stats?.documents.processed || 0)}
                </p>
                <p className="text-sm text-gray-500">Documents Processed</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-lg">
                <Database className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(stats?.chunks.total || 0)}
                </p>
                <p className="text-sm text-gray-500">Text Chunks</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                <Search className="h-6 w-6 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(stats?.queries.total_today || 0)}
                </p>
                <p className="text-sm text-gray-500">Queries Today</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
