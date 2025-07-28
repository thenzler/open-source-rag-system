import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Users,
  Clock,
  Search,
  FileText,
  Shield,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Zap,
  Eye,
  Download,
  RefreshCw,
  Calendar,
  Filter,
  ArrowUp,
  ArrowDown
} from 'lucide-react';

const AnalyticsDashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('queries');
  const [realTimeData, setRealTimeData] = useState({});
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchAnalytics = async () => {
    try {
      const response = await fetch(`/api/v1/advanced/analytics/queries?time_range=${timeRange}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setAnalytics(data);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRealTimeMetrics = async () => {
    try {
      const response = await fetch('/api/v1/analytics/real-time', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setRealTimeData(data);
    } catch (error) {
      console.error('Failed to fetch real-time metrics:', error);
    }
  };

  useEffect(() => {
    fetchRealTimeMetrics();
    const interval = setInterval(fetchRealTimeMetrics, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <RefreshCw size={32} className="animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center gap-3">
              <BarChart3 className="text-blue-600" size={32} />
              Analytics Dashboard
            </h1>
            <p className="text-gray-600">
              Real-time insights and performance metrics for your RAG system
            </p>
          </div>
          <div className="flex items-center gap-4">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="1d">Last 24 hours</option>
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
            </select>
            <button
              onClick={fetchAnalytics}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
            >
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
        </div>
        <div className="text-sm text-gray-500 mt-2">
          Last updated: {lastUpdate.toLocaleTimeString()}
        </div>
      </div>

      {/* Real-time Status */}
      <div className="mb-8">
        <RealTimeStatus data={realTimeData} />
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Total Queries"
          value={analytics?.metrics?.total_queries || 0}
          change={+15}
          icon={<Search size={24} className="text-blue-600" />}
          color="blue"
        />
        <MetricCard
          title="Success Rate"
          value={`${((analytics?.metrics?.successful_queries / analytics?.metrics?.total_queries) * 100).toFixed(1)}%`}
          change={+2.3}
          icon={<CheckCircle size={24} className="text-green-600" />}
          color="green"
        />
        <MetricCard
          title="Avg Response Time"
          value={`${analytics?.metrics?.average_response_time || 0}s`}
          change={-0.05}
          icon={<Clock size={24} className="text-yellow-600" />}
          color="yellow"
        />
        <MetricCard
          title="Active Users"
          value={analytics?.metrics?.unique_users || 0}
          change={+8}
          icon={<Users size={24} className="text-purple-600" />}
          color="purple"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp size={20} className="text-green-600" />
            Query Performance Trends
          </h3>
          <QueryTrendsChart data={analytics?.performance_trends || []} />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BarChart3 size={20} className="text-blue-600" />
            Confidence Distribution
          </h3>
          <ConfidenceChart data={analytics?.confidence_distribution || {}} />
        </div>
      </div>

      {/* Popular Queries */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Search size={20} className="text-blue-600" />
            Popular Queries
          </h3>
          <PopularQueriesList queries={analytics?.popular_queries || []} />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity size={20} className="text-purple-600" />
            System Health
          </h3>
          <SystemHealthStatus />
        </div>
      </div>

      {/* Security Overview */}
      <div className="bg-white p-6 rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield size={20} className="text-red-600" />
          Security Overview
        </h3>
        <SecurityOverview />
      </div>
    </div>
  );
};

const RealTimeStatus = ({ data }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Activity size={20} className="text-green-600" />
        Real-time Status
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="text-center p-4 bg-green-50 rounded-lg">
          <div className="text-2xl font-bold text-green-600">
            {data.active_queries || 0}
          </div>
          <div className="text-sm text-green-700">Active Queries</div>
        </div>
        <div className="text-center p-4 bg-blue-50 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">
            {data.connected_users || 0}
          </div>
          <div className="text-sm text-blue-700">Connected Users</div>
        </div>
        <div className="text-center p-4 bg-yellow-50 rounded-lg">
          <div className="text-2xl font-bold text-yellow-600">
            {data.cpu_usage || 0}%
          </div>
          <div className="text-sm text-yellow-700">CPU Usage</div>
        </div>
        <div className="text-center p-4 bg-purple-50 rounded-lg">
          <div className="text-2xl font-bold text-purple-600">
            {data.memory_usage || 0}%
          </div>
          <div className="text-sm text-purple-700">Memory Usage</div>
        </div>
      </div>
    </div>
  );
};

const MetricCard = ({ title, value, change, icon, color }) => {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200',
    green: 'bg-green-50 border-green-200',
    yellow: 'bg-yellow-50 border-yellow-200',
    purple: 'bg-purple-50 border-purple-200',
    red: 'bg-red-50 border-red-200'
  };

  return (
    <div className={`p-6 rounded-lg border-2 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-gray-600">{title}</div>
          <div className="text-2xl font-bold text-gray-900 mt-1">{value}</div>
        </div>
        <div className="flex-shrink-0">
          {icon}
        </div>
      </div>
      <div className="mt-4 flex items-center">
        <div className={`flex items-center text-sm ${
          change >= 0 ? 'text-green-600' : 'text-red-600'
        }`}>
          {change >= 0 ? (
            <ArrowUp size={16} className="mr-1" />
          ) : (
            <ArrowDown size={16} className="mr-1" />
          )}
          {Math.abs(change)}% from last period
        </div>
      </div>
    </div>
  );
};

const QueryTrendsChart = ({ data }) => {
  const maxValue = Math.max(...data.map(d => d.query_count));
  const maxTime = Math.max(...data.map(d => d.avg_response_time));

  return (
    <div className="space-y-4">
      {data.map((point, index) => (
        <div key={index} className="flex items-center space-x-4">
          <div className="w-20 text-sm text-gray-600">
            {new Date(point.date).toLocaleDateString()}
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <div className="w-32 text-sm text-gray-600">
                {point.query_count} queries
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${(point.query_count / maxValue) * 100}%` }}
                ></div>
              </div>
            </div>
            <div className="flex items-center space-x-2 mt-1">
              <div className="w-32 text-sm text-gray-600">
                {point.avg_response_time.toFixed(2)}s avg
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full"
                  style={{ width: `${(point.avg_response_time / maxTime) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

const ConfidenceChart = ({ data }) => {
  const total = Object.values(data).reduce((sum, count) => sum + count, 0);

  return (
    <div className="space-y-3">
      {Object.entries(data).map(([range, count]) => (
        <div key={range} className="flex items-center space-x-4">
          <div className="w-20 text-sm text-gray-600">{range}</div>
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <div className="w-16 text-sm text-gray-600">{count}</div>
              <div className="flex-1 bg-gray-200 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-red-500 to-green-500 h-3 rounded-full"
                  style={{ width: `${(count / total) * 100}%` }}
                ></div>
              </div>
              <div className="w-12 text-sm text-gray-600">
                {((count / total) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

const PopularQueriesList = ({ queries }) => {
  return (
    <div className="space-y-3">
      {queries.map((query, index) => (
        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex-1">
            <div className="font-medium text-gray-900">{query.query}</div>
            <div className="text-sm text-gray-600">
              {query.count} queries • {query.avg_confidence.toFixed(2)} avg confidence
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
              #{index + 1}
            </span>
            <Eye size={16} className="text-gray-400 cursor-pointer hover:text-gray-600" />
          </div>
        </div>
      ))}
    </div>
  );
};

const SystemHealthStatus = () => {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch('/api/v1/health');
        const data = await response.json();
        setHealth(data);
      } catch (error) {
        console.error('Failed to fetch health:', error);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  if (!health) {
    return <div className="text-gray-500">Loading health status...</div>;
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle size={16} className="text-green-600" />;
      case 'unhealthy':
        return <XCircle size={16} className="text-red-600" />;
      default:
        return <AlertTriangle size={16} className="text-yellow-600" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-100 text-green-800';
      case 'unhealthy':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-3">
          {getStatusIcon(health.status)}
          <span className="font-medium">Overall System</span>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm ${getStatusColor(health.status)}`}>
          {health.status}
        </span>
      </div>

      {Object.entries(health.services || {}).map(([service, status]) => (
        <div key={service} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            {getStatusIcon(status)}
            <span className="font-medium capitalize">{service.replace('_', ' ')}</span>
          </div>
          <span className={`px-3 py-1 rounded-full text-sm ${getStatusColor(status)}`}>
            {status}
          </span>
        </div>
      ))}
    </div>
  );
};

const SecurityOverview = () => {
  const [security, setSecurity] = useState(null);

  useEffect(() => {
    const fetchSecurity = async () => {
      try {
        const response = await fetch('/api/v1/security/dashboard', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setSecurity(data);
      } catch (error) {
        console.error('Failed to fetch security data:', error);
      }
    };

    fetchSecurity();
    const interval = setInterval(fetchSecurity, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (!security) {
    return <div className="text-gray-500">Loading security overview...</div>;
  }

  const getSecurityLevelColor = (level) => {
    switch (level) {
      case 'low_risk':
        return 'bg-green-100 text-green-800';
      case 'medium_risk':
        return 'bg-yellow-100 text-yellow-800';
      case 'high_risk':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Security Level */}
      <div className="col-span-1">
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-lg font-semibold text-gray-900 mb-2">Security Level</div>
          <div className={`inline-block px-4 py-2 rounded-full text-sm font-medium ${
            getSecurityLevelColor(security.summary?.security_level)
          }`}>
            {security.summary?.security_level?.replace('_', ' ').toUpperCase()}
          </div>
        </div>
      </div>

      {/* Security Stats */}
      <div className="col-span-2">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {security.summary?.active_sessions || 0}
            </div>
            <div className="text-sm text-blue-700">Active Sessions</div>
          </div>
          <div className="text-center p-4 bg-yellow-50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">
              {security.summary?.failed_logins || 0}
            </div>
            <div className="text-sm text-yellow-700">Failed Logins</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">
              {security.summary?.blocked_ips || 0}
            </div>
            <div className="text-sm text-red-700">Blocked IPs</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {security.summary?.recent_events || 0}
            </div>
            <div className="text-sm text-green-700">Recent Events</div>
          </div>
        </div>
      </div>

      {/* Recent Security Events */}
      <div className="col-span-3 mt-6">
        <h4 className="font-medium mb-3">Recent Security Events</h4>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {security.recent_events?.map((event, index) => (
            <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm">
              <div className="flex items-center space-x-2">
                <AlertTriangle size={14} className="text-yellow-600" />
                <span>{event.type}</span>
                <span className="text-gray-500">from {event.ip_address}</span>
              </div>
              <span className="text-gray-500">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;