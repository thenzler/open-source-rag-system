import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { DocumentTextIcon, MagnifyingGlassIcon, ChartBarIcon, ClockIcon } from '@heroicons/react/24/outline';

const Dashboard = () => {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: async () => {
      // Mock data for now
      return {
        total_documents: 156,
        total_queries: 2341,
        processing_queue_size: 3,
        queries_last_hour: 45
      };
    }
  });

  const statCards = [
    {
      name: 'Total Documents',
      value: stats?.total_documents || 0,
      icon: DocumentTextIcon,
      color: 'text-blue-600',
      bg: 'bg-blue-100'
    },
    {
      name: 'Total Queries',
      value: stats?.total_queries || 0,
      icon: MagnifyingGlassIcon,
      color: 'text-green-600',
      bg: 'bg-green-100'
    },
    {
      name: 'Processing Queue',
      value: stats?.processing_queue_size || 0,
      icon: ClockIcon,
      color: 'text-yellow-600',
      bg: 'bg-yellow-100'
    },
    {
      name: 'Queries (1h)',
      value: stats?.queries_last_hour || 0,
      icon: ChartBarIcon,
      color: 'text-purple-600',
      bg: 'bg-purple-100'
    }
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Overview of your RAG system performance and usage.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat) => (
          <div key={stat.name} className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`p-3 rounded-md ${stat.bg}`}>
                    <stat.icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {stat.name}
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {stat.value.toLocaleString()}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-3">
            <div className="flex items-center text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-3"></div>
              <span className="text-gray-600">Document processed: annual-report.pdf</span>
              <span className="ml-auto text-gray-400">2 min ago</span>
            </div>
            <div className="flex items-center text-sm">
              <div className="w-2 h-2 bg-blue-400 rounded-full mr-3"></div>
              <span className="text-gray-600">New query: "market analysis"</span>
              <span className="ml-auto text-gray-400">5 min ago</span>
            </div>
            <div className="flex items-center text-sm">
              <div className="w-2 h-2 bg-yellow-400 rounded-full mr-3"></div>
              <span className="text-gray-600">Document uploaded: strategy.docx</span>
              <span className="ml-auto text-gray-400">8 min ago</span>
            </div>
          </div>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">System Health</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Database</span>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Healthy
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Vector Database</span>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Healthy
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">LLM Service</span>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Healthy
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;