import React, { useState } from 'react';
import { Settings, Database, Cpu, HardDrive } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { useQuery } from '@tanstack/react-query';
import { systemApi } from '../../lib/api';
import { LoadingSpinner } from '../ui/loading-spinner';

export function SystemSettings() {
  const { data: config, isLoading } = useQuery({
    queryKey: ['system', 'config'],
    queryFn: systemApi.getConfig,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="md" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Model Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Cpu className="h-5 w-5" />
            <span>Model Configuration</span>
          </CardTitle>
          <CardDescription>
            Configure AI models and processing parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Embedding Model
              </label>
              <Input
                value={config?.embedding_model || ''}
                disabled
                className="bg-gray-50 dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                LLM Model
              </label>
              <Input
                value={config?.llm_model || ''}
                disabled
                className="bg-gray-50 dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Chunk Size
              </label>
              <Input
                type="number"
                value={config?.chunk_size || 0}
                disabled
                className="bg-gray-50 dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Chunk Overlap
              </label>
              <Input
                type="number"
                value={config?.chunk_overlap || 0}
                disabled
                className="bg-gray-50 dark:bg-gray-800"
              />
            </div>
          </div>
          <p className="text-xs text-gray-500">
            Model configuration can only be changed through environment variables
          </p>
        </CardContent>
      </Card>

      {/* Processing Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="h-5 w-5" />
            <span>Processing Settings</span>
          </CardTitle>
          <CardDescription>
            Document processing and search configuration
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Similarity Threshold
              </label>
              <Input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={config?.similarity_threshold || 0}
                disabled
                className="bg-gray-50 dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Max Query Length
              </label>
              <Input
                type="number"
                value={config?.max_query_length || 0}
                disabled
                className="bg-gray-50 dark:bg-gray-800"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Features */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Flags</CardTitle>
          <CardDescription>
            Enable or disable system features
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {config?.features && Object.entries(config.features).map(([feature, enabled]) => (
              <div key={feature} className="flex items-center justify-between">
                <div>
                  <span className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                    {feature.replace(/_/g, ' ')}
                  </span>
                  <p className="text-xs text-gray-500">System feature configuration</p>
                </div>
                <div className={`px-2 py-1 rounded-full text-xs ${
                  enabled 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                    : 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
                }`}>
                  {enabled ? 'Enabled' : 'Disabled'}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Supported Formats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <HardDrive className="h-5 w-5" />
            <span>Supported File Formats</span>
          </CardTitle>
          <CardDescription>
            File types that can be processed by the system
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {config?.supported_formats?.map((format) => (
              <span
                key={format}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
              >
                {format}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
