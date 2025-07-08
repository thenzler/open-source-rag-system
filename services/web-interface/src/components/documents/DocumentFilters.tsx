import React from 'react';
import { X } from 'lucide-react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { SearchFilters, ProcessingStatus } from '../../types';

interface DocumentFiltersProps {
  filters: SearchFilters;
  onChange: (filters: SearchFilters) => void;
}

export function DocumentFilters({ filters, onChange }: DocumentFiltersProps) {
  const statusOptions: { value: ProcessingStatus; label: string }[] = [
    { value: 'pending', label: 'Pending' },
    { value: 'processing', label: 'Processing' },
    { value: 'completed', label: 'Completed' },
    { value: 'failed', label: 'Failed' },
  ];
  
  const categoryOptions = [
    'Reports',
    'Manuals',
    'Legal',
    'Research',
    'Presentations',
    'Policies',
    'Training',
  ];
  
  const fileTypeOptions = [
    { value: 'pdf', label: 'PDF' },
    { value: 'docx', label: 'Word' },
    { value: 'xlsx', label: 'Excel' },
    { value: 'txt', label: 'Text' },
    { value: 'xml', label: 'XML' },
  ];
  
  const updateFilter = <K extends keyof SearchFilters>(
    key: K,
    value: SearchFilters[K]
  ) => {
    onChange({ ...filters, [key]: value });
  };
  
  const toggleArrayFilter = <K extends keyof SearchFilters>(
    key: K,
    value: string
  ) => {
    const currentArray = (filters[key] as string[]) || [];
    const newArray = currentArray.includes(value)
      ? currentArray.filter(item => item !== value)
      : [...currentArray, value];
    updateFilter(key, newArray as SearchFilters[K]);
  };
  
  const clearAllFilters = () => {
    onChange({
      status: [],
      category: [],
      tags: [],
      fileTypes: [],
      dateRange: undefined,
      minSize: undefined,
      maxSize: undefined,
    });
  };
  
  const hasActiveFilters = Object.values(filters).some(value => {
    if (Array.isArray(value)) return value.length > 0;
    return value !== undefined && value !== null;
  });
  
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-900 dark:text-white">
          Filters
        </h3>
        {hasActiveFilters && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAllFilters}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            Clear all
          </Button>
        )}
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Status Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
            Status
          </label>
          <div className="space-y-1">
            {statusOptions.map(option => (
              <label key={option.value} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={filters.status?.includes(option.value) || false}
                  onChange={() => toggleArrayFilter('status', option.value)}
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {option.label}
                </span>
              </label>
            ))}
          </div>
        </div>
        
        {/* Category Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
            Category
          </label>
          <div className="space-y-1">
            {categoryOptions.map(category => (
              <label key={category} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={filters.category?.includes(category) || false}
                  onChange={() => toggleArrayFilter('category', category)}
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {category}
                </span>
              </label>
            ))}
          </div>
        </div>
        
        {/* File Type Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
            File Type
          </label>
          <div className="space-y-1">
            {fileTypeOptions.map(type => (
              <label key={type.value} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={filters.fileTypes?.includes(type.value) || false}
                  onChange={() => toggleArrayFilter('fileTypes', type.value)}
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {type.label}
                </span>
              </label>
            ))}
          </div>
        </div>
        
        {/* Date Range Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
            Date Range
          </label>
          <div className="space-y-2">
            <Input
              type="date"
              placeholder="From"
              value={filters.dateRange?.start || ''}
              onChange={(e) => updateFilter('dateRange', {
                ...filters.dateRange,
                start: e.target.value,
              })}
              className="text-xs"
            />
            <Input
              type="date"
              placeholder="To"
              value={filters.dateRange?.end || ''}
              onChange={(e) => updateFilter('dateRange', {
                ...filters.dateRange,
                end: e.target.value,
              })}
              className="text-xs"
            />
          </div>
        </div>
      </div>
      
      {/* Active Filters Display */}
      {hasActiveFilters && (
        <div className="flex flex-wrap gap-2 pt-2 border-t border-gray-200 dark:border-gray-700">
          {filters.status?.map(status => (
            <span
              key={status}
              className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-200"
            >
              Status: {status}
              <button
                onClick={() => toggleArrayFilter('status', status)}
                className="ml-1 hover:text-blue-600"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
          
          {filters.category?.map(category => (
            <span
              key={category}
              className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200"
            >
              Category: {category}
              <button
                onClick={() => toggleArrayFilter('category', category)}
                className="ml-1 hover:text-green-600"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
          
          {filters.fileTypes?.map(type => (
            <span
              key={type}
              className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-purple-100 dark:bg-purple-900/20 text-purple-800 dark:text-purple-200"
            >
              Type: {type.toUpperCase()}
              <button
                onClick={() => toggleArrayFilter('fileTypes', type)}
                className="ml-1 hover:text-purple-600"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
