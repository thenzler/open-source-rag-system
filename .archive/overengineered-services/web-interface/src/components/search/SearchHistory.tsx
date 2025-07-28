import React from 'react';
import { Clock, Search, X } from 'lucide-react';
import { Button } from '../ui/button';
import { formatRelativeTime } from '../../lib/utils';

interface SearchHistoryProps {
  history: string[];
  onHistoryClick: (query: string) => void;
}

export function SearchHistory({ history, onHistoryClick }: SearchHistoryProps) {
  if (history.length === 0) {
    return (
      <div className="text-center py-8">
        <Clock className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
          No search history
        </h3>
        <p className="text-sm text-gray-500">
          Your recent searches will appear here
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {history.map((query, index) => (
        <div
          key={index}
          className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors cursor-pointer"
          onClick={() => onHistoryClick(query)}
        >
          <div className="flex items-center space-x-3">
            <Search className="h-4 w-4 text-gray-400" />
            <span className="text-sm text-gray-900 dark:text-white">
              {query}
            </span>
          </div>
          <Clock className="h-4 w-4 text-gray-400" />
        </div>
      ))}
    </div>
  );
}
