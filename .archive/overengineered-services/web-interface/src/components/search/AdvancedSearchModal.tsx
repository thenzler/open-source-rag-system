import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { AdvancedQueryRequest } from '../../types';

interface AdvancedSearchModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSearch: (request: AdvancedQueryRequest) => void;
  initialQuery?: string;
}

export function AdvancedSearchModal({
  open,
  onOpenChange,
  onSearch,
  initialQuery = ''
}: AdvancedSearchModalProps) {
  const [formData, setFormData] = useState<AdvancedQueryRequest>({
    query: initialQuery,
    retrieval_strategy: 'semantic',
    top_k: 10,
    min_score: 0.7,
    rerank: false,
    expand_query: false,
    semantic_threshold: 0.7,
    keyword_boost: 0.3,
    final_k: 5,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.query.trim()) {
      onSearch(formData);
    }
  };

  const updateFormData = (updates: Partial<AdvancedQueryRequest>) => {
    setFormData(prev => ({ ...prev, ...updates }));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Advanced Search</DialogTitle>
          <DialogDescription>
            Configure advanced search parameters for more precise results
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Query Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Search Query
            </label>
            <Input
              value={formData.query}
              onChange={(e) => updateFormData({ query: e.target.value })}
              placeholder="Enter your search query..."
              className="w-full"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Retrieval Strategy */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Retrieval Strategy</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {[
                  { value: 'semantic', label: 'Semantic Search', desc: 'AI-powered meaning-based search' },
                  { value: 'keyword', label: 'Keyword Search', desc: 'Traditional keyword matching' },
                  { value: 'hybrid', label: 'Hybrid Search', desc: 'Combination of both approaches' },
                ].map((option) => (
                  <label key={option.value} className="flex items-start space-x-3">
                    <input
                      type="radio"
                      name="retrieval_strategy"
                      value={option.value}
                      checked={formData.retrieval_strategy === option.value}
                      onChange={(e) => updateFormData({ retrieval_strategy: e.target.value as any })}
                      className="mt-1"
                    />
                    <div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {option.label}
                      </span>
                      <p className="text-xs text-gray-500">{option.desc}</p>
                    </div>
                  </label>
                ))}
              </CardContent>
            </Card>

            {/* Search Parameters */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Search Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Number of Results ({formData.top_k})
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={formData.top_k}
                    onChange={(e) => updateFormData({ top_k: parseInt(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Minimum Relevance Score ({formData.min_score?.toFixed(2)})
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={formData.min_score}
                    onChange={(e) => updateFormData({ min_score: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>

                {formData.retrieval_strategy === 'hybrid' && (
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Keyword Boost ({formData.keyword_boost?.toFixed(2)})
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={formData.keyword_boost}
                      onChange={(e) => updateFormData({ keyword_boost: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Advanced Options */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Advanced Options</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={formData.rerank || false}
                    onChange={(e) => updateFormData({ rerank: e.target.checked })}
                    className="rounded border-gray-300"
                  />
                  <div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      Re-rank Results
                    </span>
                    <p className="text-xs text-gray-500">Use advanced AI re-ranking for better relevance</p>
                  </div>
                </label>

                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={formData.expand_query || false}
                    onChange={(e) => updateFormData({ expand_query: e.target.checked })}
                    className="rounded border-gray-300"
                  />
                  <div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      Expand Query
                    </span>
                    <p className="text-xs text-gray-500">Automatically add related terms</p>
                  </div>
                </label>
              </div>

              {formData.rerank && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Re-ranking Model
                    </label>
                    <select
                      value={formData.rerank_model || 'cross-encoder'}
                      onChange={(e) => updateFormData({ rerank_model: e.target.value as any })}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-sm"
                    >
                      <option value="cross-encoder">Cross-Encoder</option>
                      <option value="colbert">ColBERT</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Final Results ({formData.final_k})
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={formData.final_k}
                      onChange={(e) => updateFormData({ final_k: parseInt(e.target.value) })}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-sm"
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="flex justify-end space-x-3">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!formData.query.trim()}
            >
              Search
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
