import React from 'react';
import {
  FileText,
  ExternalLink,
  Copy,
  ThumbsUp,
  ThumbsDown,
  Star,
  Clock,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { QueryResponse } from '../../types';
import { formatRelativeTime, copyToClipboard, highlightSearchTerms } from '../../lib/utils';
import { useToast } from '../../hooks/useToast';

interface SearchResultsProps {
  result: QueryResponse;
}

export function SearchResults({ result }: SearchResultsProps) {
  const { toast } = useToast();

  const handleCopyResponse = async () => {
    const success = await copyToClipboard(result.response);
    if (success) {
      toast({
        title: 'Copied to clipboard',
        description: 'The response has been copied to your clipboard.',
        variant: 'success',
      });
    }
  };

  const handleCopySource = async (text: string) => {
    const success = await copyToClipboard(text);
    if (success) {
      toast({
        title: 'Copied to clipboard',
        description: 'The source text has been copied to your clipboard.',
        variant: 'success',
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Response */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Response</CardTitle>
            <div className="flex items-center space-x-2">
              {result.confidence_score && (
                <span className="text-sm text-gray-500">
                  Confidence: {Math.round(result.confidence_score * 100)}%
                </span>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopyResponse}
                className="flex items-center space-x-2"
              >
                <Copy className="h-4 w-4" />
                <span>Copy</span>
              </Button>
            </div>
          </div>
          
          {/* Search metadata */}
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>Query: "{result.query}"</span>
            {result.expanded_query && result.expanded_query !== result.query && (
              <span>Expanded: "{result.expanded_query}"</span>
            )}
            {result.retrieval_strategy && (
              <span>Strategy: {result.retrieval_strategy}</span>
            )}
            {result.reranking_applied && (
              <span>Re-ranked</span>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <p className="text-gray-900 dark:text-white leading-relaxed">
              {result.response}
            </p>
          </div>
          
          {/* Response Actions */}
          <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-100 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <Button variant="ghost" size="sm" className="flex items-center space-x-2">
                <ThumbsUp className="h-4 w-4" />
                <span>Helpful</span>
              </Button>
              <Button variant="ghost" size="sm" className="flex items-center space-x-2">
                <ThumbsDown className="h-4 w-4" />
                <span>Not helpful</span>
              </Button>
              <Button variant="ghost" size="sm" className="flex items-center space-x-2">
                <Star className="h-4 w-4" />
                <span>Save</span>
              </Button>
            </div>
            
            <div className="text-xs text-gray-500">
              {result.total_sources} source{result.total_sources !== 1 ? 's' : ''} found
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sources */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Sources ({result.sources.length})
        </h3>
        <div className="space-y-4">
          {result.sources.map((source, index) => (
            <Card key={`${source.document_id}-${source.chunk_id}`} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center justify-center w-8 h-8 bg-primary/10 text-primary rounded-full text-sm font-medium">
                      {index + 1}
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {source.filename}
                      </h4>
                      <div className="flex items-center space-x-3 text-sm text-gray-500 mt-1">
                        <span>Score: {Math.round(source.relevance_score * 100)}%</span>
                        {source.page_number && (
                          <span>Page {source.page_number}</span>
                        )}
                        <span>Chunk {source.chunk_index + 1}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleCopySource(source.text_snippet)}
                      className="flex items-center space-x-2"
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex items-center space-x-2"
                    >
                      <ExternalLink className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                  <p 
                    className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed"
                    dangerouslySetInnerHTML={{
                      __html: highlightSearchTerms(source.text_snippet, result.query)
                    }}
                  />
                </div>
                
                {/* Source metadata */}
                {Object.keys(source.metadata).length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(source.metadata).map(([key, value]) => (
                        <span
                          key={key}
                          className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                        >
                          {key}: {String(value)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Retrieval Metrics */}
      {result.retrieval_metrics && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Search Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {Object.entries(result.retrieval_metrics).map(([key, value]) => (
                <div key={key}>
                  <span className="text-gray-500 block">{key.replace(/_/g, ' ')}</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {typeof value === 'number' ? value.toFixed(3) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
