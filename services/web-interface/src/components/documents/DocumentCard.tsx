import React from 'react';
import {
  MoreVertical,
  Download,
  Trash2,
  Edit,
  Eye,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
} from 'lucide-react';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from '../ui/dropdown-menu';
import { Document } from '../../types';
import { formatBytes, formatRelativeTime, getFileIcon } from '../../lib/utils';

interface DocumentCardProps {
  document: Document;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

export function DocumentCard({ document, selected, onSelect, onDelete }: DocumentCardProps) {
  const getStatusIcon = () => {
    switch (document.status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-yellow-500 animate-spin" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-gray-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (document.status) {
      case 'completed':
        return 'text-green-700 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
      case 'processing':
        return 'text-yellow-700 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'failed':
        return 'text-red-700 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
      case 'pending':
        return 'text-gray-700 bg-gray-100 dark:bg-gray-900/20 dark:text-gray-400';
      default:
        return 'text-gray-700 bg-gray-100 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  return (
    <Card className={`hover:shadow-md transition-all cursor-pointer ${
      selected ? 'ring-2 ring-primary' : ''
    }`}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-3 flex-1 min-w-0">
            <input
              type="checkbox"
              checked={selected}
              onChange={onSelect}
              className="rounded border-gray-300"
              onClick={(e) => e.stopPropagation()}
            />
            <div className={`file-icon ${getFileIcon(document.filename, document.mime_type)}`}>
              {document.filename.split('.').pop()?.toLowerCase().substring(0, 3)}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white truncate">
                {document.filename}
              </h3>
              <p className="text-xs text-gray-500 mt-1">
                {formatRelativeTime(document.created_at)}
              </p>
            </div>
          </div>
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={(e) => e.stopPropagation()}
              >
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem>
                <Eye className="h-4 w-4 mr-2" />
                View Details
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Download className="h-4 w-4 mr-2" />
                Download
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Edit className="h-4 w-4 mr-2" />
                Edit Metadata
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
                className="text-red-600 dark:text-red-400"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        
        {/* Status and Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {getStatusIcon()}
              <span className={`text-xs px-2 py-1 rounded-full capitalize ${
                getStatusColor()
              }`}>
                {document.status}
              </span>
            </div>
            <span className="text-xs text-gray-500">
              {formatBytes(document.file_size || 0)}
            </span>
          </div>
          
          {document.status === 'processing' && document.processing_progress > 0 && (
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-primary h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${document.processing_progress}%` }}
              />
            </div>
          )}
          
          {document.processing_message && (
            <p className="text-xs text-gray-500 truncate">
              {document.processing_message}
            </p>
          )}
        </div>
        
        {/* Metadata */}
        <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>
              {document.total_chunks > 0 ? `${document.total_chunks} chunks` : 'Not processed'}
            </span>
            {document.total_pages && (
              <span>{document.total_pages} pages</span>
            )}
          </div>
          
          {document.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {document.tags.slice(0, 3).map((tag, index) => (
                <span
                  key={index}
                  className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 px-2 py-0.5 rounded"
                >
                  {tag}
                </span>
              ))}
              {document.tags.length > 3 && (
                <span className="text-xs text-gray-400">
                  +{document.tags.length - 3} more
                </span>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
