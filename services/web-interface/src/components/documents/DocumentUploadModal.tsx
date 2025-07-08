import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  X,
  FileText,
  AlertCircle,
  CheckCircle,
  Loader2,
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Card, CardContent } from '../ui/card';
import { useUploadDocument } from '../../hooks/useDocuments';
import { formatBytes, getFileIcon } from '../../lib/utils';
import { UploadProgress } from '../../types';

interface DocumentUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function DocumentUploadModal({ open, onOpenChange }: DocumentUploadModalProps) {
  const [files, setFiles] = useState<UploadProgress[]>([]);
  const [category, setCategory] = useState('');
  const [tags, setTags] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  
  const uploadMutation = useUploadDocument();
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadProgress[] = acceptedFiles.map(file => ({
      file,
      progress: 0,
      status: 'uploading',
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/plain': ['.txt'],
      'application/xml': ['.xml'],
      'text/xml': ['.xml'],
    },
    maxFileSize: 100 * 1024 * 1024, // 100MB
    multiple: true,
  });
  
  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };
  
  const handleUpload = async () => {
    if (files.length === 0) return;
    
    setIsUploading(true);
    
    try {
      for (let i = 0; i < files.length; i++) {
        const fileUpload = files[i];
        if (fileUpload.status !== 'uploading') continue;
        
        const metadata = {
          category: category || undefined,
          tags: tags ? tags.split(',').map(t => t.trim()).filter(Boolean) : [],
        };
        
        await uploadMutation.mutateAsync({
          file: fileUpload.file,
          metadata,
          onProgress: (progress) => {
            setFiles(prev => prev.map((f, index) => 
              index === i ? { ...f, progress } : f
            ));
          },
        });
        
        // Mark as completed
        setFiles(prev => prev.map((f, index) => 
          index === i ? { ...f, status: 'completed', progress: 100 } : f
        ));
      }
      
      // Close modal after successful upload
      setTimeout(() => {
        onOpenChange(false);
        setFiles([]);
        setCategory('');
        setTags('');
      }, 1500);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };
  
  const handleClose = () => {
    if (!isUploading) {
      onOpenChange(false);
      setFiles([]);
      setCategory('');
      setTags('');
    }
  };
  
  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Upload className="h-5 w-5" />
            <span>Upload Documents</span>
          </DialogTitle>
          <DialogDescription>
            Upload documents to be processed and added to your knowledge base
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6">
          {/* File Drop Zone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-primary bg-primary/5'
                : 'border-gray-300 dark:border-gray-600 hover:border-primary'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-primary">Drop the files here...</p>
            ) : (
              <div>
                <p className="text-gray-600 dark:text-gray-400 mb-2">
                  Drag & drop files here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supports PDF, Word, Excel, Text, and XML files (max 100MB each)
                </p>
              </div>
            )}
          </div>
          
          {/* Metadata Inputs */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Category (optional)
              </label>
              <Input
                placeholder="e.g. Reports, Manuals, Legal"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                disabled={isUploading}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Tags (optional)
              </label>
              <Input
                placeholder="tag1, tag2, tag3"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                disabled={isUploading}
              />
            </div>
          </div>
          
          {/* File List */}
          {files.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                Files to Upload ({files.length})
              </h3>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {files.map((fileUpload, index) => (
                  <Card key={index}>
                    <CardContent className="p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3 flex-1 min-w-0">
                          <div className={`file-icon ${getFileIcon(fileUpload.file.name)}`}>
                            {fileUpload.file.name.split('.').pop()?.substring(0, 3)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                              {fileUpload.file.name}
                            </p>
                            <p className="text-xs text-gray-500">
                              {formatBytes(fileUpload.file.size)}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {fileUpload.status === 'uploading' && (
                            <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                          )}
                          {fileUpload.status === 'completed' && (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          )}
                          {fileUpload.status === 'error' && (
                            <AlertCircle className="h-4 w-4 text-red-500" />
                          )}
                          
                          {!isUploading && fileUpload.status === 'uploading' && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={() => removeFile(index)}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          )}
                        </div>
                      </div>
                      
                      {fileUpload.progress > 0 && fileUpload.status === 'uploading' && (
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                            <div
                              className="bg-primary h-1.5 rounded-full transition-all duration-300"
                              style={{ width: `${fileUpload.progress}%` }}
                            />
                          </div>
                          <p className="text-xs text-gray-500 mt-1">
                            {fileUpload.progress}% uploaded
                          </p>
                        </div>
                      )}
                      
                      {fileUpload.error && (
                        <p className="text-xs text-red-600 mt-1">
                          {fileUpload.error}
                        </p>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={handleClose}
              disabled={isUploading}
            >
              {isUploading ? 'Uploading...' : 'Cancel'}
            </Button>
            <Button
              onClick={handleUpload}
              disabled={files.length === 0 || isUploading}
            >
              {isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                `Upload ${files.length} file${files.length !== 1 ? 's' : ''}`
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
