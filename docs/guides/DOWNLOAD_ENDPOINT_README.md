# Download Endpoint Documentation

## Overview
The download endpoint allows users to download documents that have been uploaded to the RAG system.

## Endpoint Details

### URL
```
GET /api/v1/documents/{document_id}/download
```

### Parameters
- `document_id` (path parameter): The unique ID of the document to download

### Authentication
- If authentication is enabled, only the document uploader or admin users can download documents
- If authentication is disabled, all users can download any document

### Response
- **200 OK**: Returns the file with appropriate headers for download
- **400 Bad Request**: Invalid document ID
- **403 Forbidden**: User doesn't have permission to download this document
- **404 Not Found**: Document not found or file doesn't exist on disk
- **500 Internal Server Error**: Server error

### Response Headers
- `Content-Type`: The MIME type of the file
- `Content-Disposition`: `attachment; filename="original_filename.ext"`
- `Content-Length`: Size of the file in bytes

### Security Features
1. **Path Traversal Protection**: Ensures files can only be accessed from allowed directories
2. **Authentication Check**: Verifies user permissions (when auth is enabled)
3. **Input Validation**: Validates document ID format
4. **File Existence Check**: Ensures file exists before serving

### Supported File Types
- PDF documents (`application/pdf`)
- Word documents (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
- Text files (`text/plain`)
- CSV files (`text/csv`)
- JSON files (`application/json`)
- Other files (`application/octet-stream`)

## Usage Examples

### Using curl
```bash
# Download document with ID 1
curl -O -H "Content-Type: application/json" http://localhost:8000/api/v1/documents/1/download

# With authentication
curl -O -H "Authorization: Bearer your_token" http://localhost:8000/api/v1/documents/1/download
```

### Using Python requests
```python
import requests

# Download document
response = requests.get("http://localhost:8000/api/v1/documents/1/download")
if response.status_code == 200:
    filename = response.headers.get('Content-Disposition').split('filename=')[1].strip('"')
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded: {filename}")
```

### Using JavaScript (fetch)
```javascript
fetch('/api/v1/documents/1/download')
    .then(response => {
        if (response.ok) {
            const filename = response.headers.get('Content-Disposition')
                .split('filename=')[1].replace(/"/g, '');
            return response.blob().then(blob => ({blob, filename}));
        }
        throw new Error('Download failed');
    })
    .then(({blob, filename}) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => console.error('Error:', error));
```

## Testing the Endpoint

Run the test script to verify the download endpoint works:

```bash
python test_download_endpoint.py
```

## File Storage Location
The endpoint searches for files in the following locations:
1. `storage/uploads/` - Original uploaded files
2. `storage/processed/` - Processed files

Files may have numeric prefixes (e.g., `1_document.pdf`) which the endpoint handles automatically.

## Error Handling
All errors are logged and return appropriate HTTP status codes with descriptive error messages. The endpoint includes comprehensive error handling for:
- Missing files
- Invalid document IDs
- Permission issues
- Path traversal attempts
- Server errors

## Integration with Frontend
The download endpoint can be easily integrated into the existing frontend by adding download buttons to the document list that call this endpoint with the appropriate document ID.