"""
Validation Service
Centralizes input validation, security checks, and response validation
"""
import logging
import re
import mimetypes
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import html

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, response validation will use basic methods")

try:
    from config.config import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)

class ValidationService:
    """Service for input validation and security checks"""
    
    def __init__(self):
        # File validation settings
        self.max_file_size = getattr(config, 'MAX_FILE_SIZE', 50 * 1024 * 1024)  # 50MB
        self.max_filename_length = 255
        self.allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx'}
        self.allowed_mime_types = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'text/markdown',
            'text/csv',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        # Query validation settings
        self.max_query_length = getattr(config, 'MAX_QUERY_LENGTH', 1000)
        self.min_query_length = 3
        
        # Response validation settings
        self.content_similarity_threshold = 0.3
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = None
        
        # Security patterns
        self.xss_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'eval\s*\(',
            r'alert\s*\(',
            r'document\.cookie',
            r'document\.write'
        ]
        
        self.sql_injection_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*\s+set',
            r'--\s*$',
            r'/\*.*\*/',
            r';\s*drop',
            r';\s*delete',
            r';\s*insert'
        ]
        
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'/etc/',
            r'\\windows\\',
            r'/var/',
            r'/usr/',
            r'/root/',
            r'~/'
        ]
        
        # External knowledge patterns for response validation
        self.external_knowledge_patterns = [
            r'\b(heute|morgen|gestern)\b',  # temporal references
            r'\b(wetter|temperatur|regen|schnee)\b',  # weather
            r'\b(präsident|kanzler|minister)\b',  # politics
            r'\b(aktuell|neueste|letzte nachrichten)\b',  # current events
            r'\b(börse|aktien|kurs)\b',  # finance
        ]
    
    def validate_filename(self, filename: str) -> Tuple[bool, str]:
        """Validate uploaded filename"""
        try:
            if not filename or filename.strip() == "":
                return False, "Filename cannot be empty"
            
            # Check length
            if len(filename) > self.max_filename_length:
                return False, f"Filename too long. Maximum {self.max_filename_length} characters"
            
            # Check extension
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.allowed_extensions:
                return False, f"File type {file_ext} not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            
            # Check for invalid characters
            invalid_chars = '<>:"/\\|?*'
            if any(char in filename for char in invalid_chars):
                return False, "Filename contains invalid characters"
            
            # Check for path traversal attempts
            for pattern in self.path_traversal_patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return False, "Filename contains suspicious path elements"
            
            # Check for hidden files
            if filename.startswith('.'):
                return False, "Hidden files not allowed"
            
            return True, "Valid filename"
            
        except Exception as e:
            logger.error(f"Filename validation error: {e}")
            return False, f"Validation failed: {str(e)}"
    
    def validate_file_content(self, content: bytes, content_type: str, filename: str) -> Tuple[bool, str]:
        """Validate file content and type"""
        try:
            # Check size
            if len(content) > self.max_file_size:
                return False, f"File size {len(content)} exceeds maximum {self.max_file_size} bytes"
            
            # Check if empty
            if len(content) == 0:
                return False, "File cannot be empty"
            
            # Validate MIME type
            detected_type = mimetypes.guess_type(filename)[0]
            if detected_type not in self.allowed_mime_types:
                return False, f"MIME type {detected_type} not allowed"
            
            # Basic content validation based on file type
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.pdf':
                # Check PDF header
                if not content.startswith(b'%PDF-'):
                    return False, "Invalid PDF file format"
            
            elif file_ext == '.txt' or file_ext == '.md':
                # Check if it's valid text
                try:
                    content.decode('utf-8')
                except UnicodeDecodeError:
                    return False, "Invalid text file encoding"
            
            elif file_ext == '.docx':
                # Check DOCX header (ZIP file)
                if not content.startswith(b'PK'):
                    return False, "Invalid DOCX file format"
            
            return True, "Valid file content"
            
        except Exception as e:
            logger.error(f"File content validation error: {e}")
            return False, f"Validation failed: {str(e)}"
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate query input"""
        try:
            if not query or query.strip() == "":
                return False, "Query cannot be empty"
            
            # Check length
            if len(query) > self.max_query_length:
                return False, f"Query too long. Maximum {self.max_query_length} characters"
            
            if len(query.strip()) < self.min_query_length:
                return False, f"Query too short. Minimum {self.min_query_length} characters"
            
            # Check for XSS attempts
            for pattern in self.xss_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return False, "Query contains potentially malicious content"
            
            # Check for SQL injection attempts
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return False, "Query contains suspicious SQL patterns"
            
            # Check for path traversal attempts
            for pattern in self.path_traversal_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return False, "Query contains suspicious path elements"
            
            return True, "Valid query"
            
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return False, f"Validation failed: {str(e)}"
    
    def sanitize_string(self, input_string: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input for safe processing"""
        try:
            if not input_string:
                return ""
            
            # HTML escape
            sanitized = html.escape(input_string)
            
            # Remove null bytes
            sanitized = sanitized.replace('\x00', '')
            
            # Normalize whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            # Truncate if needed
            if max_length and len(sanitized) > max_length:
                sanitized = sanitized[:max_length]
            
            return sanitized
            
        except Exception as e:
            logger.error(f"String sanitization error: {e}")
            return ""
    
    def validate_pagination(self, page: int, page_size: int) -> Tuple[bool, str]:
        """Validate pagination parameters"""
        try:
            if page < 1:
                return False, "Page number must be at least 1"
            
            if page_size < 1:
                return False, "Page size must be at least 1"
            
            if page_size > 100:
                return False, "Page size cannot exceed 100"
            
            return True, "Valid pagination"
            
        except Exception as e:
            logger.error(f"Pagination validation error: {e}")
            return False, f"Validation failed: {str(e)}"
    
    def validate_document_id(self, document_id: Any) -> Tuple[bool, str]:
        """Validate document ID parameter"""
        try:
            # Convert to int
            try:
                doc_id = int(document_id)
            except (ValueError, TypeError):
                return False, "Document ID must be a valid integer"
            
            # Check range
            if doc_id < 1:
                return False, "Document ID must be positive"
            
            # Check for reasonable upper limit (prevent DoS)
            if doc_id > 2147483647:  # Max 32-bit int
                return False, "Document ID too large"
            
            return True, "Valid document ID"
            
        except Exception as e:
            logger.error(f"Document ID validation error: {e}")
            return False, f"Validation failed: {str(e)}"
    
    def validate_search_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate and sanitize search parameters"""
        try:
            validated_params = {}
            
            # Validate query
            if 'query' in params:
                is_valid, message = self.validate_query(params['query'])
                if not is_valid:
                    return False, message, {}
                validated_params['query'] = self.sanitize_string(params['query'])
            
            # Validate limit
            if 'limit' in params:
                try:
                    limit = int(params['limit'])
                    if limit < 1 or limit > 100:
                        return False, "Limit must be between 1 and 100", {}
                    validated_params['limit'] = limit
                except (ValueError, TypeError):
                    return False, "Limit must be a valid integer", {}
            
            # Validate threshold
            if 'threshold' in params:
                try:
                    threshold = float(params['threshold'])
                    if threshold < 0.0 or threshold > 1.0:
                        return False, "Threshold must be between 0.0 and 1.0", {}
                    validated_params['threshold'] = threshold
                except (ValueError, TypeError):
                    return False, "Threshold must be a valid float", {}
            
            # Validate boolean flags
            for bool_param in ['use_llm', 'include_metadata']:
                if bool_param in params:
                    if isinstance(params[bool_param], bool):
                        validated_params[bool_param] = params[bool_param]
                    elif isinstance(params[bool_param], str):
                        validated_params[bool_param] = params[bool_param].lower() in ['true', '1', 'yes']
                    else:
                        return False, f"{bool_param} must be a boolean value", {}
            
            return True, "Valid search parameters", validated_params
            
        except Exception as e:
            logger.error(f"Search parameters validation error: {e}")
            return False, f"Validation failed: {str(e)}", {}
    
    def check_rate_limit_compliance(self, operation: str, user_id: str = "anonymous") -> Tuple[bool, str]:
        """Check if operation complies with rate limits"""
        try:
            # This is a placeholder for rate limiting logic
            # In a production system, this would check against:
            # - Redis cache for user request counts
            # - Database for user quotas
            # - Current system load
            
            # For now, always return True
            return True, "Rate limit OK"
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return False, f"Rate limit check failed: {str(e)}"
    
    def generate_security_report(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate security analysis report for request"""
        try:
            report = {
                "timestamp": logger.name,  # Placeholder
                "security_level": "safe",
                "warnings": [],
                "blocked_patterns": [],
                "sanitized_fields": []
            }
            
            # Analyze request data for security issues
            for key, value in request_data.items():
                if isinstance(value, str):
                    # Check for XSS patterns
                    for pattern in self.xss_patterns:
                        if re.search(pattern, value, re.IGNORECASE):
                            report["warnings"].append(f"XSS pattern detected in {key}")
                            report["security_level"] = "warning"
                    
                    # Check for SQL injection patterns
                    for pattern in self.sql_injection_patterns:
                        if re.search(pattern, value, re.IGNORECASE):
                            report["warnings"].append(f"SQL injection pattern detected in {key}")
                            report["security_level"] = "danger"
            
            return report
            
        except Exception as e:
            logger.error(f"Security report generation error: {e}")
            return {"error": str(e)}
    
    def validate_response_content(
        self, 
        llm_response: str, 
        source_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that LLM response content actually exists in source chunks
        
        Args:
            llm_response: The generated LLM response
            source_chunks: List of source document chunks
            
        Returns:
            Dict with validation results
        """
        try:
            if not source_chunks:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "reason": "no_source_chunks",
                    "details": "No source chunks provided for validation"
                }
            
            # Extract the main content (remove source citations)
            clean_response = self._clean_response_for_validation(llm_response)
            
            if not clean_response or len(clean_response.strip()) < 10:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "reason": "insufficient_content",
                    "details": "Response too short for validation"
                }
            
            # Combine all source chunk texts
            source_texts = []
            for chunk in source_chunks:
                content = chunk.get('source_text', '') or chunk.get('content', '')
                if content:
                    source_texts.append(content)
            
            if not source_texts:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "reason": "no_source_content",
                    "details": "No content found in source chunks"
                }
            
            # Calculate content similarity
            if SKLEARN_AVAILABLE and self.vectorizer:
                similarity_scores = self._calculate_content_similarity_tfidf(
                    clean_response, source_texts
                )
            else:
                similarity_scores = self._calculate_content_similarity_basic(
                    clean_response, source_texts
                )
            
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            
            # Validate against threshold
            is_valid = max_similarity >= self.content_similarity_threshold
            
            # Additional validation: Check for external knowledge indicators
            has_external_knowledge = self._check_external_knowledge(clean_response)
            
            if has_external_knowledge:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "reason": "external_knowledge_detected",
                    "details": "Response contains external knowledge not from sources",
                    "content_similarity": max_similarity
                }
            
            return {
                "is_valid": is_valid,
                "confidence": max_similarity,
                "reason": "validation_complete",
                "details": f"Content similarity: {max_similarity:.3f} (threshold: {self.content_similarity_threshold})",
                "content_similarity": max_similarity,
                "chunk_similarities": similarity_scores
            }
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "reason": "validation_error",
                "details": f"Validation error: {str(e)}"
            }
    
    def validate_source_citations(
        self, 
        llm_response: str, 
        source_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that all source citations in response correspond to actual sources
        
        Args:
            llm_response: The generated LLM response
            source_chunks: List of source document chunks
            
        Returns:
            Dict with citation validation results
        """
        try:
            # Extract citations from response
            citation_pattern = r'\[Quelle (\d+)\]'
            citations = re.findall(citation_pattern, llm_response)
            
            if not citations:
                return {
                    "is_valid": True,
                    "reason": "no_citations_found",
                    "details": "Response contains no source citations",
                    "citations_found": [],
                    "valid_citations": []
                }
            
            # Check if citations correspond to actual sources
            valid_citations = []
            invalid_citations = []
            
            available_source_numbers = set(range(1, len(source_chunks) + 1))
            
            for citation in citations:
                citation_num = int(citation)
                if citation_num in available_source_numbers:
                    valid_citations.append(citation_num)
                else:
                    invalid_citations.append(citation_num)
            
            is_valid = len(invalid_citations) == 0
            
            return {
                "is_valid": is_valid,
                "reason": "citation_validation_complete",
                "details": f"Valid citations: {len(valid_citations)}, Invalid: {len(invalid_citations)}",
                "citations_found": [int(c) for c in citations],
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
                "available_sources": len(source_chunks)
            }
            
        except Exception as e:
            logger.error(f"Citation validation failed: {e}")
            return {
                "is_valid": False,
                "reason": "citation_validation_error",
                "details": f"Citation validation error: {str(e)}"
            }
    
    def _clean_response_for_validation(self, response: str) -> str:
        """Remove source citations and formatting for content validation"""
        try:
            # Remove source citations like [Quelle 1], [Quelle X]
            response = re.sub(r'\[Quelle \d+\]', '', response)
            
            # Remove download links
            response = re.sub(r'Download:.*?(/api/v1/documents/\d+/download)', '', response)
            
            # Remove source footer section
            if '📚 VERWENDETE QUELLEN:' in response:
                response = response.split('📚 VERWENDETE QUELLEN:')[0]
            
            # Remove multiple whitespace
            response = re.sub(r'\s+', ' ', response).strip()
            
            return response
            
        except Exception as e:
            logger.warning(f"Error cleaning response: {e}")
            return response
    
    def _calculate_content_similarity_tfidf(
        self, 
        response: str, 
        source_texts: List[str]
    ) -> List[float]:
        """Calculate TF-IDF cosine similarity between response and source texts"""
        try:
            # Prepare texts for comparison
            all_texts = [response] + source_texts
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Get response vector (first row)
            response_vector = tfidf_matrix[0:1]
            
            # Get source vectors (remaining rows)
            source_vectors = tfidf_matrix[1:]
            
            # Calculate similarities
            similarities = cosine_similarity(response_vector, source_vectors)[0]
            
            return [float(sim) for sim in similarities]
            
        except Exception as e:
            logger.warning(f"Error calculating TF-IDF similarity: {e}")
            return [0.0] * len(source_texts)
    
    def _calculate_content_similarity_basic(
        self, 
        response: str, 
        source_texts: List[str]
    ) -> List[float]:
        """Basic content similarity using word overlap (fallback when sklearn unavailable)"""
        try:
            response_words = set(response.lower().split())
            similarities = []
            
            for source_text in source_texts:
                source_words = set(source_text.lower().split())
                
                if not source_words:
                    similarities.append(0.0)
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(response_words.intersection(source_words))
                union = len(response_words.union(source_words))
                
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Error calculating basic similarity: {e}")
            return [0.0] * len(source_texts)
    
    def _check_external_knowledge(self, response: str) -> bool:
        """Check if response contains external knowledge indicators"""
        try:
            response_lower = response.lower()
            
            for pattern in self.external_knowledge_patterns:
                if re.search(pattern, response_lower):
                    logger.info(f"External knowledge pattern detected: {pattern}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking external knowledge: {e}")
            return False