#!/usr/bin/env python3
"""
Comprehensive Input Validation Service
Provides robust validation and sanitization for all user inputs
"""

import re
import os
import hashlib
import mimetypes
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import html
import unicodedata
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class SanitizationLevel(Enum):
    """Sanitization levels for different contexts"""
    STRICT = "strict"      # Remove all potentially dangerous content
    MODERATE = "moderate"  # Escape but preserve most content
    MINIMAL = "minimal"    # Basic sanitization only

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    sanitized_value: Any
    errors: List[str]
    warnings: List[str]
    original_value: Any

class FileValidator:
    """Validates uploaded files for security and format compliance"""
    
    def __init__(self):
        # Allowed file extensions and MIME types
        self.allowed_extensions = {
            '.pdf', '.txt', '.doc', '.docx', '.rtf', '.md', '.csv', '.json'
        }
        
        self.allowed_mime_types = {
            'application/pdf',
            'text/plain',
            'text/markdown',
            'text/csv',
            'application/json',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/rtf'
        }
        
        # Maximum file sizes (in bytes)
        self.max_file_sizes = {
            'pdf': 50 * 1024 * 1024,    # 50MB
            'doc': 25 * 1024 * 1024,    # 25MB
            'txt': 10 * 1024 * 1024,    # 10MB
            'default': 20 * 1024 * 1024  # 20MB
        }
        
        # Dangerous file patterns
        self.dangerous_patterns = [
            rb'<script',
            rb'javascript:',
            rb'data:text/html',
            rb'<?php',
            rb'<%',
            rb'exec\(',
            rb'eval\(',
            rb'system\(',
            rb'shell_exec\('
        ]
    
    def validate_file(self, file_path: str, original_filename: str, 
                     content_type: Optional[str] = None) -> ValidationResult:
        """Validate uploaded file for security and format compliance"""
        errors = []
        warnings = []
        
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                errors.append(f"File does not exist: {file_path}")
                return ValidationResult(False, None, errors, warnings, file_path)
            
            if not os.path.isfile(file_path):
                errors.append(f"Path is not a file: {file_path}")
                return ValidationResult(False, None, errors, warnings, file_path)
            
            # Validate filename
            filename_result = self.validate_filename(original_filename)
            if not filename_result.is_valid:
                errors.extend(filename_result.errors)
            
            sanitized_filename = filename_result.sanitized_value
            
            # Check file extension
            file_ext = Path(sanitized_filename).suffix.lower()
            if file_ext not in self.allowed_extensions:
                errors.append(f"File extension '{file_ext}' not allowed. "
                            f"Allowed: {', '.join(sorted(self.allowed_extensions))}")
            
            # Check MIME type
            if content_type:
                if content_type not in self.allowed_mime_types:
                    errors.append(f"MIME type '{content_type}' not allowed. "
                                f"Allowed: {', '.join(sorted(self.allowed_mime_types))}")
            else:
                # Try to detect MIME type
                detected_mime, _ = mimetypes.guess_type(sanitized_filename)
                if detected_mime and detected_mime not in self.allowed_mime_types:
                    warnings.append(f"Detected MIME type '{detected_mime}' may not be supported")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = self.max_file_sizes.get(file_ext.lstrip('.'), self.max_file_sizes['default'])
            
            if file_size > max_size:
                errors.append(f"File size ({file_size:,} bytes) exceeds maximum "
                            f"allowed size ({max_size:,} bytes)")
            
            if file_size == 0:
                errors.append("File is empty")
            
            # Scan file content for dangerous patterns
            try:
                danger_result = self._scan_file_content(file_path)
                if danger_result:
                    errors.append(f"Potentially dangerous content detected: {danger_result}")
            except Exception as e:
                warnings.append(f"Could not scan file content: {str(e)}")
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                errors.append("File is not readable")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                sanitized_value=sanitized_filename,
                errors=errors,
                warnings=warnings,
                original_value=original_filename
            )
        
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, None, errors, warnings, file_path)
    
    def validate_filename(self, filename: str) -> ValidationResult:
        """Validate and sanitize filename"""
        errors = []
        warnings = []
        original_filename = filename
        
        if not filename:
            errors.append("Filename cannot be empty")
            return ValidationResult(False, "", errors, warnings, original_filename)
        
        # Basic sanitization
        sanitized = filename.strip()
        
        # Remove or replace dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        for char in dangerous_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '_')
                warnings.append(f"Replaced dangerous character '{char}' with '_'")
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if unicodedata.category(char)[0] != 'C')
        
        # Limit length
        max_length = 255
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            warnings.append(f"Filename truncated to {max_length} characters")
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        name_without_ext = Path(sanitized).stem.upper()
        if name_without_ext in reserved_names:
            sanitized = f"file_{sanitized}"
            warnings.append(f"Added prefix to avoid reserved name")
        
        # Ensure filename is not empty after sanitization
        if not sanitized.strip():
            errors.append("Filename is empty after sanitization")
            sanitized = "unnamed_file.txt"
        
        # Ensure filename has an extension
        if '.' not in sanitized:
            sanitized += ".txt"
            warnings.append("Added .txt extension")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings,
            original_value=original_filename
        )
    
    def _scan_file_content(self, file_path: str, max_scan_size: int = 1024 * 1024) -> Optional[str]:
        """Scan file content for dangerous patterns"""
        try:
            with open(file_path, 'rb') as f:
                # Read only first part of file for performance
                content = f.read(max_scan_size)
                
                # Convert to lowercase for case-insensitive matching
                content_lower = content.lower()
                
                for pattern in self.dangerous_patterns:
                    if pattern in content_lower:
                        return f"Pattern '{pattern.decode('utf-8', errors='ignore')}' found"
                
                return None
        
        except Exception as e:
            logger.warning(f"Could not scan file content: {str(e)}")
            return None

class TextValidator:
    """Validates and sanitizes text inputs"""
    
    def __init__(self):
        # Common XSS patterns
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'data:text/html',
            r'vbscript:',
            r'<\?php',
            r'<%.*?%>',
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b',
            r'--\s',
            r'/\*.*?\*/',
            r';\s*(drop|delete|insert|update)',
            r'\'\s*(or|and)\s*\'\w*\'\s*=\s*\'\w*\'',
            r'\d+\s*(or|and)\s*\d+',
        ]
    
    def validate_text(self, text: str, max_length: int = 10000, 
                     sanitization_level: SanitizationLevel = SanitizationLevel.MODERATE,
                     allow_html: bool = False) -> ValidationResult:
        """Validate and sanitize text input"""
        errors = []
        warnings = []
        original_text = text
        
        if text is None:
            errors.append("Text cannot be None")
            return ValidationResult(False, "", errors, warnings, original_text)
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            warnings.append("Converted input to string")
        
        # Check length
        if len(text) > max_length:
            errors.append(f"Text length ({len(text)}) exceeds maximum ({max_length})")
        
        # Sanitize based on level
        sanitized = self._sanitize_text(text, sanitization_level, allow_html)
        
        # Check for suspicious patterns
        if not allow_html:
            xss_found = self._check_xss_patterns(sanitized)
            if xss_found:
                errors.append(f"Potential XSS pattern detected: {xss_found}")
        
        sql_found = self._check_sql_patterns(sanitized)
        if sql_found:
            warnings.append(f"Potential SQL injection pattern detected: {sql_found}")
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            warnings.append(f"Text truncated to {max_length} characters")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings,
            original_value=original_text
        )
    
    def validate_query(self, query: str) -> ValidationResult:
        """Validate search query input"""
        return self.validate_text(
            query, 
            max_length=1000, 
            sanitization_level=SanitizationLevel.STRICT,
            allow_html=False
        )
    
    def validate_document_content(self, content: str) -> ValidationResult:
        """Validate document content"""
        return self.validate_text(
            content,
            max_length=1000000,  # 1MB of text
            sanitization_level=SanitizationLevel.MINIMAL,
            allow_html=False
        )
    
    def _sanitize_text(self, text: str, level: SanitizationLevel, allow_html: bool) -> str:
        """Sanitize text based on sanitization level"""
        if level == SanitizationLevel.STRICT:
            # Remove all HTML tags and escape special characters
            text = re.sub(r'<[^>]+>', '', text)
            text = html.escape(text)
            # Remove control characters except newlines and tabs
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        elif level == SanitizationLevel.MODERATE:
            if not allow_html:
                # Escape HTML but preserve structure
                text = html.escape(text)
            else:
                # Remove dangerous tags but keep safe ones
                dangerous_tags = r'<(script|iframe|object|embed|link|meta|style)[^>]*>.*?</\1>'
                text = re.sub(dangerous_tags, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        elif level == SanitizationLevel.MINIMAL:
            # Only remove null bytes and other control characters
            text = text.replace('\0', '')
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
        
        return text.strip()
    
    def _check_xss_patterns(self, text: str) -> Optional[str]:
        """Check for XSS patterns in text"""
        text_lower = text.lower()
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                return pattern
        return None
    
    def _check_sql_patterns(self, text: str) -> Optional[str]:
        """Check for SQL injection patterns in text"""
        text_lower = text.lower()
        for pattern in self.sql_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return pattern
        return None

class NumericValidator:
    """Validates numeric inputs"""
    
    @staticmethod
    def validate_integer(value: Any, min_value: Optional[int] = None, 
                        max_value: Optional[int] = None) -> ValidationResult:
        """Validate integer input"""
        errors = []
        warnings = []
        original_value = value
        
        try:
            # Convert to integer
            if isinstance(value, str):
                value = value.strip()
            
            sanitized = int(value)
            
            # Check range
            if min_value is not None and sanitized < min_value:
                errors.append(f"Value {sanitized} is less than minimum {min_value}")
            
            if max_value is not None and sanitized > max_value:
                errors.append(f"Value {sanitized} is greater than maximum {max_value}")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                sanitized_value=sanitized,
                errors=errors,
                warnings=warnings,
                original_value=original_value
            )
        
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid integer value: {str(e)}")
            return ValidationResult(False, None, errors, warnings, original_value)
    
    @staticmethod
    def validate_float(value: Any, min_value: Optional[float] = None,
                      max_value: Optional[float] = None) -> ValidationResult:
        """Validate float input"""
        errors = []
        warnings = []
        original_value = value
        
        try:
            # Convert to float
            if isinstance(value, str):
                value = value.strip()
            
            sanitized = float(value)
            
            # Check for special values
            if not float('inf') > sanitized > float('-inf'):
                errors.append("Value is infinite")
            
            if sanitized != sanitized:  # NaN check
                errors.append("Value is NaN")
            
            # Check range
            if min_value is not None and sanitized < min_value:
                errors.append(f"Value {sanitized} is less than minimum {min_value}")
            
            if max_value is not None and sanitized > max_value:
                errors.append(f"Value {sanitized} is greater than maximum {max_value}")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                sanitized_value=sanitized,
                errors=errors,
                warnings=warnings,
                original_value=original_value
            )
        
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid float value: {str(e)}")
            return ValidationResult(False, None, errors, warnings, original_value)

class InputValidator:
    """Main input validation service"""
    
    def __init__(self):
        self.file_validator = FileValidator()
        self.text_validator = TextValidator()
        self.numeric_validator = NumericValidator()
        logger.info("Input validator initialized")
    
    def validate_search_request(self, query: str, limit: Optional[int] = None,
                              offset: Optional[int] = None) -> Dict[str, ValidationResult]:
        """Validate search request parameters"""
        results = {}
        
        # Validate query
        results['query'] = self.text_validator.validate_query(query)
        
        # Validate limit
        if limit is not None:
            results['limit'] = self.numeric_validator.validate_integer(
                limit, min_value=1, max_value=1000
            )
        
        # Validate offset
        if offset is not None:
            results['offset'] = self.numeric_validator.validate_integer(
                offset, min_value=0, max_value=1000000
            )
        
        return results
    
    def validate_document_upload(self, filename: str, file_path: str,
                               content_type: Optional[str] = None) -> Dict[str, ValidationResult]:
        """Validate document upload parameters"""
        results = {}
        
        # Validate file
        results['file'] = self.file_validator.validate_file(file_path, filename, content_type)
        
        return results
    
    def validate_user_input(self, username: str, email: str, password: str) -> Dict[str, ValidationResult]:
        """Validate user registration/login input"""
        results = {}
        
        # Validate username
        results['username'] = self._validate_username(username)
        
        # Validate email
        results['email'] = self._validate_email(email)
        
        # Validate password
        results['password'] = self._validate_password(password)
        
        return results
    
    def _validate_username(self, username: str) -> ValidationResult:
        """Validate username"""
        errors = []
        warnings = []
        original_username = username
        
        if not username:
            errors.append("Username cannot be empty")
            return ValidationResult(False, "", errors, warnings, original_username)
        
        # Basic sanitization
        sanitized = username.strip()
        
        # Check length
        if len(sanitized) < 3:
            errors.append("Username must be at least 3 characters")
        
        if len(sanitized) > 50:
            errors.append("Username cannot be longer than 50 characters")
        
        # Check format
        if not re.match(r'^[a-zA-Z0-9_-]+$', sanitized):
            errors.append("Username can only contain letters, numbers, underscores, and hyphens")
        
        # Check for reserved names
        reserved_names = {'admin', 'root', 'system', 'api', 'www', 'mail', 'ftp'}
        if sanitized.lower() in reserved_names:
            errors.append("Username is reserved")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings,
            original_value=original_username
        )
    
    def _validate_email(self, email: str) -> ValidationResult:
        """Validate email address"""
        errors = []
        warnings = []
        original_email = email
        
        if not email:
            errors.append("Email cannot be empty")
            return ValidationResult(False, "", errors, warnings, original_email)
        
        # Basic sanitization
        sanitized = email.strip().lower()
        
        # Check format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, sanitized):
            errors.append("Invalid email format")
        
        # Check length
        if len(sanitized) > 254:
            errors.append("Email address too long")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings,
            original_value=original_email
        )
    
    def _validate_password(self, password: str) -> ValidationResult:
        """Validate password strength"""
        errors = []
        warnings = []
        original_password = password
        
        if not password:
            errors.append("Password cannot be empty")
            return ValidationResult(False, "", errors, warnings, original_password)
        
        # Check length
        if len(password) < 8:
            errors.append("Password must be at least 8 characters")
        
        if len(password) > 128:
            errors.append("Password cannot be longer than 128 characters")
        
        # Check complexity
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        
        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        
        if complexity_score < 3:
            warnings.append("Password should contain uppercase, lowercase, numbers, and special characters")
        
        # Check for common patterns
        common_patterns = ['123456', 'password', 'qwerty', 'abc123', 'admin']
        if any(pattern in password.lower() for pattern in common_patterns):
            warnings.append("Password contains common patterns")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=password,  # Don't sanitize passwords
            errors=errors,
            warnings=warnings,
            original_value=original_password
        )

# Global validator instance
input_validator = InputValidator()

def get_input_validator() -> InputValidator:
    """Get global input validator instance"""
    return input_validator