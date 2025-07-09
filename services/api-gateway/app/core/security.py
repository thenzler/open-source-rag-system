"""
Advanced security features for the RAG system.
Includes authentication, authorization, security monitoring, and threat detection.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import re
import json
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, Gauge
import ipaddress
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

# Security metrics
SECURITY_EVENTS = Counter('rag_security_events_total', 'Total security events', ['event_type', 'severity'])
LOGIN_ATTEMPTS = Counter('rag_login_attempts_total', 'Total login attempts', ['status', 'user_type'])
RATE_LIMIT_VIOLATIONS = Counter('rag_rate_limit_violations_total', 'Rate limit violations', ['endpoint', 'user_id'])
SUSPICIOUS_ACTIVITY = Gauge('rag_suspicious_activity_count', 'Current suspicious activity count')
JWT_TOKENS_ISSUED = Counter('rag_jwt_tokens_issued_total', 'JWT tokens issued', ['token_type'])
SECURITY_SCAN_DURATION = Histogram('rag_security_scan_duration_seconds', 'Security scan duration')

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Security
security = HTTPBearer()


class SecurityEventType(Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_QUERY = "suspicious_query"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    TOKEN_EXPIRED = "token_expired"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class SecuritySeverity(Enum):
    """Security severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: SecurityEventType
    severity: SecuritySeverity
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    endpoint: str
    details: Dict[str, Any]
    timestamp: datetime
    

@dataclass
class UserSession:
    """User session information."""
    user_id: str
    username: str
    role: UserRole
    permissions: List[str]
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    session_id: str


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    jwt_expiry_minutes: int = 60
    refresh_token_expiry_days: int = 7
    rate_limit_requests_per_minute: int = 100
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 5


class SecurityManager:
    """Central security management system."""
    
    def __init__(self):
        self.policy = SecurityPolicy()
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.rate_limit_cache: Dict[str, List[datetime]] = {}
        self.security_events: List[SecurityEvent] = []
        self.ip_whitelist: List[str] = []
        self.ip_blacklist: List[str] = []
        self.suspicious_patterns = self._load_suspicious_patterns()
        
    def _load_suspicious_patterns(self) -> List[str]:
        """Load suspicious query patterns."""
        return [
            r"(?i)(select|insert|update|delete|drop|create|alter|exec|execute)",
            r"(?i)(script|javascript|vbscript|onload|onerror|onclick)",
            r"(?i)(union|order\s+by|group\s+by|having)",
            r"(?i)(eval|exec|system|shell|cmd|powershell)",
            r"(?i)(\.\.\/|\.\.\\|\/etc\/passwd|\/etc\/shadow)",
            r"(?i)(base64|hex|decode|encode|unescape)"
        ]
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[UserSession]:
        """Authenticate user with security checks."""
        try:
            # Check IP blacklist
            if self._is_ip_blacklisted(ip_address):
                await self._log_security_event(
                    SecurityEventType.UNAUTHORIZED_ACCESS,
                    SecuritySeverity.HIGH,
                    None,
                    ip_address,
                    user_agent,
                    "/auth/login",
                    {"reason": "IP blacklisted"}
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied from this IP address"
                )
            
            # Check rate limiting
            if not await self._check_rate_limit(ip_address, "login"):
                await self._log_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    SecuritySeverity.MEDIUM,
                    None,
                    ip_address,
                    user_agent,
                    "/auth/login",
                    {"limit_type": "login"}
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts"
                )
            
            # Check brute force protection
            if self._is_user_locked(username):
                await self._log_security_event(
                    SecurityEventType.BRUTE_FORCE_ATTEMPT,
                    SecuritySeverity.HIGH,
                    username,
                    ip_address,
                    user_agent,
                    "/auth/login",
                    {"reason": "User locked due to failed attempts"}
                )
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail="Account temporarily locked due to too many failed attempts"
                )
            
            # Validate credentials
            user = await self._validate_credentials(username, password)
            
            if not user:
                await self._record_failed_login(username)
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    SecuritySeverity.MEDIUM,
                    username,
                    ip_address,
                    user_agent,
                    "/auth/login",
                    {"reason": "Invalid credentials"}
                )
                LOGIN_ATTEMPTS.labels(status="failed", user_type="standard").inc()
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Create session
            session = await self._create_session(user, ip_address, user_agent)
            
            await self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                SecuritySeverity.LOW,
                user["id"],
                ip_address,
                user_agent,
                "/auth/login",
                {"role": user["role"]}
            )
            
            LOGIN_ATTEMPTS.labels(status="success", user_type="standard").inc()
            
            return session
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service unavailable"
            )
    
    async def validate_token(self, token: str) -> Optional[UserSession]:
        """Validate JWT token and return session."""
        try:
            # Decode JWT
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=["HS256"]
            )
            
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            
            if not user_id or not session_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format"
                )
            
            # Check session
            session = self.active_sessions.get(session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session not found"
                )
            
            # Check session timeout
            if datetime.now() - session.last_activity > timedelta(minutes=self.policy.session_timeout_minutes):
                await self._invalidate_session(session_id)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired"
                )
            
            # Update last activity
            session.last_activity = datetime.now()
            
            return session
            
        except jwt.ExpiredSignatureError:
            await self._log_security_event(
                SecurityEventType.TOKEN_EXPIRED,
                SecuritySeverity.LOW,
                None,
                "unknown",
                "unknown",
                "/auth/validate",
                {"reason": "JWT expired"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def authorize_request(
        self,
        session: UserSession,
        endpoint: str,
        method: str
    ) -> bool:
        """Authorize request based on user permissions."""
        try:
            # Admin can access everything
            if session.role == UserRole.ADMIN:
                return True
            
            # Define permission mappings
            permission_mappings = {
                "GET /api/v1/documents": ["read_documents"],
                "POST /api/v1/documents": ["create_documents"],
                "PUT /api/v1/documents": ["update_documents"],
                "DELETE /api/v1/documents": ["delete_documents"],
                "POST /api/v1/query": ["query_documents"],
                "GET /api/v1/analytics": ["view_analytics"],
                "POST /api/v1/admin": ["admin_access"]
            }
            
            required_permission = permission_mappings.get(f"{method} {endpoint}")
            
            if not required_permission:
                return True  # No specific permission required
            
            # Check if user has required permission
            has_permission = any(perm in session.permissions for perm in required_permission)
            
            if not has_permission:
                await self._log_security_event(
                    SecurityEventType.UNAUTHORIZED_ACCESS,
                    SecuritySeverity.MEDIUM,
                    session.user_id,
                    session.ip_address,
                    session.user_agent,
                    endpoint,
                    {"required_permission": required_permission}
                )
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            return False
    
    async def scan_query_for_threats(
        self,
        query: str,
        user_id: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Scan query for potential security threats."""
        try:
            threats = []
            
            # Check for SQL injection patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, query):
                    threats.append({
                        "type": "sql_injection",
                        "severity": "high",
                        "pattern": pattern,
                        "description": "Potential SQL injection attempt detected"
                    })
            
            # Check for XSS patterns
            xss_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ]
            
            for pattern in xss_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    threats.append({
                        "type": "xss",
                        "severity": "high",
                        "pattern": pattern,
                        "description": "Potential XSS attempt detected"
                    })
            
            # Check query length
            if len(query) > 10000:
                threats.append({
                    "type": "query_length",
                    "severity": "medium",
                    "description": "Unusually long query detected"
                })
            
            # Check for directory traversal
            if "../" in query or "..\\" in query:
                threats.append({
                    "type": "directory_traversal",
                    "severity": "high",
                    "description": "Directory traversal attempt detected"
                })
            
            # Log threats
            if threats:
                await self._log_security_event(
                    SecurityEventType.SUSPICIOUS_QUERY,
                    SecuritySeverity.HIGH,
                    user_id,
                    ip_address,
                    "unknown",
                    "/api/v1/query",
                    {"threats": threats, "query": query[:100]}
                )
                
                SUSPICIOUS_ACTIVITY.set(len(threats))
            
            return {
                "threats_detected": len(threats),
                "threats": threats,
                "allow_query": len([t for t in threats if t["severity"] == "high"]) == 0
            }
            
        except Exception as e:
            logger.error(f"Threat scanning failed: {e}")
            return {
                "threats_detected": 0,
                "threats": [],
                "allow_query": True,
                "error": str(e)
            }
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        ip_address: str
    ) -> bool:
        """Check if request is within rate limits."""
        try:
            # Create rate limit key
            rate_key = f"{user_id}_{endpoint}"
            
            # Check rate limit
            if not await self._check_rate_limit(rate_key, endpoint):
                await self._log_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    SecuritySeverity.MEDIUM,
                    user_id,
                    ip_address,
                    "unknown",
                    endpoint,
                    {"limit_type": "endpoint"}
                )
                
                RATE_LIMIT_VIOLATIONS.labels(
                    endpoint=endpoint,
                    user_id=user_id
                ).inc()
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        try:
            # Recent security events
            recent_events = sorted(
                self.security_events,
                key=lambda x: x.timestamp,
                reverse=True
            )[:50]
            
            # Event statistics
            event_counts = {}
            for event in recent_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Active sessions
            active_sessions_count = len(self.active_sessions)
            
            # Failed login attempts
            failed_attempts_count = sum(
                len(attempts) for attempts in self.failed_login_attempts.values()
            )
            
            # Security recommendations
            recommendations = await self._generate_security_recommendations()
            
            return {
                "summary": {
                    "active_sessions": active_sessions_count,
                    "recent_events": len(recent_events),
                    "failed_logins": failed_attempts_count,
                    "blocked_ips": len(self.ip_blacklist),
                    "security_level": self._calculate_security_level()
                },
                "recent_events": [
                    {
                        "type": event.event_type.value,
                        "severity": event.severity.value,
                        "user_id": event.user_id,
                        "ip_address": event.ip_address,
                        "timestamp": event.timestamp.isoformat(),
                        "details": event.details
                    }
                    for event in recent_events[:10]
                ],
                "event_statistics": event_counts,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to generate security dashboard: {e}")
            return {"error": str(e)}
    
    # Helper methods
    async def _validate_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Validate user credentials."""
        # Mock implementation - would integrate with actual user store
        mock_users = {
            "admin": {
                "id": "admin_user",
                "username": "admin",
                "password_hash": pwd_context.hash("admin123"),
                "role": UserRole.ADMIN,
                "permissions": ["*"]
            },
            "user": {
                "id": "regular_user",
                "username": "user",
                "password_hash": pwd_context.hash("user123"),
                "role": UserRole.USER,
                "permissions": ["read_documents", "query_documents"]
            }
        }
        
        user = mock_users.get(username)
        if user and pwd_context.verify(password, user["password_hash"]):
            return user
        
        return None
    
    async def _create_session(
        self,
        user: Dict[str, Any],
        ip_address: str,
        user_agent: str
    ) -> UserSession:
        """Create user session."""
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            user_id=user["id"],
            username=user["username"],
            role=user["role"],
            permissions=user["permissions"],
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            session_id=session_id
        )
        
        self.active_sessions[session_id] = session
        
        return session
    
    async def _invalidate_session(self, session_id: str):
        """Invalidate user session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def _check_rate_limit(self, key: str, endpoint: str) -> bool:
        """Check rate limit for key."""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        # Clean old requests
        if key in self.rate_limit_cache:
            self.rate_limit_cache[key] = [
                timestamp for timestamp in self.rate_limit_cache[key]
                if timestamp > window_start
            ]
        else:
            self.rate_limit_cache[key] = []
        
        # Check if within limit
        if len(self.rate_limit_cache[key]) >= self.policy.rate_limit_requests_per_minute:
            return False
        
        # Add current request
        self.rate_limit_cache[key].append(now)
        
        return True
    
    def _is_ip_blacklisted(self, ip_address: str) -> bool:
        """Check if IP is blacklisted."""
        return ip_address in self.ip_blacklist
    
    def _is_user_locked(self, username: str) -> bool:
        """Check if user is locked due to failed attempts."""
        if username not in self.failed_login_attempts:
            return False
        
        attempts = self.failed_login_attempts[username]
        recent_attempts = [
            attempt for attempt in attempts
            if datetime.now() - attempt < timedelta(seconds=self.policy.lockout_duration)
        ]
        
        return len(recent_attempts) >= self.policy.max_login_attempts
    
    async def _record_failed_login(self, username: str):
        """Record failed login attempt."""
        if username not in self.failed_login_attempts:
            self.failed_login_attempts[username] = []
        
        self.failed_login_attempts[username].append(datetime.now())
        
        # Clean old attempts
        cutoff = datetime.now() - timedelta(seconds=self.policy.lockout_duration)
        self.failed_login_attempts[username] = [
            attempt for attempt in self.failed_login_attempts[username]
            if attempt > cutoff
        ]
    
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        endpoint: str,
        details: Dict[str, Any]
    ):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            details=details,
            timestamp=datetime.now()
        )
        
        self.security_events.append(event)
        
        # Maintain event buffer size
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        # Update metrics
        SECURITY_EVENTS.labels(
            event_type=event_type.value,
            severity=severity.value
        ).inc()
        
        # Log to application log
        logger.warning(
            f"Security event: {event_type.value} - {severity.value} - "
            f"User: {user_id} - IP: {ip_address} - Details: {details}"
        )
    
    def _calculate_security_level(self) -> str:
        """Calculate current security level."""
        # Simple calculation based on recent events
        recent_events = [
            event for event in self.security_events
            if datetime.now() - event.timestamp < timedelta(hours=24)
        ]
        
        high_severity_count = len([
            event for event in recent_events
            if event.severity == SecuritySeverity.HIGH
        ])
        
        if high_severity_count > 10:
            return "high_risk"
        elif high_severity_count > 5:
            return "medium_risk"
        else:
            return "low_risk"
    
    async def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Check for common issues
        if len(self.failed_login_attempts) > 50:
            recommendations.append("High number of failed login attempts detected. Consider implementing CAPTCHA.")
        
        if len(self.active_sessions) > 100:
            recommendations.append("High number of active sessions. Monitor for unusual activity.")
        
        # Check for suspicious patterns
        recent_threats = [
            event for event in self.security_events
            if event.event_type == SecurityEventType.SUSPICIOUS_QUERY
            and datetime.now() - event.timestamp < timedelta(hours=1)
        ]
        
        if len(recent_threats) > 5:
            recommendations.append("Multiple suspicious queries detected. Review query validation.")
        
        return recommendations


# Global security manager instance
security_manager = SecurityManager()


async def get_current_user_secure(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserSession:
    """Get current user with security validation."""
    token = credentials.credentials
    session = await security_manager.validate_token(token)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return session


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get session from kwargs (injected by dependency)
            session = kwargs.get('current_user')
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if session.role != UserRole.ADMIN and permission not in session.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


async def initialize_security_system():
    """Initialize the security system."""
    logger.info("Security system initialized")


async def cleanup_security_system():
    """Cleanup the security system."""
    logger.info("Security system cleanup completed")
