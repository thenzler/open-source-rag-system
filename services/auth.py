#!/usr/bin/env python3
"""
JWT Authentication System for RAG API
Provides secure authentication with role-based access control
"""

import os
import jwt
import bcrypt
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import logging
import secrets
import hashlib

logger = logging.getLogger(__name__)

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

@dataclass
class User:
    """User model for authentication"""
    user_id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

@dataclass
class TokenData:
    """JWT token payload data"""
    user_id: str
    username: str
    role: str
    exp: int
    iat: int
    jti: str  # JWT ID for token revocation

class JWTManager:
    """JWT token management with refresh token support"""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.algorithm = algorithm
        self.access_token_expire_minutes = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        
        # In-memory storage for revoked tokens (in production, use Redis)
        self.revoked_tokens = set()
        
        logger.info("JWT Manager initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate a secure random secret key"""
        return secrets.token_urlsafe(32)
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": secrets.token_urlsafe(16),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": secrets.token_urlsafe(16),
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti in self.revoked_tokens:
                logger.warning(f"Revoked token used: {jti}")
                return None
            
            return TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                role=payload["role"],
                exp=payload["exp"],
                iat=payload["iat"],
                jti=jti
            )
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token (add to blacklist)"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
                logger.info(f"Token revoked: {jti}")
                return True
        except jwt.InvalidTokenError:
            pass
        return False
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token"""
        token_data = self.verify_token(refresh_token)
        if not token_data:
            return None
        
        # Verify it's a refresh token
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "refresh":
                return None
        except jwt.InvalidTokenError:
            return None
        
        # Create new access token (would need to fetch user from database)
        # For now, return None as we need user store integration
        return None

class PasswordManager:
    """Secure password handling with bcrypt"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

class SimpleUserStore:
    """Simple in-memory user store (in production, use database)"""
    
    def __init__(self):
        self.users: Dict[str, Dict] = {}
        self.username_to_id: Dict[str, str] = {}
        self.email_to_id: Dict[str, str] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")  # Change in production!
        admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
        
        admin_user = self.create_user(
            username="admin",
            email=admin_email,
            password=admin_password,
            role=UserRole.ADMIN
        )
        
        logger.info(f"Default admin user created: {admin_user.username}")
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> User:
        """Create a new user"""
        # Check if username or email already exists
        if username in self.username_to_id:
            raise ValueError(f"Username '{username}' already exists")
        
        if email in self.email_to_id:
            raise ValueError(f"Email '{email}' already exists")
        
        # Generate user ID
        user_id = hashlib.sha256(f"{username}{email}{time.time()}".encode()).hexdigest()[:16]
        
        # Hash password
        hashed_password = PasswordManager.hash_password(password)
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            created_at=datetime.utcnow()
        )
        
        # Store user data
        self.users[user_id] = {
            "user": user,
            "password_hash": hashed_password
        }
        
        self.username_to_id[username] = user_id
        self.email_to_id[email] = user_id
        
        return user
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        user_data = self.users.get(user_id)
        return user_data["user"] if user_data else None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_id = self.username_to_id.get(username)
        return self.get_user_by_id(user_id) if user_id else None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        user_id = self.username_to_id.get(username)
        if not user_id:
            return None
        
        user_data = self.users.get(user_id)
        if not user_data:
            return None
        
        # Verify password
        if not PasswordManager.verify_password(password, user_data["password_hash"]):
            return None
        
        # Update last login
        user = user_data["user"]
        user.last_login = datetime.utcnow()
        
        return user if user.is_active else None
    
    def update_user_password(self, user_id: str, new_password: str) -> bool:
        """Update user password"""
        user_data = self.users.get(user_id)
        if not user_data:
            return False
        
        user_data["password_hash"] = PasswordManager.hash_password(new_password)
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user"""
        user_data = self.users.get(user_id)
        if not user_data:
            return False
        
        user_data["user"].is_active = False
        return True
    
    def list_users(self) -> List[User]:
        """List all users"""
        return [data["user"] for data in self.users.values()]

class AuthManager:
    """Main authentication manager"""
    
    def __init__(self):
        self.jwt_manager = JWTManager()
        self.user_store = SimpleUserStore()
        logger.info("Auth manager initialized")
    
    def login(self, username: str, password: str) -> Optional[Dict[str, str]]:
        """Authenticate user and return tokens"""
        user = self.user_store.authenticate_user(username, password)
        if not user:
            return None
        
        access_token = self.jwt_manager.create_access_token(user)
        refresh_token = self.jwt_manager.create_refresh_token(user)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": self.jwt_manager.access_token_expire_minutes * 60,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value
            }
        }
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify token and return user"""
        token_data = self.jwt_manager.verify_token(token)
        if not token_data:
            return None
        
        user = self.user_store.get_user_by_id(token_data.user_id)
        return user if user and user.is_active else None
    
    def logout(self, token: str) -> bool:
        """Logout user (revoke token)"""
        return self.jwt_manager.revoke_token(token)
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> User:
        """Create new user"""
        return self.user_store.create_user(username, email, password, role)
    
    def check_permission(self, user: User, required_role: UserRole) -> bool:
        """Check if user has required role"""
        role_hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.USER: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level

# Global auth manager instance
auth_manager = AuthManager()

def get_auth_manager() -> AuthManager:
    """Get global auth manager instance"""
    return auth_manager