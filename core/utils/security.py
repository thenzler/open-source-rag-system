"""
Security utilities for the RAG system
Includes ID obfuscation and randomization
"""

import base64
import hashlib
import logging
import secrets
from typing import Optional

logger = logging.getLogger(__name__)


class IDObfuscator:
    """Secure ID obfuscation for database IDs"""

    def __init__(self, secret_key: str):
        """Initialize with secret key"""
        self.secret_key = (
            secret_key.encode() if isinstance(secret_key, str) else secret_key
        )

    def encode_id(self, real_id: int, entity_type: str = "doc") -> str:
        """Encode a real database ID into an obfuscated ID"""
        try:
            # Create a deterministic salt based on entity type and secret
            salt = hashlib.sha256(
                f"{entity_type}:{self.secret_key.decode()}".encode()
            ).digest()[:8]

            # XOR the ID with a hash of the salt for basic obfuscation
            id_bytes = real_id.to_bytes(8, byteorder="big")
            hash_key = hashlib.sha256(salt + self.secret_key).digest()[:8]

            obfuscated = bytes(a ^ b for a, b in zip(id_bytes, hash_key))

            # Add the salt and encode as base64 URL-safe
            combined = salt + obfuscated
            encoded = base64.urlsafe_b64encode(combined).decode().rstrip("=")

            return f"{entity_type}_{encoded}"

        except Exception as e:
            logger.error(f"Failed to encode ID {real_id}: {e}")
            # Fallback to simple encoding
            return f"{entity_type}_{real_id:08x}"

    def decode_id(self, obfuscated_id: str) -> Optional[int]:
        """Decode an obfuscated ID back to the real database ID"""
        try:
            if "_" not in obfuscated_id:
                return None

            entity_type, encoded_part = obfuscated_id.split("_", 1)

            # Handle simple fallback encoding
            if len(encoded_part) <= 8 and all(
                c in "0123456789abcdef" for c in encoded_part
            ):
                return int(encoded_part, 16)

            # Decode base64
            # Add padding if needed
            missing_padding = len(encoded_part) % 4
            if missing_padding:
                encoded_part += "=" * (4 - missing_padding)

            combined = base64.urlsafe_b64decode(encoded_part.encode())

            if len(combined) < 16:  # salt (8) + obfuscated_id (8)
                return None

            salt = combined[:8]
            obfuscated = combined[8:16]

            # Recreate the hash key
            hash_key = hashlib.sha256(salt + self.secret_key).digest()[:8]

            # XOR back to get original
            id_bytes = bytes(a ^ b for a, b in zip(obfuscated, hash_key))
            real_id = int.from_bytes(id_bytes, byteorder="big")

            return real_id

        except Exception as e:
            logger.error(f"Failed to decode ID {obfuscated_id}: {e}")
            return None

    def is_valid_id(self, obfuscated_id: str) -> bool:
        """Check if an obfuscated ID is valid"""
        return self.decode_id(obfuscated_id) is not None


class RandomIDGenerator:
    """Generate random, non-sequential IDs for documents"""

    @staticmethod
    def generate_document_id() -> str:
        """Generate a random document ID"""
        # Use a combination of timestamp and random data
        import time

        timestamp = int(time.time() * 1000)  # milliseconds
        random_part = secrets.randbelow(999999)  # 6 digits

        # Combine them in a way that's not easily guessable
        combined = (timestamp << 20) + random_part

        # Convert to base36 for shorter strings
        return base36_encode(combined)

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_api_key() -> str:
        """Generate an API key"""
        prefix = "rag_"
        key_part = secrets.token_urlsafe(32)
        return f"{prefix}{key_part}"


def base36_encode(number: int) -> str:
    """Encode number in base36"""
    if number == 0:
        return "0"

    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""

    while number:
        number, remainder = divmod(number, 36)
        result = alphabet[remainder] + result

    return result


def base36_decode(encoded: str) -> int:
    """Decode base36 string to number"""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = 0

    for char in encoded.lower():
        if char not in alphabet:
            raise ValueError(f"Invalid character in base36 string: {char}")
        result = result * 36 + alphabet.index(char)

    return result


def generate_secure_filename(
    original_filename: str, user_id: Optional[int] = None
) -> str:
    """Generate a secure filename that doesn't reveal the original"""
    # Extract extension
    parts = original_filename.rsplit(".", 1)
    extension = parts[1] if len(parts) > 1 else ""

    # Generate random part
    random_part = secrets.token_urlsafe(16)

    # Add user context if provided
    if user_id:
        user_hash = hashlib.sha256(f"user_{user_id}".encode()).hexdigest()[:8]
        random_part = f"{user_hash}_{random_part}"

    # Combine with extension
    if extension:
        return f"{random_part}.{extension}"
    else:
        return random_part


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, bytes]:
    """Hash password securely with salt"""
    if salt is None:
        salt = secrets.token_bytes(32)

    # Use PBKDF2 for password hashing
    from hashlib import pbkdf2_hmac

    hashed = pbkdf2_hmac("sha256", password.encode(), salt, 100000)

    # Return base64 encoded hash and salt
    hash_b64 = base64.b64encode(hashed).decode()
    return hash_b64, salt


def verify_password(password: str, hash_b64: str, salt: bytes) -> bool:
    """Verify password against hash"""
    try:
        stored_hash = base64.b64decode(hash_b64.encode())
        from hashlib import pbkdf2_hmac

        computed_hash = pbkdf2_hmac("sha256", password.encode(), salt, 100000)

        # Constant time comparison
        import hmac

        return hmac.compare_digest(stored_hash, computed_hash)
    except Exception:
        return False


def sanitize_slug(text: str) -> str:
    """Convert text to URL-safe slug"""
    import re

    # Convert to lowercase
    text = text.lower()

    # Replace spaces and special chars with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)

    # Remove leading/trailing hyphens
    text = text.strip("-")

    # Limit length
    if len(text) > 50:
        text = text[:50].rstrip("-")

    return text


# Global obfuscator instance (used by convenience functions)
_id_obfuscator: Optional[IDObfuscator] = None


def get_id_obfuscator() -> IDObfuscator:
    """Get global ID obfuscator"""
    if _id_obfuscator is None:
        raise RuntimeError("ID obfuscator not initialized")
    return _id_obfuscator


def initialize_id_obfuscator(secret_key: str):
    """Initialize global ID obfuscator"""
    global _id_obfuscator
    _id_obfuscator = IDObfuscator(secret_key)


# Convenience functions
def encode_document_id(doc_id: int) -> str:
    """Encode document ID"""
    return get_id_obfuscator().encode_id(doc_id, "doc")


def decode_document_id(obfuscated_id: str) -> Optional[int]:
    """Decode document ID"""
    return get_id_obfuscator().decode_id(obfuscated_id)


def encode_tenant_id(tenant_id: int) -> str:
    """Encode tenant ID"""
    return get_id_obfuscator().encode_id(tenant_id, "tenant")


def decode_tenant_id(obfuscated_id: str) -> Optional[int]:
    """Decode tenant ID"""
    return get_id_obfuscator().decode_id(obfuscated_id)
