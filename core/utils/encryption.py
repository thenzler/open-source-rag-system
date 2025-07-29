"""
Encryption utilities for data at rest
Handles document content and sensitive data encryption
"""

import base64
import logging
import os
import secrets
from pathlib import Path
from typing import Optional, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class DocumentEncryption:
    """Handles encryption/decryption of document content"""

    def __init__(self, master_key: str):
        """Initialize with master encryption key"""
        self.master_key = (
            master_key.encode() if isinstance(master_key, str) else master_key
        )
        self._fernet_cache = {}

    def _derive_key(self, tenant_id: int, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derive encryption key for specific tenant"""
        if salt is None:
            salt = secrets.token_bytes(32)

        # Combine master key with tenant ID for tenant-specific encryption
        tenant_context = f"tenant_{tenant_id}".encode()
        combined_key = self.master_key + tenant_context

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(combined_key))
        return key, salt

    def _get_fernet(self, tenant_id: int, salt: bytes) -> Fernet:
        """Get Fernet instance for tenant (with caching)"""
        cache_key = f"{tenant_id}:{base64.b64encode(salt).decode()}"

        if cache_key not in self._fernet_cache:
            key, _ = self._derive_key(tenant_id, salt)
            self._fernet_cache[cache_key] = Fernet(key)

        return self._fernet_cache[cache_key]

    def encrypt_content(self, content: bytes, tenant_id: int) -> Tuple[bytes, bytes]:
        """Encrypt document content for specific tenant"""
        try:
            # Generate tenant-specific key
            key, salt = self._derive_key(tenant_id)
            fernet = Fernet(key)

            # Encrypt content
            encrypted_content = fernet.encrypt(content)

            logger.debug(
                f"Encrypted content for tenant {tenant_id} (size: {len(content)} -> {len(encrypted_content)})"
            )
            return encrypted_content, salt

        except Exception as e:
            logger.error(f"Failed to encrypt content for tenant {tenant_id}: {e}")
            raise

    def decrypt_content(
        self, encrypted_content: bytes, salt: bytes, tenant_id: int
    ) -> bytes:
        """Decrypt document content for specific tenant"""
        try:
            # Get Fernet instance
            fernet = self._get_fernet(tenant_id, salt)

            # Decrypt content
            decrypted_content = fernet.decrypt(encrypted_content)

            logger.debug(
                f"Decrypted content for tenant {tenant_id} (size: {len(encrypted_content)} -> {len(decrypted_content)})"
            )
            return decrypted_content

        except Exception as e:
            logger.error(f"Failed to decrypt content for tenant {tenant_id}: {e}")
            raise

    def encrypt_file(self, file_path: Path, tenant_id: int) -> Tuple[Path, bytes]:
        """Encrypt file on disk and return encrypted file path and salt"""
        try:
            # Read original file
            with open(file_path, "rb") as f:
                content = f.read()

            # Encrypt content
            encrypted_content, salt = self.encrypt_content(content, tenant_id)

            # Write encrypted file
            encrypted_path = file_path.with_suffix(file_path.suffix + ".enc")
            with open(encrypted_path, "wb") as f:
                f.write(encrypted_content)

            # Remove original file
            file_path.unlink()

            logger.info(f"Encrypted file: {file_path} -> {encrypted_path}")
            return encrypted_path, salt

        except Exception as e:
            logger.error(f"Failed to encrypt file {file_path}: {e}")
            raise

    def decrypt_file(
        self,
        encrypted_path: Path,
        salt: bytes,
        tenant_id: int,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Decrypt file and return decrypted file path"""
        try:
            # Read encrypted file
            with open(encrypted_path, "rb") as f:
                encrypted_content = f.read()

            # Decrypt content
            decrypted_content = self.decrypt_content(encrypted_content, salt, tenant_id)

            # Determine output path
            if output_path is None:
                output_path = encrypted_path.with_suffix("")
                if output_path.suffix == ".enc":
                    output_path = output_path.with_suffix("")

            # Write decrypted file
            with open(output_path, "wb") as f:
                f.write(decrypted_content)

            logger.info(f"Decrypted file: {encrypted_path} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to decrypt file {encrypted_path}: {e}")
            raise


class DatabaseEncryption:
    """Handles encryption of sensitive database fields"""

    def __init__(self, encryption_key: str):
        """Initialize with encryption key"""
        key = (
            encryption_key.encode()
            if isinstance(encryption_key, str)
            else encryption_key
        )

        # Derive key for database encryption
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"database_salt_v1",  # Fixed salt for database fields
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key))
        self.fernet = Fernet(derived_key)

    def encrypt_field(self, value: str) -> str:
        """Encrypt sensitive database field"""
        try:
            if not value:
                return value

            encrypted = self.fernet.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()

        except Exception as e:
            logger.error(f"Failed to encrypt field: {e}")
            raise

    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt sensitive database field"""
        try:
            if not encrypted_value:
                return encrypted_value

            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()

        except Exception as e:
            logger.error(f"Failed to decrypt field: {e}")
            raise


class EncryptionManager:
    """Central encryption manager"""

    def __init__(self, master_key: str):
        """Initialize encryption manager"""
        self.document_encryption = DocumentEncryption(master_key)
        self.database_encryption = DatabaseEncryption(master_key)
        self.master_key = master_key

    def generate_master_key(self) -> str:
        """Generate a new master encryption key"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def rotate_keys(self, new_master_key: str):
        """Rotate encryption keys (for key rotation)"""
        logger.warning(
            "Key rotation not yet implemented - requires re-encryption of all data"
        )
        # TODO: Implement key rotation with gradual re-encryption
        pass

    def encrypt_document_content(
        self, content: Union[str, bytes], tenant_id: int
    ) -> Tuple[bytes, bytes]:
        """Encrypt document content"""
        if isinstance(content, str):
            content = content.encode()
        return self.document_encryption.encrypt_content(content, tenant_id)

    def decrypt_document_content(
        self, encrypted_content: bytes, salt: bytes, tenant_id: int
    ) -> bytes:
        """Decrypt document content"""
        return self.document_encryption.decrypt_content(
            encrypted_content, salt, tenant_id
        )

    def encrypt_sensitive_field(self, value: str) -> str:
        """Encrypt sensitive database field"""
        return self.database_encryption.encrypt_field(value)

    def decrypt_sensitive_field(self, encrypted_value: str) -> str:
        """Decrypt sensitive database field"""
        return self.database_encryption.decrypt_field(encrypted_value)


# Global encryption manager
_encryption_manager: Optional[EncryptionManager] = None


def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager"""
    if _encryption_manager is None:
        raise RuntimeError("Encryption manager not initialized")
    return _encryption_manager


def initialize_encryption_manager(master_key: str):
    """Initialize global encryption manager"""
    global _encryption_manager
    _encryption_manager = EncryptionManager(master_key)
    logger.info("Encryption manager initialized")


def is_encryption_enabled() -> bool:
    """Check if encryption is enabled"""
    return _encryption_manager is not None


# Key management utilities
def load_or_generate_master_key(key_file: Path) -> str:
    """Load existing master key or generate new one"""
    try:
        if key_file.exists():
            with open(key_file, "r") as f:
                key = f.read().strip()
            logger.info(f"Loaded master key from {key_file}")
            return key
        else:
            # Generate new key
            manager = EncryptionManager("")  # Temporary instance
            key = manager.generate_master_key()

            # Save key to file
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, "w") as f:
                f.write(key)

            # Secure file permissions
            os.chmod(key_file, 0o600)

            logger.info(f"Generated new master key and saved to {key_file}")
            return key

    except Exception as e:
        logger.error(f"Failed to load/generate master key: {e}")
        raise


def setup_encryption_from_config(config) -> bool:
    """Setup encryption from configuration"""
    try:
        # Check if encryption is enabled
        encryption_enabled = getattr(config, "ENCRYPTION_ENABLED", False)
        if not encryption_enabled:
            logger.info("Encryption is disabled in configuration")
            return False

        # Get key file path
        key_file_path = getattr(config, "ENCRYPTION_KEY_FILE", "data/encryption.key")
        key_file = Path(key_file_path)

        # Load or generate master key
        master_key = load_or_generate_master_key(key_file)

        # Initialize encryption manager
        initialize_encryption_manager(master_key)

        logger.info("Encryption setup completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to setup encryption: {e}")
        return False
