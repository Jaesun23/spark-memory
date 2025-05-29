"""
Data encryption module for Memory One Spark.

This module provides encryption capabilities for data at rest and in transit,
using AES-256 for symmetric encryption and Fernet for simplified encryption tasks.
"""

import base64
import hashlib
import json
import os
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..memory.models import MemoryError


class EncryptionError(MemoryError):
    """Raised when encryption/decryption fails."""

    pass


class EncryptionService:
    """
    Service for encrypting and decrypting data.

    Supports:
    - AES-256 encryption for sensitive data
    - Fernet encryption for general purpose
    - Key derivation from passwords
    - JSON data encryption
    """

    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize encryption service.

        Args:
            master_key: Master encryption key. If None, generates a new one.
        """
        self.master_key = master_key or Fernet.generate_key()
        self.fernet = Fernet(self.master_key)

    @classmethod
    def from_password(
        cls, password: str, salt: Optional[bytes] = None
    ) -> "EncryptionService":
        """
        Create encryption service from password.

        Args:
            password: Password to derive key from
            salt: Salt for key derivation. If None, generates random salt.

        Returns:
            EncryptionService instance
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

        return cls(master_key=key)

    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using Fernet.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data

        Raises:
            EncryptionError: If encryption fails
        """
        try:
            if isinstance(data, str):
                data = data.encode()
            return self.fernet.encrypt(data)
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using Fernet.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data

        Raises:
            EncryptionError: If decryption fails
        """
        try:
            return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")

    def encrypt_json(self, data: Dict[str, Any]) -> bytes:
        """
        Encrypt JSON data.

        Args:
            data: Dictionary to encrypt

        Returns:
            Encrypted JSON data
        """
        json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return self.encrypt(json_str)

    def decrypt_json(self, encrypted_data: bytes) -> Dict[str, Any]:
        """
        Decrypt JSON data.

        Args:
            encrypted_data: Encrypted JSON data

        Returns:
            Decrypted dictionary
        """
        decrypted = self.decrypt(encrypted_data)
        return json.loads(decrypted.decode())

    def encrypt_aes256(
        self, data: bytes, key: Optional[bytes] = None
    ) -> Dict[str, bytes]:
        """
        Encrypt data using AES-256-GCM.

        Args:
            data: Data to encrypt
            key: Encryption key (32 bytes). If None, uses derived key.

        Returns:
            Dictionary with 'ciphertext', 'nonce', and 'tag'
        """
        if key is None:
            key = hashlib.sha256(self.master_key).digest()

        # Generate nonce
        nonce = os.urandom(12)

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
        encryptor = cipher.encryptor()

        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()

        return {"ciphertext": ciphertext, "nonce": nonce, "tag": encryptor.tag}

    def decrypt_aes256(
        self, encrypted_data: Dict[str, bytes], key: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt data using AES-256-GCM.

        Args:
            encrypted_data: Dictionary with 'ciphertext', 'nonce', and 'tag'
            key: Decryption key (32 bytes). If None, uses derived key.

        Returns:
            Decrypted data

        Raises:
            EncryptionError: If decryption fails
        """
        if key is None:
            key = hashlib.sha256(self.master_key).digest()

        try:
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(encrypted_data["nonce"], encrypted_data["tag"]),
            )
            decryptor = cipher.decryptor()

            return decryptor.update(encrypted_data["ciphertext"]) + decryptor.finalize()
        except Exception as e:
            raise EncryptionError(f"AES-256 decryption failed: {e}")

    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return Fernet.generate_key()

    def hash_data(self, data: Union[str, bytes]) -> str:
        """
        Create SHA-256 hash of data.

        Args:
            data: Data to hash

        Returns:
            Hex digest of hash
        """
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()


class FieldLevelEncryption:
    """
    Provides field-level encryption for specific data fields.

    This allows encrypting only sensitive fields while keeping
    other data searchable and queryable.
    """

    def __init__(self, encryption_service: EncryptionService):
        """
        Initialize field-level encryption.

        Args:
            encryption_service: Encryption service to use
        """
        self.encryption_service = encryption_service
        self.encrypted_fields = {
            "password",
            "api_key",
            "secret",
            "token",
            "private_key",
        }

    def encrypt_dict(
        self, data: Dict[str, Any], additional_fields: Optional[set] = None
    ) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in dictionary.

        Args:
            data: Dictionary to process
            additional_fields: Additional fields to encrypt

        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()
        fields_to_encrypt = self.encrypted_fields.copy()

        if additional_fields:
            fields_to_encrypt.update(additional_fields)

        for key, value in data.items():
            if key in fields_to_encrypt and isinstance(value, (str, bytes)):
                encrypted_data[key] = {
                    "_encrypted": True,
                    "value": base64.b64encode(
                        self.encryption_service.encrypt(value)
                    ).decode(),
                }
            elif isinstance(value, dict):
                encrypted_data[key] = self.encrypt_dict(value, additional_fields)
            elif isinstance(value, list):
                encrypted_data[key] = [
                    (
                        self.encrypt_dict(item, additional_fields)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]

        return encrypted_data

    def decrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted fields in dictionary.

        Args:
            data: Dictionary with encrypted fields

        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()

        for key, value in data.items():
            if isinstance(value, dict):
                if value.get("_encrypted") and "value" in value:
                    try:
                        encrypted_bytes = base64.b64decode(value["value"])
                        decrypted = self.encryption_service.decrypt(encrypted_bytes)
                        decrypted_data[key] = decrypted.decode()
                    except Exception:
                        # If decryption fails, keep encrypted value
                        decrypted_data[key] = value
                else:
                    decrypted_data[key] = self.decrypt_dict(value)
            elif isinstance(value, list):
                decrypted_data[key] = [
                    self.decrypt_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]

        return decrypted_data
