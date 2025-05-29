"""
Key management module for Memory One Spark.

This module handles secure key generation, storage, rotation, and retrieval.
"""

import base64
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..memory.models import MemoryError


class KeyManagementError(MemoryError):
    """Raised when key management operations fail."""

    pass


class KeyManager:
    """
    Manages encryption keys with rotation and secure storage.

    Features:
    - Master key encryption
    - Key rotation with versioning
    - Secure key storage
    - Key derivation
    """

    def __init__(
        self,
        key_store_path: Optional[Path] = None,
        master_password: Optional[str] = None,
    ):
        """
        Initialize key manager.

        Args:
            key_store_path: Path to store encrypted keys
            master_password: Master password for key encryption
        """
        self.key_store_path = (
            key_store_path or Path.home() / ".memory-one-spark" / "keys"
        )
        self.key_store_path.mkdir(parents=True, exist_ok=True)

        # Initialize or load master key
        self.master_key = self._init_master_key(master_password)
        self.fernet = Fernet(self.master_key)

        # Load existing keys
        self.keys: Dict[str, Dict] = self._load_keys()

    def _init_master_key(self, master_password: Optional[str]) -> bytes:
        """Initialize or derive master key."""
        master_key_file = self.key_store_path / ".master"

        if master_key_file.exists() and master_password:
            # Derive key from password and verify
            with open(master_key_file, "rb") as f:
                data = json.loads(f.read())
                salt = base64.b64decode(data["salt"])

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            return base64.urlsafe_b64encode(kdf.derive(master_password.encode()))

        elif not master_key_file.exists():
            # Generate new master key
            if master_password:
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                master_key = base64.urlsafe_b64encode(
                    kdf.derive(master_password.encode())
                )

                # Save salt for future derivation
                with open(master_key_file, "wb") as f:
                    f.write(
                        json.dumps(
                            {
                                "salt": base64.b64encode(salt).decode(),
                                "created": datetime.utcnow().isoformat(),
                            }
                        ).encode()
                    )
            else:
                master_key = Fernet.generate_key()

            return master_key

        else:
            raise KeyManagementError("Master password required to access existing keys")

    def _load_keys(self) -> Dict[str, Dict]:
        """Load encrypted keys from storage."""
        keys_file = self.key_store_path / "keys.enc"

        if not keys_file.exists():
            return {}

        try:
            with open(keys_file, "rb") as f:
                encrypted_data = f.read()

            decrypted = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted)
        except Exception as e:
            raise KeyManagementError(f"Failed to load keys: {e}")

    def _save_keys(self) -> None:
        """Save encrypted keys to storage."""
        keys_file = self.key_store_path / "keys.enc"

        try:
            data = json.dumps(self.keys, indent=2)
            encrypted = self.fernet.encrypt(data.encode())

            with open(keys_file, "wb") as f:
                f.write(encrypted)
        except Exception as e:
            raise KeyManagementError(f"Failed to save keys: {e}")

    def generate_key(self, key_id: str, key_type: str = "aes256") -> bytes:
        """
        Generate a new encryption key.

        Args:
            key_id: Unique identifier for the key
            key_type: Type of key (aes256, fernet, etc.)

        Returns:
            Generated key
        """
        if key_type == "aes256":
            key = os.urandom(32)  # 256 bits
        elif key_type == "fernet":
            key = Fernet.generate_key()
        else:
            raise KeyManagementError(f"Unsupported key type: {key_type}")

        # Store key metadata
        self.keys[key_id] = {
            "key": base64.b64encode(key).decode(),
            "type": key_type,
            "created": datetime.utcnow().isoformat(),
            "version": 1,
            "active": True,
            "rotations": [],
        }

        self._save_keys()
        return key

    def get_key(self, key_id: str, version: Optional[int] = None) -> bytes:
        """
        Retrieve an encryption key.

        Args:
            key_id: Key identifier
            version: Key version (None for latest)

        Returns:
            Encryption key

        Raises:
            KeyManagementError: If key not found
        """
        if key_id not in self.keys:
            raise KeyManagementError(f"Key not found: {key_id}")

        key_data = self.keys[key_id]

        if version is None:
            # Get active key
            key_b64 = key_data["key"]
        else:
            # Get specific version
            if version == key_data["version"]:
                key_b64 = key_data["key"]
            else:
                # Look in rotations
                for rotation in key_data["rotations"]:
                    if rotation["version"] == version:
                        key_b64 = rotation["key"]
                        break
                else:
                    raise KeyManagementError(
                        f"Key version not found: {key_id} v{version}"
                    )

        return base64.b64decode(key_b64)

    def rotate_key(self, key_id: str) -> bytes:
        """
        Rotate an encryption key.

        Args:
            key_id: Key to rotate

        Returns:
            New key
        """
        if key_id not in self.keys:
            raise KeyManagementError(f"Key not found: {key_id}")

        key_data = self.keys[key_id]

        # Archive current key
        key_data["rotations"].append(
            {
                "key": key_data["key"],
                "version": key_data["version"],
                "rotated": datetime.utcnow().isoformat(),
            }
        )

        # Generate new key
        if key_data["type"] == "aes256":
            new_key = os.urandom(32)
        else:
            new_key = Fernet.generate_key()

        # Update key data
        key_data["key"] = base64.b64encode(new_key).decode()
        key_data["version"] += 1
        key_data["last_rotated"] = datetime.utcnow().isoformat()

        self._save_keys()
        return new_key

    def list_keys(self) -> List[Dict]:
        """List all managed keys."""
        return [
            {
                "id": key_id,
                "type": data["type"],
                "created": data["created"],
                "version": data["version"],
                "active": data["active"],
                "last_rotated": data.get("last_rotated"),
            }
            for key_id, data in self.keys.items()
        ]

    def deactivate_key(self, key_id: str) -> None:
        """Deactivate a key (keeps it for decryption)."""
        if key_id not in self.keys:
            raise KeyManagementError(f"Key not found: {key_id}")

        self.keys[key_id]["active"] = False
        self.keys[key_id]["deactivated"] = datetime.utcnow().isoformat()
        self._save_keys()

    def get_active_keys(self) -> Dict[str, bytes]:
        """Get all active keys."""
        return {
            key_id: self.get_key(key_id)
            for key_id, data in self.keys.items()
            if data["active"]
        }

    def should_rotate(self, key_id: str, max_age_days: int = 90) -> bool:
        """
        Check if a key should be rotated.

        Args:
            key_id: Key to check
            max_age_days: Maximum age before rotation

        Returns:
            True if key should be rotated
        """
        if key_id not in self.keys:
            return False

        key_data = self.keys[key_id]

        # Check last rotation
        last_rotation = key_data.get("last_rotated", key_data["created"])
        rotation_date = datetime.fromisoformat(last_rotation)

        return (datetime.utcnow() - rotation_date) > timedelta(days=max_age_days)
