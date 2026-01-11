from cryptography.fernet import Fernet
from pathlib import Path
from typing import Tuple


def generate_key(path: Path) -> bytes:
    key = Fernet.generate_key()
    path.write_bytes(key)
    return key


def load_key(path: Path) -> bytes:
    return path.read_bytes()


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(data)


def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    f = Fernet(key)
    return f.decrypt(token)


def encrypt_and_write(data: bytes, out_path: Path, key_path: Path) -> Tuple[bytes, bytes]:
    """Encrypt `data`, write to `out_path` and write key to `key_path`. Returns (cipher, key)."""
    key = generate_key(key_path)
    cipher = encrypt_bytes(data, key)
    out_path.write_bytes(cipher)
    return cipher, key
