from cryptography.fernet import Fernet
import secrets

# 生成 Fernet 密钥
fernet_key = Fernet.generate_key().decode()
print(f"FERNET_KEY={fernet_key}")

# 生成随机密钥
secret_key = secrets.token_urlsafe(32)
print(f"SECRET_KEY={secret_key}")
