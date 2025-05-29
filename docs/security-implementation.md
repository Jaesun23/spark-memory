# Security Implementation Guide

Memory One Spark implements comprehensive security features to protect sensitive data and ensure proper access control.

## Overview

The security system consists of four main components:

1. **Encryption**: Data at-rest and in-transit encryption
2. **Access Control**: Role-based access control (RBAC) with API key management
3. **Audit Logging**: Comprehensive logging with anomaly detection
4. **Key Management**: Secure key generation, storage, and rotation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server Layer                      │
│                    (m_security tool)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                   Memory Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Encryption  │  │Access Control│  │ Audit Logger │  │
│  │   Service    │  │   Service    │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Encryption Service

The `EncryptionService` provides multiple encryption methods:

- **Fernet Encryption**: General-purpose symmetric encryption
- **AES-256-GCM**: High-security encryption for sensitive data
- **Field-Level Encryption**: Automatic encryption of sensitive fields

#### Usage Example

```python
# Initialize encryption
encryption = EncryptionService()

# Encrypt data
encrypted = encryption.encrypt("sensitive data")
decrypted = encryption.decrypt(encrypted)

# Field-level encryption
field_enc = FieldLevelEncryption(encryption)
data = {
    "username": "john",
    "password": "secret123",  # Automatically encrypted
    "api_key": "sk-12345"     # Automatically encrypted
}
encrypted_data = field_enc.encrypt_dict(data)
```

### 2. Access Control Service

The `AccessControlService` implements RBAC with:

- **Principals**: Users, services, or API keys
- **Roles**: Predefined permission sets (viewer, user, editor, admin)
- **Permissions**: Fine-grained access rights
- **API Keys**: Secure token generation with rate limiting

#### Predefined Roles

- **VIEWER**: read, search
- **USER**: read, write, search
- **EDITOR**: read, write, delete, search, consolidate
- **ADMIN**: all permissions
- **SYSTEM**: system-level operations

#### Usage Example

```python
# Create principal
principal = access_control.create_principal(
    principal_id="user1",
    roles={Role.USER}
)

# Grant additional permissions
access_control.grant_permission(
    principal_id="user1",
    resource="/projects/secret",
    permissions={Permission.DELETE}
)

# Generate API key
api_key = access_control.generate_api_key(
    principal_id="user1",
    name="dev-key",
    expires_in_days=30,
    rate_limit=100  # 100 requests per minute
)
```

### 3. Audit Logger

The `AuditLogger` provides:

- **Event Logging**: All access and modifications
- **Anomaly Detection**: Automatic detection of suspicious patterns
- **Compliance Reports**: Security and access reports
- **Event Retention**: Configurable retention period

#### Detected Anomaly Patterns

- **Brute Force Attack**: Multiple failed login attempts
- **Privilege Escalation**: Unusual permission changes
- **Data Exfiltration**: Excessive data access
- **Mass Deletion**: Multiple delete operations
- **API Abuse**: Excessive API usage

#### Usage Example

```python
# Log event
audit_logger.log_event(
    event_type=AuditEventType.WRITE,
    principal_id="user1",
    resource="/data/sensitive",
    action="update",
    result="success"
)

# Query events
events = audit_logger.query_events(
    principal_id="user1",
    start_time=start,
    end_time=end
)

# Generate compliance report
report = audit_logger.generate_compliance_report(
    start_time=start,
    end_time=end
)
```

### 4. Key Manager

The `KeyManager` handles:

- **Key Generation**: Secure key creation
- **Key Storage**: Encrypted key storage
- **Key Rotation**: Automatic key rotation
- **Version Management**: Support for multiple key versions

## MCP Tool: m_security

The security features are exposed through the `m_security` MCP tool.

### Actions

#### create_principal
Create a new security principal (user or service).

```python
await m_security("create_principal", [], {
    "id": "user1",
    "type": "user",
    "roles": ["user"],
    "metadata": {"department": "engineering"}
})
```

#### grant
Grant permissions on a resource.

```python
await m_security("grant", ["projects", "lrmm"], {
    "principal_id": "user1",
    "permissions": ["read", "write"]
})
```

#### revoke
Revoke permissions on a resource.

```python
await m_security("revoke", ["projects", "lrmm"], {
    "principal_id": "user1",
    "permissions": ["write"]
})
```

#### api_key
Generate or revoke API keys.

```python
# Generate key
result = await m_security("api_key", [], {
    "principal_id": "user1",
    "name": "dev-key",
    "expires_in_days": 30,
    "rate_limit": 100
})
api_key = result["api_key"]

# Revoke key
await m_security("api_key", [], {
    "operation": "revoke",
    "key": api_key
})
```

#### audit
Query audit logs.

```python
await m_security("audit", [], options={
    "hours": 24,
    "principal_id": "user1",
    "limit": 100
})
```

#### report
Generate security reports.

```python
await m_security("report", [], options={
    "period": "7d"  # Last 7 days
})
```

## Integration with Memory Engine

The Memory Engine automatically integrates security when enabled:

```python
engine = MemoryEngine(
    redis_client=redis,
    enable_security=True,
    encryption_key=None  # Auto-generated if None
)
```

### Automatic Features

1. **Encryption**: Sensitive fields are automatically encrypted
2. **Access Control**: All operations check permissions
3. **Audit Logging**: All operations are logged
4. **Anomaly Detection**: Suspicious patterns trigger alerts

### Manual Control

You can disable encryption for specific operations:

```python
# Save without encryption
await m_memory("save", ["public", "data"], content, {
    "encrypt": False
})
```

## Security Best Practices

1. **Always Enable Security in Production**
   ```python
   engine = MemoryEngine(enable_security=True)
   ```

2. **Use Strong Master Keys**
   ```python
   encryption = EncryptionService.from_password(
       password="strong_password_123",
       salt=os.environ["SALT"]
   )
   ```

3. **Rotate Keys Regularly**
   ```python
   if key_manager.should_rotate("master_key"):
       new_key = key_manager.rotate_key("master_key")
   ```

4. **Monitor Audit Logs**
   ```python
   anomalies = audit_logger.get_anomalies(min_risk_score=0.7)
   if anomalies:
       # Alert security team
   ```

5. **Implement Least Privilege**
   - Start with minimal permissions
   - Grant additional permissions only when needed
   - Regularly review and revoke unnecessary permissions

6. **Secure API Keys**
   - Never commit API keys to version control
   - Use environment variables
   - Set expiration dates
   - Implement rate limiting

## Compliance

The security implementation helps meet various compliance requirements:

- **GDPR**: Personal data encryption and access controls
- **SOC 2**: Audit logging and access management
- **HIPAA**: Encryption at rest and in transit
- **PCI DSS**: Key management and access controls

## Performance Considerations

1. **Encryption Overhead**: ~5-10% for field-level encryption
2. **Access Control**: <1ms per permission check
3. **Audit Logging**: Asynchronous, minimal impact
4. **Key Rotation**: Background process, no downtime

## Troubleshooting

### Common Issues

1. **"Access denied" errors**
   - Check principal permissions
   - Verify resource path
   - Review audit logs

2. **Decryption failures**
   - Ensure correct encryption key
   - Check key version
   - Verify data integrity

3. **API key not working**
   - Check expiration
   - Verify rate limits
   - Ensure principal exists

### Debug Mode

Enable detailed security logging:

```python
import logging
logging.getLogger("src.security").setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Multi-Factor Authentication**: Additional authentication layers
2. **Certificate-Based Auth**: PKI integration
3. **Hardware Security Module**: HSM support for key storage
4. **Zero-Knowledge Proofs**: Privacy-preserving authentication
5. **Homomorphic Encryption**: Compute on encrypted data
