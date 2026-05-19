# Authentication

## API Keys

All Nexus API requests must be authenticated using an API key passed in the
`Authorization` header:

```http
Authorization: Bearer nxk_live_xxxxxxxxxxxxxxxx
```

### Key Types

| Type          | Prefix         | Scope                                | Expiry          |
|---------------|----------------|--------------------------------------|-----------------|
| Live key      | `nxk_live_`    | Full API access for production       | Never (revocable) |
| Test key      | `nxk_test_`    | Sandboxed; events not billed         | Never (revocable) |
| Scoped key    | `nxk_scope_`   | Limited to specified streams/actions | Configurable    |
| Webhook key   | `nxk_wh_`      | Webhook signature verification only  | Never           |

### Creating API Keys

Via the dashboard: **Workspace → Settings → API Keys → Generate Key**

Via the API:
```bash
curl -X POST https://api.nexus.io/v2/keys \
  -H "Authorization: Bearer nxk_live_admin_key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-service", "scopes": ["streams:read", "events:publish"]}'
```

Response:
```json
{
  "key_id": "key_01hx...",
  "secret": "nxk_scope_xxxxxxxxxxxxxxxx",
  "scopes": ["streams:read", "events:publish"],
  "created_at": "2025-01-15T10:00:00Z"
}
```

The `secret` is shown only once. Store it securely.

## Scoped Keys

Scoped keys follow the format `resource:action`:

| Scope               | Description                         |
|---------------------|-------------------------------------|
| `streams:read`      | List and describe streams           |
| `streams:write`     | Create and delete streams           |
| `events:publish`    | Write events to streams             |
| `events:consume`    | Read events from streams            |
| `pipelines:read`    | View pipeline configurations        |
| `pipelines:write`   | Create, update, and delete pipelines |
| `keys:manage`       | Create and revoke API keys          |
| `*`                 | Full access (live keys only)        |

## Key Rotation

Nexus supports zero-downtime key rotation:

1. Create a new key with the same scopes
2. Update your service to use the new key
3. Revoke the old key via dashboard or API

## JWT Authentication (Enterprise)

Enterprise workspaces can use short-lived JWTs signed with your workspace's
RSA-2048 private key. Token TTL is configurable from 60 seconds to 24 hours.

```python
from nexus import NexusClient
from nexus.auth import JWTSigner

signer = JWTSigner(private_key_path="/etc/nexus/private.pem", ttl=3600)
client = NexusClient(auth=signer)
```
