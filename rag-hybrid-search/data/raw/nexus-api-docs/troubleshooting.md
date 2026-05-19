# Troubleshooting

## Common Issues

### NXE-4001: Invalid API Key

**Symptoms:** Every request returns HTTP 401.

**Causes and fixes:**
1. Key copied incorrectly — ensure no leading/trailing whitespace.
2. Using a test key (`nxk_test_`) against the live API — test keys are sandboxed.
3. Key was rotated — update your environment variable or config file.

### NXE-4003: Insufficient Scope

**Symptoms:** HTTP 403 on specific endpoints.

**Fix:** The key lacks the required scope. Either use a key with `*` scope or create
a new scoped key that includes the needed scope (e.g. `events:publish`).

### NXE-1002: Invalid Stream Name

**Symptoms:** `streams.create()` raises `InvalidStreamNameError`.

**Rules:**
- Must start with a lowercase letter
- May contain only lowercase letters, digits, `.`, `_`, `-`
- Length: 3–128 characters
- Pattern: `^[a-z][a-z0-9._-]{2,127}$`

**Invalid examples:** `MyStream`, `123-stream`, `a`, `stream name with spaces`

### NXE-2002: Payload Too Large

**Symptoms:** HTTP 413 when publishing events.

**Fix:** Each event is limited to **1 MB**. Split large payloads or move binary
data to object storage (e.g. S3) and reference it by URL in the event.

### NXE-5004: Upstream Timeout

**Symptoms:** Intermittent 504 errors, usually on batch publish.

**Behaviour:** This error is **ambiguous** — the server may or may not have
persisted the events. To handle it safely:

```python
from nexus.exceptions import UpstreamTimeoutError

try:
    result = client.events.publish_batch(stream, events, idempotency_key="batch-xyz")
except UpstreamTimeoutError:
    # Check if events landed before retrying
    last_event = client.events.consume(stream, from_sequence="latest", batch_size=1)
    # Compare sequence numbers to decide whether to retry
```

Use `idempotency_key` on batch publishes to make retries safe.

## Debugging with Request IDs

Every API response includes an `X-Request-ID` header. Include this in support
requests to enable Nexus engineers to trace the exact server-side execution.

```python
import nexus

client = NexusClient(api_key="...", debug=True)
# The SDK automatically logs X-Request-ID for every request
```

## Connection Issues

### SSL Certificate Errors

If you see `SSL: CERTIFICATE_VERIFY_FAILED`, ensure your system CA bundle is
up to date. Do NOT disable TLS verification in production (`NEXUS_TLS_VERIFY=false`).

### Timeout Errors

Default timeout is 30 seconds. For large batch operations, increase it:

```python
client = NexusClient(api_key="...", timeout=120)
```

## Getting Support

- Documentation: docs.nexus.io
- Status page: status.nexus.io
- Support tickets (Standard/Enterprise): support.nexus.io
- Community forum (Developer): community.nexus.io
- Emergency (Enterprise 24/7): +1-800-NEXUS-OPS
