# Webhooks

Webhooks allow Nexus to push notifications to your service when events occur,
rather than your service polling the API.

## Supported Webhook Events

| Event Type                  | Trigger                                                    |
|-----------------------------|------------------------------------------------------------|
| `stream.created`            | A new stream is created                                    |
| `stream.deleted`            | A stream is deleted                                        |
| `pipeline.started`          | A pipeline begins processing                               |
| `pipeline.paused`           | A pipeline is paused                                       |
| `pipeline.error`            | A pipeline encounters a processing error                   |
| `events.threshold_reached`  | A stream crosses a configured event count threshold        |
| `billing.limit_warning`     | Daily event usage reaches 80% of tier limit                |
| `key.expiring`              | A scoped API key is within 7 days of expiry                |

## Creating a Webhook

```bash
curl -X POST https://api.nexus.io/v2/webhooks \
  -H "Authorization: Bearer nxk_live_..." \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-service.example.com/nexus-webhook",
    "events": ["pipeline.error", "billing.limit_warning"],
    "secret": "your-signing-secret"
  }'
```

## Webhook Payload

```json
{
  "webhook_id": "wh_01hx...",
  "event_type": "pipeline.error",
  "workspace": "my-workspace",
  "timestamp": "2025-01-15T10:05:00Z",
  "data": {
    "pipeline_name": "scoring-pipeline",
    "error_code": "NXE-3001",
    "message": "Filter expression syntax error at position 14"
  }
}
```

## Signature Verification

Every webhook request includes an `X-Nexus-Signature` header:

```
X-Nexus-Signature: sha256=a1b2c3d4...
```

Verification example (Python):

```python
import hmac
import hashlib

def verify_webhook(payload_bytes: bytes, signature_header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)
```

Always use `hmac.compare_digest` to prevent timing attacks.

## Retry Policy

If your endpoint returns a non-2xx response or times out (30-second limit),
Nexus retries the delivery:

- Attempt 1: immediate
- Attempt 2: 1 minute
- Attempt 3: 5 minutes
- Attempt 4: 30 minutes
- Attempt 5: 2 hours

After 5 failed attempts, the event is marked as **undelivered** and visible in
the webhook delivery log in your dashboard. You can manually retry undelivered events.

## Webhook Keys

Webhook secrets are separate from API keys. Rotate them via:

```bash
curl -X POST https://api.nexus.io/v2/webhooks/{webhook_id}/rotate-secret \
  -H "Authorization: Bearer nxk_live_..."
```

The old secret remains valid for 60 seconds after rotation to allow in-flight
requests to complete.
