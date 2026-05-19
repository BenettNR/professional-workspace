# Rate Limiting

## Limits by Tier

Nexus enforces rate limits per API key per minute and per day.

| Tier       | Requests/minute | Publish events/minute | Consume requests/min | Events/day  |
|------------|-----------------|----------------------|----------------------|-------------|
| Developer  | 60              | 500                  | 60                   | 100,000     |
| Standard   | 1,000           | 50,000               | 1,000                | 10,000,000  |
| Enterprise | Custom          | Custom               | Custom               | Unlimited   |

## Rate Limit Headers

Every response includes rate limit metadata:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1736940060
X-RateLimit-Window: 60
```

- `X-RateLimit-Limit` — maximum requests allowed in the window
- `X-RateLimit-Remaining` — requests remaining in the current window
- `X-RateLimit-Reset` — Unix timestamp when the window resets
- `X-RateLimit-Window` — window duration in seconds

## Handling Rate Limit Errors

When the limit is exceeded, the API returns:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 12
Content-Type: application/json

{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 12 seconds.",
    "limit": 1000,
    "reset_at": "2025-01-15T10:01:00Z"
  }
}
```

### Recommended Retry Strategy

Use exponential backoff with jitter:

```python
import time
import random
from nexus.exceptions import RateLimitError

def publish_with_retry(client, stream, data, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.events.publish(stream=stream, data=data)
        except RateLimitError as e:
            wait = e.retry_after + random.uniform(0, 1)
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")
```

## Burst Limits

In addition to per-minute limits, Nexus enforces a **burst limit** to handle
instantaneous spikes:

- Standard tier: 5,000 events per 10-second burst window
- Developer tier: 100 events per 10-second burst window

Bursting beyond these limits triggers `BURST_LIMIT_EXCEEDED` (HTTP 429).

## Increasing Limits

Standard tier: upgrade to Enterprise via the billing dashboard.

Enterprise: contact your account manager or open a support ticket at
support.nexus.io with your workspace ID and expected throughput requirements.
