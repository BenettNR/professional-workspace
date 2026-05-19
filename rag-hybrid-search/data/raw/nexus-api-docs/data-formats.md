# Data Formats & Schemas

## Event Payload Formats

Nexus supports four payload formats. The format is configured at stream creation
and cannot be changed without migration.

### JSON (default)

Events must be valid JSON objects. Maximum size: 1 MB per event.

```json
{
  "user_id": "u_123",
  "action": "purchase",
  "amount": 49.99,
  "currency": "USD",
  "item_ids": ["sku_456", "sku_789"]
}
```

### Avro

Requires an Avro schema registered in the Schema Registry:

```python
schema = client.schemas.register(
    name="purchase-event",
    format="avro",
    definition=open("purchase.avsc").read(),
)
client.streams.create("purchases", config=StreamConfig(schema=f"avro:{schema.id}"))
```

### Protocol Buffers (Protobuf)

Register your `.proto` definition:
```python
schema = client.schemas.register(
    name="purchase-event",
    format="proto",
    definition=open("purchase.proto").read(),
)
```

### Raw (binary)

Arbitrary bytes, no schema validation. Useful for pre-encoded payloads.

## Timestamps

All timestamps in the Nexus API use **RFC 3339 format** (a profile of ISO 8601)
with UTC timezone:

```
2025-01-15T10:05:30.123456Z
```

Millisecond precision is supported. Nanosecond timestamps are truncated to microseconds.

## Sequence Numbers

Sequence numbers are 64-bit unsigned integers starting from 1 for the first event
in each stream. They are monotonically increasing but not necessarily contiguous
(gaps may appear due to internal partitioning).

Use `from_sequence=0` when consuming from the very beginning of a stream.

## Event IDs

Event IDs use the `evt_` prefix followed by a base32-encoded ULID:
- `evt_01hx8k3s5v3n4d0e8f6g7h9j2k`
- Sortable by time (lexicographically ordered)
- Globally unique across all workspaces

## Pagination

All list endpoints use cursor-based pagination:

```python
result = client.streams.list(limit=20)
while result.has_more:
    result = client.streams.list(limit=20, cursor=result.next_cursor)
    for stream in result.items:
        process(stream)
```

Cursors are opaque strings; do not attempt to parse or construct them manually.
They are valid for 10 minutes after issue.

## Compression

| Algorithm | Header Value | Notes                               |
|-----------|--------------|-------------------------------------|
| gzip      | `gzip`       | Default; best compatibility         |
| zstd      | `zstd`       | ~30% better ratio; preferred for bulk |
| none      | (omit)       | Uncompressed; useful for debugging   |

Set via `Content-Encoding` request header or `NEXUS_COMPRESSION` environment variable.
