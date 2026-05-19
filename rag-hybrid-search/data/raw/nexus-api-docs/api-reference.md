# API Reference

## Streams

### `streams.create(name, config=None)`

Creates a new stream in the current workspace.

**Parameters:**
- `name` (str): Unique stream name within the workspace. Must match `^[a-z][a-z0-9._-]{2,127}$`.
- `config` (StreamConfig, optional): Advanced configuration. Defaults to `StreamConfig(schema="raw")`.

**Returns:** `Stream` object.

**Raises:**
- `StreamAlreadyExistsError` (NXE-1001) — a stream with this name already exists
- `InvalidStreamNameError` (NXE-1002) — name does not match the allowed pattern
- `WorkspaceLimitExceededError` (NXE-1010) — workspace has reached its stream limit

---

### `streams.get(name)`

Retrieve stream metadata.

**Returns:** `Stream` object with fields: `name`, `created_at`, `schema`, `partitions`, `retention`, `event_count`, `last_event_at`.

---

### `streams.delete(name, force=False)`

Delete a stream and all its events.

**Parameters:**
- `force` (bool): Required if the stream has active consumers. Default: `False`.

**Raises:** `StreamHasActiveConsumersError` (NXE-1005) if `force=False` and consumers exist.

---

### `streams.list(prefix=None, limit=100, cursor=None)`

List streams with optional prefix filter. Returns a paginated result.

---

## Events

### `events.publish(stream, data, key=None, headers=None)`

Publish a single event to a stream.

**Parameters:**
- `stream` (str): Target stream name.
- `data` (dict | str | bytes): Event payload.
- `key` (str, optional): Partition key for ordering guarantees.
- `headers` (dict, optional): Arbitrary metadata attached to the event.

**Returns:** `PublishedEvent` with `event_id`, `sequence_number`, `ingested_at`.

---

### `events.publish_batch(stream, events, ordered=True)`

Publish up to 1,000 events atomically. If `ordered=True`, all events are assigned
consecutive sequence numbers.

**Raises:** `BatchTooLargeError` (NXE-2005) if `len(events) > 1000`.

---

### `events.consume(stream, from_sequence=None, from_timestamp=None, batch_size=100)`

Consume events from a stream. Returns a generator.

**Parameters:**
- `from_sequence` (int): Start consuming from this sequence number (inclusive).
- `from_timestamp` (str): Start consuming from events at or after this RFC 3339 timestamp.
- `batch_size` (int): Number of events to fetch per API call (1–1000). Default: 100.

Exactly one of `from_sequence` or `from_timestamp` must be specified.

---

### `events.get(stream, event_id)`

Retrieve a single event by its ID.

**Raises:** `EventNotFoundError` (NXE-2010) if the event does not exist.

---

## Pipelines

### `pipelines.create(name, spec)`

Create a processing pipeline from a `PipelineSpec`.

```python
from nexus.pipelines import PipelineSpec, FilterProcessor, TransformProcessor

spec = PipelineSpec(
    inputs=["raw-events"],
    outputs=["processed-events"],
    processors=[
        FilterProcessor(expression="event.data.status == 'active'"),
        TransformProcessor(script="event.data['score'] = event.data['value'] * 0.01"),
    ],
)
pipeline = client.pipelines.create("scoring-pipeline", spec=spec)
```

**Raises:** `InvalidPipelineSpecError` (NXE-3001) if the spec fails validation.

---

### `pipelines.pause(name)` / `pipelines.resume(name)`

Pause or resume a running pipeline without destroying state.

---

### `pipelines.get_metrics(name, window="1h")`

Returns pipeline throughput, latency p50/p95/p99, and error rate for the given window.
Valid windows: `5m`, `1h`, `24h`, `7d`.
