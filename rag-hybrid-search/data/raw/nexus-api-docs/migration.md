# Migration Guide: v1 → v2

API v1 reaches end-of-life on **2026-03-01**. This guide covers all breaking
changes between v1 and v2 and provides migration examples.

## Breaking Changes

### 1. Authentication Header

**v1:**
```http
X-Nexus-API-Key: nxk_live_...
```

**v2:**
```http
Authorization: Bearer nxk_live_...
```

Update all HTTP clients and SDK initialisation code.

### 2. Stream Name Validation

v2 enforces stricter stream name validation: names must now start with a lowercase
letter (v1 allowed digits as the first character). Existing streams with non-conforming
names will continue to work but cannot be created anew.

### 3. Event ID Format

v1 event IDs were plain UUIDs (e.g. `550e8400-e29b-41d4-a716-446655440000`).  
v2 uses ULID-based IDs with the `evt_` prefix (e.g. `evt_01hx8k3s5v3n4d0e8f6g7h9j2k`).

v1 IDs remain queryable via `events.get(stream, event_id)` for 12 months after migration.

### 4. Consume API

**v1:** `events.read(stream, offset=0)`  
**v2:** `events.consume(stream, from_sequence=0)`

The `offset` parameter is renamed to `from_sequence`. Additionally, v2 adds
`from_timestamp` as an alternative starting point.

### 5. Error Code Format

v1 used string error types (e.g. `"stream_not_found"`).  
v2 uses numeric codes with the `NXE-XXXX` prefix (e.g. `NXE-1003`).

Update any code that inspects error types:

```python
# v1
except nexus.StreamNotFoundError:
    ...

# v2 (exception class name unchanged, but error.code is now "NXE-1003")
except nexus.StreamNotFoundError as e:
    print(e.code)  # "NXE-1003"
```

### 6. Pagination

v1 used offset-based pagination (`page`, `per_page`).  
v2 uses cursor-based pagination (`cursor`, `limit`). Offset pagination is not supported in v2.

### 7. Pipeline API

v1 pipelines used a YAML-based spec format.  
v2 uses a structured Python/JSON `PipelineSpec` object. YAML specs can be migrated using:

```bash
nexus migrate-pipeline --spec pipeline.yaml --output pipeline.json
```

## Migration Steps

1. Update the SDK: `pip install --upgrade nexus-sdk>=2.0`
2. Replace `X-Nexus-API-Key` headers with `Authorization: Bearer`
3. Replace `events.read()` calls with `events.consume()`
4. Update pagination code to cursor-based pattern
5. Update error handling to use `NXE-XXXX` codes
6. Test in staging with the v2 endpoint
7. Flip the base URL to v2 in production

## Backwards Compatibility

The v1 API remains fully functional until **2026-03-01**. After that date,
v1 requests will return `NXE-5010` with message: "API v1 has reached end-of-life."
