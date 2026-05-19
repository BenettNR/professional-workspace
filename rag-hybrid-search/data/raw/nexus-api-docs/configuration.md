# Configuration Reference

## Environment Variables

All configuration keys use the `NEXUS_` prefix.

| Variable                   | Type    | Default         | Description                                    |
|----------------------------|---------|-----------------|------------------------------------------------|
| `NEXUS_API_KEY`            | string  | —               | API key for authentication (required)          |
| `NEXUS_WORKSPACE`          | string  | —               | Workspace slug (required)                      |
| `NEXUS_REGION`             | string  | `us-east-1`     | API endpoint region                            |
| `NEXUS_BASE_URL`           | string  | auto            | Override API base URL (useful for proxies)     |
| `NEXUS_TIMEOUT`            | int     | `30`            | Request timeout in seconds                     |
| `NEXUS_MAX_RETRIES`        | int     | `3`             | Maximum retry attempts on transient errors     |
| `NEXUS_RETRY_BACKOFF`      | float   | `0.5`           | Initial backoff in seconds (doubles each retry)|
| `NEXUS_LOG_LEVEL`          | string  | `WARNING`       | Python logging level for SDK internals         |
| `NEXUS_TLS_VERIFY`         | bool    | `true`          | Verify TLS certificates                        |
| `NEXUS_COMPRESSION`        | string  | `gzip`          | Payload compression: `gzip`, `zstd`, `none`    |
| `NEXUS_BATCH_SIZE`         | int     | `100`           | Default batch size for bulk publish operations |
| `NEXUS_FLUSH_INTERVAL`     | float   | `0.1`           | Auto-flush interval for buffered publishers (s)|
| `NEXUS_CONSUMER_GROUP`     | string  | —               | Consumer group name for cooperative consumption|
| `NEXUS_CONSUMER_ID`        | string  | auto (hostname) | Unique consumer identifier within a group      |

## Configuration File (TOML)

`~/.nexus/config.toml` supports multiple named profiles:

```toml
[default]
api_key = "nxk_live_..."
workspace = "prod-workspace"
region = "us-east-1"
timeout = 30

[staging]
api_key = "nxk_test_..."
workspace = "staging-workspace"
region = "eu-west-1"
timeout = 60
log_level = "DEBUG"
```

Switch profiles:
```bash
NEXUS_PROFILE=staging nexus ping
```

Or in code:
```python
client = NexusClient(profile="staging")
```

## SDK Configuration Object

```python
from nexus import NexusClient, NexusConfig

config = NexusConfig(
    api_key="nxk_live_...",
    workspace="my-workspace",
    region="us-east-1",
    timeout=30,
    max_retries=3,
    compression="gzip",
    batch_size=100,
)
client = NexusClient(config=config)
```

## Stream-Level Configuration

Streams support per-stream retention, schema enforcement, and ordering guarantees:

```python
from nexus.streams import StreamConfig, RetentionPolicy

config = StreamConfig(
    schema="json",                                    # json | avro | protobuf | raw
    retention=RetentionPolicy(days=30, max_bytes=10_000_000_000),
    ordering="per_partition",                         # strict | per_partition | best_effort
    partitions=8,
    compression="zstd",
)
client.streams.create("my-stream", config=config)
```
