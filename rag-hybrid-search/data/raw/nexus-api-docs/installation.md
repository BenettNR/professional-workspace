# Installation & Quickstart

## Requirements

- Python 3.9 or higher
- `pip` 22+ or `uv`

## Installing the SDK

```bash
pip install nexus-sdk
```

For uv users:
```bash
uv add nexus-sdk
```

To install with optional extras:
```bash
# Include async support (aiohttp-based transport)
pip install "nexus-sdk[async]"

# Include the CLI tools
pip install "nexus-sdk[cli]"

# Full install
pip install "nexus-sdk[all]"
```

## Initial Configuration

The SDK reads configuration from the following sources, in priority order:

1. Explicit parameters passed to the client constructor
2. Environment variables (see [Configuration Reference](configuration.md))
3. `~/.nexus/config.toml` (created by `nexus configure`)
4. `/etc/nexus/config.toml` (system-wide defaults)

### Using Environment Variables

```bash
export NEXUS_API_KEY="nxk_live_xxxxxxxxxxxxxxxx"
export NEXUS_WORKSPACE="my-workspace"
export NEXUS_REGION="us-east-1"
```

### Using the Configuration File

Run the interactive setup wizard:
```bash
nexus configure
```

This creates `~/.nexus/config.toml`:
```toml
[default]
api_key = "nxk_live_xxxxxxxxxxxxxxxx"
workspace = "my-workspace"
region = "us-east-1"
```

## Quickstart: Publish Your First Event

```python
from nexus import NexusClient

client = NexusClient(api_key="nxk_live_...")

# Create a stream
client.streams.create("my-first-stream", schema="raw")

# Publish an event
event = client.events.publish(
    stream="my-first-stream",
    data={"user_id": "u_123", "action": "signup", "timestamp": "2025-01-15T10:00:00Z"},
)
print(event.event_id)   # e.g. "evt_01hx..."
print(event.sequence_number)  # e.g. 1

# Consume events
for event in client.events.consume("my-first-stream", from_sequence=0):
    print(event.data)
```

## Verifying Installation

```bash
nexus ping
# → Nexus API v2 reachable. Latency: 42ms. Workspace: my-workspace
```
