# Error Codes Reference

All Nexus API errors follow this structure:

```json
{
  "error": {
    "code": "NXE-XXXX",
    "message": "Human-readable description",
    "details": {}
  }
}
```

## Stream Errors (NXE-1xxx)

| Code      | HTTP | Name                          | Description                                              |
|-----------|------|-------------------------------|----------------------------------------------------------|
| NXE-1001  | 409  | StreamAlreadyExistsError      | A stream with this name already exists in the workspace  |
| NXE-1002  | 422  | InvalidStreamNameError        | Stream name does not match `^[a-z][a-z0-9._-]{2,127}$`  |
| NXE-1003  | 404  | StreamNotFoundError           | The specified stream does not exist                      |
| NXE-1004  | 423  | StreamLockedError             | Stream is locked during a schema migration               |
| NXE-1005  | 409  | StreamHasActiveConsumersError | Cannot delete stream; force=True required                |
| NXE-1006  | 422  | InvalidSchemaError            | Schema validation failed for the submitted payload       |
| NXE-1010  | 402  | WorkspaceLimitExceededError   | Workspace has reached its stream quota                   |

## Event Errors (NXE-2xxx)

| Code      | HTTP | Name                     | Description                                               |
|-----------|------|--------------------------|-----------------------------------------------------------|
| NXE-2001  | 422  | InvalidEventPayloadError | Payload is not valid JSON (or doesn't match stream schema)|
| NXE-2002  | 413  | PayloadTooLargeError     | Single event exceeds 1 MB size limit                     |
| NXE-2003  | 422  | InvalidSequenceError     | Requested sequence number is out of range                 |
| NXE-2004  | 422  | InvalidTimestampError    | Timestamp is not valid RFC 3339 or is in the future       |
| NXE-2005  | 422  | BatchTooLargeError       | Batch contains more than 1,000 events                     |
| NXE-2010  | 404  | EventNotFoundError       | Event with the given ID does not exist                    |
| NXE-2011  | 410  | EventExpiredError        | Event has been deleted per retention policy               |

## Pipeline Errors (NXE-3xxx)

| Code      | HTTP | Name                      | Description                                             |
|-----------|------|---------------------------|---------------------------------------------------------|
| NXE-3001  | 422  | InvalidPipelineSpecError  | Pipeline DAG failed validation                          |
| NXE-3002  | 404  | PipelineNotFoundError     | No pipeline with this name in the workspace             |
| NXE-3003  | 409  | PipelineConflictError     | A pipeline with this name already exists                |
| NXE-3004  | 409  | PipelineRunningError      | Cannot delete a running pipeline; pause first           |

## Authentication Errors (NXE-4xxx)

| Code      | HTTP | Name                    | Description                                              |
|-----------|------|-------------------------|----------------------------------------------------------|
| NXE-4001  | 401  | InvalidKeyError         | API key is malformed or not recognised                   |
| NXE-4002  | 401  | RevokedKeyError         | API key has been revoked                                 |
| NXE-4003  | 403  | InsufficientScopeError  | Key does not have the required scope for this action     |
| NXE-4004  | 403  | WorkspaceMismatchError  | Key belongs to a different workspace                     |

## System Errors (NXE-5xxx)

| Code      | HTTP | Name                  | Description                                                |
|-----------|------|-----------------------|------------------------------------------------------------|
| NXE-5001  | 429  | RateLimitExceeded     | Request rate limit exceeded (see Retry-After header)       |
| NXE-5002  | 429  | BurstLimitExceeded    | Burst limit exceeded; back off and retry                   |
| NXE-5003  | 503  | ServiceUnavailable    | Nexus is temporarily unavailable; retry with backoff       |
| NXE-5004  | 504  | UpstreamTimeout       | Internal timeout; request may or may not have succeeded    |
| NXE-5010  | 500  | InternalServerError   | Unexpected server error; include Request-ID when reporting |
