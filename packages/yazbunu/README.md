# Yazbunu

Structured JSONL logging with a built-in web viewer. Zero dependencies for logging, optional `aiohttp` for the viewer.

## Install

```bash
pip install yazbunu              # logging only
pip install yazbunu[server]      # logging + web viewer
```

## Logging

```python
from yazbunu import init_logging, get_logger

init_logging(log_dir="./logs", project="myapp")
logger = get_logger("core.engine")

logger.info("request processed", request_id="r-42", duration_ms=120)
logger.error("payment failed", user_id="u-7", amount=99.90)
```

Each log line is a JSON object:

```json
{"ts": "2026-04-11T12:00:00+00:00", "level": "INFO", "src": "myapp.core.engine", "msg": "request processed", "request_id": "r-42", "duration_ms": 120}
```

### Bound loggers

```python
log = logger.bind(request_id="r-42", user_id="u-7")
log.info("step 1 done")   # request_id and user_id included automatically
log.info("step 2 done")
```

### Configuration

```python
from yazbunu import init_logging
from yazbunu.formatter import YazFormatter

# Custom context fields promoted to top-level keys
init_logging(
    project="myapp",
    quiet_libs=("boto3", "urllib3"),  # silence noisy loggers
)

# Or customize the formatter directly
formatter = YazFormatter(context_fields=frozenset({"request_id", "trace_id"}))
```

## Web Viewer

```bash
yazbunu-server --log-dir ./logs --port 9880
```

Open `http://localhost:9880` to view logs with:

- **File browser** with an "All files" merged view
- **Level filter** (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- **Time range** filter (5m/15m/1h/6h/24h)
- **Regex search** with match highlighting
- **Click-to-expand** long messages
- **Cross-component tracing** — click a trace pill (purple) to filter by that value across all files
- **Live tail** via WebSocket (falls back to HTTP polling)
- **Level counts** in status bar (e.g. "142 lines · 3 ERR · 12 WARN")
- **URL hash state** — bookmarkable/shareable filter state

### Custom trace keys

By default, the viewer highlights `task_id` and `mission_id` as trace keys. Override for your project:

```bash
yazbunu-server --log-dir ./logs --trace-keys request_id,user_id,trace_id
```

Or programmatically:

```python
from yazbunu.server import create_app
app = create_app("./logs", trace_keys=["request_id", "user_id"])
```

## License

MIT
