"""Authentication helpers for yazbunu log viewer."""

import json
import os
import secrets
from pathlib import Path


def get_or_create_token(auth_dir: str) -> str:
    """Generate a random token, persist in auth_dir/auth.json, return it."""
    auth_path = Path(auth_dir) / "auth.json"
    if auth_path.is_file():
        try:
            data = json.loads(auth_path.read_text(encoding="utf-8"))
            if data.get("token"):
                return data["token"]
        except (json.JSONDecodeError, KeyError):
            pass
    token = secrets.token_urlsafe(32)
    os.makedirs(auth_dir, exist_ok=True)
    auth_path.write_text(json.dumps({"token": token}), encoding="utf-8")
    return token


def validate_token(request_token: str, expected: str) -> bool:
    """Constant-time comparison of tokens."""
    if not request_token or not expected:
        return False
    return secrets.compare_digest(request_token, expected)


def build_url(host: str, port: int, token: str, tls: bool = False) -> str:
    """Build viewer URL with token query parameter."""
    scheme = "https" if tls else "http"
    return f"{scheme}://{host}:{port}?token={token}"


def render_qr_page(url: str) -> str:
    """Return self-contained HTML page displaying the URL prominently with copy-on-click."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yazbunu — Access Link</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; margin: 0;
    background: #1a1a2e; color: #e0e0e0;
  }}
  .card {{
    background: #16213e; border-radius: 16px; padding: 3rem;
    max-width: 600px; width: 90%; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }}
  h1 {{ color: #0f3460; font-size: 1.5rem; margin-bottom: 0.5rem; }}
  h1 {{ color: #e94560; }}
  .url-box {{
    background: #0f3460; border-radius: 8px; padding: 1rem;
    margin: 1.5rem 0; word-break: break-all; cursor: pointer;
    font-family: monospace; font-size: 0.95rem; color: #a8d8ea;
    transition: background 0.2s;
  }}
  .url-box:hover {{ background: #1a4080; }}
  .hint {{ font-size: 0.85rem; color: #888; }}
  .copied {{ color: #4ecca3 !important; }}
</style>
</head>
<body>
<div class="card">
  <h1>Yazbunu Log Viewer</h1>
  <p>Open this link to access the log viewer:</p>
  <div class="url-box" id="url" onclick="copyUrl()">{url}</div>
  <p class="hint" id="hint">Click to copy</p>
</div>
<script>
function copyUrl() {{
  navigator.clipboard.writeText("{url}").then(function() {{
    var h = document.getElementById("hint");
    h.textContent = "Copied!";
    h.classList.add("copied");
    setTimeout(function() {{ h.textContent = "Click to copy"; h.classList.remove("copied"); }}, 2000);
  }});
}}
</script>
</body>
</html>"""
