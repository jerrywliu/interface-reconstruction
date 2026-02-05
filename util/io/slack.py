import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.request import HTTPRedirectHandler, build_opener
from urllib.error import HTTPError, URLError


DEFAULT_CHANNEL = "interface-reconstruction"

# Slack channel IDs look like C0123ABCD (public), G... (private), D... (DM), Z... (???)
_CHANNEL_ID_RE = re.compile(r"^[CGDZ][A-Z0-9]{8,}$")


class SlackSendError(RuntimeError):
    pass


def _find_git_root(start: Optional[Path] = None) -> Optional[Path]:
    """
    Best-effort search for the git repo root by walking up parents looking for .git/.
    This matches the behavior described in SLACK_RESULTS.md (repo-root-relative paths).
    """
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return None


def _default_env_path() -> Path:
    repo_root = _find_git_root(Path(__file__).resolve())
    if repo_root is not None:
        return repo_root / "config" / "local" / "slack.env"
    return Path(__file__).resolve().parent / "config" / "local" / "slack.env"


def _load_env_file():
    env_path = os.getenv("SLACK_ENV_FILE")
    path = Path(env_path).expanduser() if env_path else _default_env_path()
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
    return path


def load_slack_env():
    """
    Load local Slack env file (config/local/slack.env by default) into os.environ.
    Returns the path if found.
    """
    return _load_env_file()


def _resolve_path(file_path: Union[str, Path], repo_root: Optional[Path]) -> Path:
    p = Path(file_path)
    if p.is_absolute():
        return p

    # The contract in SLACK_RESULTS.md is "relative to repo root, not cwd".
    if repo_root is None:
        raise SlackSendError(
            f"Relative path provided but no git repo root found: {file_path!s}"
        )
    return (repo_root / p).resolve()


def send_results_to_slack(message, file_paths=None, channel=None):
    """
    Send a message (and optional files) to Slack.

    Uses one of:
    - SLACK_BOT_TOKEN (preferred) for chat.postMessage + external upload flow
    - SLACK_WEBHOOK_URL for message only (files not supported)
    """
    _load_env_file()
    file_paths = list(file_paths or [])
    channel = channel or os.getenv("SLACK_CHANNEL") or DEFAULT_CHANNEL

    token = os.getenv("SLACK_BOT_TOKEN")
    webhook = os.getenv("SLACK_WEBHOOK_URL")

    if token:
        ok, channel_id = _post_message(token, channel, message)
        if not ok:
            # If we can't post, it's very likely we also can't share files.
            return False

        repo_root = _find_git_root()

        all_ok = True
        for fp in file_paths:
            try:
                resolved = _resolve_path(fp, repo_root)
            except SlackSendError as e:
                print(f"Slack upload skipped (path resolution error): {e}")
                all_ok = False
                continue

            if not _upload_file(token, channel, resolved, channel_id):
                all_ok = False
        return all_ok

    if webhook:
        if file_paths:
            message = _append_file_list(message, file_paths)
        return _post_webhook(webhook, message)

    print("Slack not configured. Set SLACK_BOT_TOKEN or SLACK_WEBHOOK_URL.")
    return False


def _append_file_list(message, file_paths):
    lines = [message, "\nFiles:"]
    for file_path in file_paths:
        lines.append(f"- {file_path}")
    return "\n".join(lines)


def _post_webhook(webhook_url, message):
    payload = json.dumps({"text": message}).encode("utf-8")
    req = Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req) as resp:
            resp.read()
        return True
    except (HTTPError, URLError) as err:
        print(f"Slack webhook error: {err}")
        return False


def _parse_slack_response(resp, context):
    raw = resp.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        # Slack Web API should always return JSON; if we can't parse it, treat as failure.
        preview = raw[:200]
        print(f"Slack {context} error: non-JSON response (first 200 bytes): {preview!r}")
        return False, None

    if payload.get("ok", True):
        return True, payload

    error = payload.get("error", "unknown_error")
    details = []
    for key in ("needed", "required_scope", "provided"):
        value = payload.get(key)
        if value:
            details.append(f"{key}={value}")
    metadata = payload.get("response_metadata", {})
    messages = metadata.get("messages") if isinstance(metadata, dict) else None
    if messages:
        details.append("messages=" + ",".join(messages))
    detail_text = f" ({', '.join(details)})" if details else ""
    print(f"Slack {context} error: {error}{detail_text}")
    return False, payload


def _post_message(token, channel, message):
    if channel is None:
        raise ValueError("SLACK_CHANNEL is required when using SLACK_BOT_TOKEN")
    url = "https://slack.com/api/chat.postMessage"
    payload = json.dumps({"channel": channel, "text": message}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    req = Request(url, data=payload, headers=headers)
    try:
        with urlopen(req) as resp:
            ok, payload = _parse_slack_response(resp, "chat.postMessage")
            channel_id = None
            if isinstance(payload, dict):
                channel_id = payload.get("channel")
            return ok, channel_id
    except (HTTPError, URLError) as err:
        print(f"Slack chat.postMessage error: {err}")
        return False, None


def _upload_file(token, channel, file_path: Path, channel_id=None):
    if channel is None:
        raise ValueError("SLACK_CHANNEL is required when using SLACK_BOT_TOKEN")

    path = Path(file_path)
    if not path.exists():
        print(f"Slack upload skipped (missing file): {file_path}")
        # Missing files are non-fatal; caller decides whether to fail the run.
        return False

    return _upload_file_v2(token, channel, path, channel_id)


def _upload_file_v2(token, channel, path: Path, channel_id=None):
    # Request an upload URL + file_id
    length = path.stat().st_size
    if length <= 0:
        print(f"Slack upload skipped (empty file): {path}")
        return False

    get_url = "https://slack.com/api/files.getUploadURLExternal"
    payload = urlencode({"filename": path.name, "length": length}).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}",
    }
    req = Request(get_url, data=payload, headers=headers)

    try:
        with urlopen(req) as resp:
            ok, data = _parse_slack_response(resp, "files.getUploadURLExternal")
            if not ok or not isinstance(data, dict):
                return False
            upload_url = data.get("upload_url")
            file_id = data.get("file_id")
            if not upload_url or not file_id:
                print("Slack files.getUploadURLExternal missing upload_url/file_id.")
                return False
    except (HTTPError, URLError) as err:
        print(f"Slack files.getUploadURLExternal error: {err}")
        return False

    # Upload the bytes to Slack's upload service.
    # Slack's docs describe an HTTP POST with Content-Type: application/octet-stream.
    # See: https://docs.slack.dev/messaging/working-with-files/
    if not _post_with_redirects(
        upload_url,
        path.read_bytes(),
        {"Content-Type": "application/octet-stream"},
    ):
        return False

    # Finalize + share to a channel.
    # If no channel_id is provided, Slack will keep the file private.
    # See: https://docs.slack.dev/reference/methods/files.completeUploadExternal/
    resolved_channel_id = None
    if channel_id:
        resolved_channel_id = channel_id
    elif channel and _CHANNEL_ID_RE.match(channel):
        resolved_channel_id = channel

    if not resolved_channel_id:
        print(
            "Slack files.completeUploadExternal requires a channel ID to share the file. "
            "Set SLACK_CHANNEL to a channel ID (e.g. C0123AB4CDE), or ensure chat.postMessage succeeds."
        )
        return False

    complete_url = "https://slack.com/api/files.completeUploadExternal"
    files_param = json.dumps([{"id": file_id, "title": path.name}])

    complete_body = urlencode({"files": files_param, "channel_id": resolved_channel_id}).encode(
        "utf-8"
    )
    complete_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}",
    }

    complete_req = Request(complete_url, data=complete_body, headers=complete_headers)
    try:
        with urlopen(complete_req) as resp:
            ok, _ = _parse_slack_response(resp, "files.completeUploadExternal")
            return ok
    except (HTTPError, URLError) as err:
        print(f"Slack files.completeUploadExternal error: {err}")
        return False


def _post_with_redirects(url, data, headers, max_redirects=3):
    """
    Slack's upload_url may redirect. Follow redirects while preserving POST.
    Slack returns HTTP 200 on success, with a non-JSON body like "OK - 12".
    """
    current_url = url
    opener = build_opener(HTTPRedirectHandler())
    for _ in range(max_redirects + 1):
        req = Request(current_url, data=data, headers=headers, method="POST")
        try:
            with opener.open(req) as resp:
                status = getattr(resp, "status", None) or 200
                if 300 <= status < 400:
                    location = resp.headers.get("Location")
                    if not location:
                        print("Slack upload redirect missing Location header.")
                        return False
                    current_url = location
                    continue
                # consume body (may be "OK - <bytes>")
                resp.read()
                return 200 <= status < 300
        except HTTPError as err:
            # Some redirects surface as HTTPError in urllib.
            if err.code in (301, 302, 303, 307, 308):
                location = err.headers.get("Location")
                if not location:
                    print("Slack upload redirect missing Location header.")
                    return False
                current_url = location
                continue
            body = b""
            try:
                body = err.read()  # type: ignore[attr-defined]
            except Exception:
                pass
            preview = body[:200]
            print(f"Slack upload POST error: {err} (first 200 bytes: {preview!r})")
            return False
        except URLError as err:
            print(f"Slack upload POST error: {err}")
            return False
    print("Slack upload POST error: too many redirects.")
    return False
