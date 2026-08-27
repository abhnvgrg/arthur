from __future__ import annotations

import hmac
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, status

TOKEN_BYTES = 32
TOKEN_HEADER = "Authorization"
TOKEN_QUERY = "token"


def config_path() -> Path:
    configured = os.getenv("ARTHUR_CONFIG_FILE")
    if configured:
        return Path(configured)
    return Path.home() / ".arthur" / "config.json"


def _restrict_permissions(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except (OSError, NotImplementedError):
        pass


def load_or_create_token(path: Path | None = None) -> str:
    """Return the API token, minting and storing one on first run.

    An environment variable wins if set, so a deployment can supply its own
    without a file on disk. Otherwise the token is generated once and kept in
    a user-only-readable config file, which is what makes the server usable
    without the user inventing a secret.
    """
    from_env = os.getenv("ARTHUR_API_TOKEN", "").strip()
    if from_env:
        return from_env

    location = path if path is not None else config_path()

    if location.exists():
        try:
            stored = json.loads(location.read_text(encoding="utf-8")).get("api_token")
        except (json.JSONDecodeError, OSError):
            stored = None
        if stored:
            return stored

    token = secrets.token_urlsafe(TOKEN_BYTES)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps({"api_token": token}, indent=2), encoding="utf-8")
    _restrict_permissions(location)
    return token


def presented_token(request: Request) -> Optional[str]:
    """Read the caller's token from the header, or the query string.

    EventSource cannot set headers, so the SSE endpoint has no way to send an
    Authorization header. A query parameter is the only option the browser API
    leaves open; it is accepted for that reason and no other.
    """
    header = request.headers.get(TOKEN_HEADER, "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()

    query = request.query_params.get(TOKEN_QUERY, "")
    return query.strip() or None


class TokenGuard:
    """Rejects any request that does not present the API token.

    Comparison is constant-time. The token is a 32-byte random string, so a
    timing side channel is not a realistic route in, but the cost of doing it
    correctly is one function call.
    """

    def __init__(self, token: str, enabled: bool = True) -> None:
        self.token = token
        self.enabled = enabled

    async def __call__(self, request: Request) -> None:
        if not self.enabled:
            return

        candidate = presented_token(request)
        if candidate is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="This API requires a token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not hmac.compare_digest(candidate, self.token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="That token is not valid.",
                headers={"WWW-Authenticate": "Bearer"},
            )


def binding_is_public(host: str) -> bool:
    return host not in {"127.0.0.1", "::1", "localhost"}
