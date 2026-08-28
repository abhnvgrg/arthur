from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional

from fastapi import Depends, HTTPException, Request, status

TOKEN_BYTES = 32
TOKEN_HEADER = "Authorization"
TICKET_QUERY = "ticket"
TICKET_TTL_SECONDS = 30.0
CONFIG_VERSION = 2
OWNER_NAME = "owner"
ENV_PRINCIPAL_ID = "env"


class Scope(str, Enum):
    READ = "read"
    CHAT = "chat"
    APPROVE = "approve"
    ADMIN = "admin"


ALL_SCOPES = frozenset(Scope)


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


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_scopes(raw: Iterable[str]) -> frozenset[Scope]:
    """Turn stored or requested scope names into scopes, refusing unknown ones."""
    scopes = set()
    for name in raw:
        try:
            scopes.add(Scope(name))
        except ValueError as error:
            raise ValueError(f"Unknown scope: {name!r}") from error
    return frozenset(scopes)


@dataclass(frozen=True)
class Principal:
    """Who a request is from, and what that identity is allowed to do."""

    id: str
    name: str
    scopes: frozenset[Scope]

    def allows(self, scope: Scope) -> bool:
        return scope in self.scopes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "scopes": sorted(scope.value for scope in self.scopes),
        }


ANONYMOUS = Principal(id="anonymous", name="anonymous", scopes=ALL_SCOPES)


@dataclass
class TokenRecord:
    id: str
    name: str
    hashed: str
    scopes: frozenset[Scope]
    created_at: float
    revoked_at: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def principal(self) -> Principal:
        return Principal(id=self.id, name=self.name, scopes=self.scopes)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hashed": self.hashed,
            "scopes": sorted(scope.value for scope in self.scopes),
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
        }

    def to_public(self) -> dict:
        record = self.to_dict()
        record.pop("hashed")
        record["active"] = self.active
        return record

    @classmethod
    def from_dict(cls, raw: dict) -> Optional[TokenRecord]:
        try:
            return cls(
                id=str(raw["id"]),
                name=str(raw["name"]),
                hashed=str(raw["hashed"]),
                scopes=parse_scopes(raw.get("scopes", [])),
                created_at=float(raw.get("created_at", 0.0)),
                revoked_at=(
                    float(raw["revoked_at"])
                    if raw.get("revoked_at") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None


class TokenStore:
    """Named, scoped, revocable API tokens, stored as hashes.

    Nothing on disk can be replayed as a credential. The plaintext of a token
    exists once, in the response that mints it; after that only its SHA-256
    hash is kept, so a leaked config file grants nothing.
    """

    def __init__(self, path: Path | None = None, load: bool = True) -> None:
        self.path = path if path is not None else config_path()
        self._records: dict[str, TokenRecord] = {}
        self._by_hash: dict[str, str] = {}
        self._environment: Optional[TokenRecord] = None
        self._adopt_environment_token()
        if load:
            self.load()

    def _adopt_environment_token(self) -> None:
        from_env = os.getenv("ARTHUR_API_TOKEN", "").strip()
        if not from_env:
            return
        self._environment = TokenRecord(
            id=ENV_PRINCIPAL_ID,
            name="environment",
            hashed=hash_token(from_env),
            scopes=ALL_SCOPES,
            created_at=time.time(),
        )

    def _index(self, record: TokenRecord) -> None:
        self._records[record.id] = record
        self._by_hash[record.hashed] = record.id

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(raw, dict):
            return

        legacy = raw.get("api_token")
        if legacy and not raw.get("tokens"):
            self._index(
                TokenRecord(
                    id=secrets.token_hex(8),
                    name=OWNER_NAME,
                    hashed=hash_token(str(legacy)),
                    scopes=ALL_SCOPES,
                    created_at=time.time(),
                )
            )
            self.save()
            return

        for entry in raw.get("tokens", []):
            if not isinstance(entry, dict):
                continue
            record = TokenRecord.from_dict(entry)
            if record is not None:
                self._index(record)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "tokens": [record.to_dict() for record in self._records.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _restrict_permissions(self.path)

    def records(self) -> list[TokenRecord]:
        return list(self._records.values())

    def active_records(self) -> list[TokenRecord]:
        return [record for record in self._records.values() if record.active]

    def adopt(
        self, name: str, secret: str, scopes: Iterable[Scope] | None = None
    ) -> TokenRecord:
        """Register a token whose plaintext the caller already holds.

        Used to seed a store from a supplied secret without writing to disk,
        which is how a test or an embedded deployment gets a known credential.
        """
        record = TokenRecord(
            id=secrets.token_hex(8),
            name=name,
            hashed=hash_token(secret),
            scopes=frozenset(scopes) if scopes is not None else ALL_SCOPES,
            created_at=time.time(),
        )
        self._index(record)
        return record

    def issue(
        self, name: str, scopes: Iterable[Scope] | None = None
    ) -> tuple[TokenRecord, str]:
        """Mint a token and return it alongside its only plaintext."""
        label = name.strip()
        if not label:
            raise ValueError("A token needs a name.")
        secret = secrets.token_urlsafe(TOKEN_BYTES)
        record = TokenRecord(
            id=secrets.token_hex(8),
            name=label,
            hashed=hash_token(secret),
            scopes=frozenset(scopes) if scopes is not None else ALL_SCOPES,
            created_at=time.time(),
        )
        self._index(record)
        self.save()
        return record, secret

    def revoke(self, token_id: str) -> bool:
        record = self._records.get(token_id)
        if record is None or not record.active:
            return False
        record.revoked_at = time.time()
        self.save()
        return True

    def resolve(self, secret: str) -> Optional[Principal]:
        digest = hash_token(secret)

        if self._environment is not None and hmac.compare_digest(
            digest, self._environment.hashed
        ):
            return self._environment.principal()

        token_id = self._by_hash.get(digest)
        if token_id is None:
            return None
        record = self._records[token_id]
        if not record.active or not hmac.compare_digest(digest, record.hashed):
            return None
        return record.principal()

    def ensure_owner(self) -> Optional[str]:
        """Mint the first owner token if this installation has none.

        Returns the plaintext when one was created, and None when a usable
        credential already exists. This is the only moment the owner token can
        be read, which is why the server prints it exactly once.
        """
        if self._environment is not None or self.active_records():
            return None
        _, secret = self.issue(OWNER_NAME, ALL_SCOPES)
        return secret


@dataclass
class Ticket:
    principal: Principal
    expires_at: float


class StreamTickets:
    """Single-use, short-lived credentials for the event stream.

    `EventSource` cannot set an Authorization header, so the browser has to put
    something in the URL. A ticket is what goes there instead of the API token:
    it is redeemed once, expires in seconds, and grants nothing but a stream
    subscription, so its appearance in a proxy log or browser history is not a
    disclosure of the caller's real credential.
    """

    def __init__(
        self,
        ttl: float = TICKET_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl
        self.clock = clock
        self._tickets: dict[str, Ticket] = {}

    def _prune(self) -> None:
        now = self.clock()
        for value in [k for k, t in self._tickets.items() if t.expires_at <= now]:
            self._tickets.pop(value, None)

    def issue(self, principal: Principal) -> str:
        self._prune()
        value = secrets.token_urlsafe(TOKEN_BYTES)
        self._tickets[value] = Ticket(
            principal=principal, expires_at=self.clock() + self.ttl
        )
        return value

    def redeem(self, value: str) -> Optional[Principal]:
        self._prune()
        ticket = self._tickets.pop(value, None)
        if ticket is None or ticket.expires_at <= self.clock():
            return None
        return ticket.principal

    def __len__(self) -> int:
        self._prune()
        return len(self._tickets)


class RateLimiter:
    """A sliding window of request timestamps per caller.

    Keyed by principal rather than by address, so a leaked token cannot be used
    faster than the limit no matter where it is used from.
    """

    def __init__(
        self,
        limit: int,
        window: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("A rate limit must allow at least one request.")
        self.limit = limit
        self.window = window
        self.clock = clock
        self._hits: dict[str, Deque[float]] = {}

    def check(self, key: str) -> Optional[float]:
        """Record a request, returning seconds to wait if it exceeds the limit."""
        now = self.clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(0.0, hits[0] + self.window - now)
        hits.append(now)
        return None

    def reset(self) -> None:
        self._hits.clear()


def presented_token(request: Request) -> Optional[str]:
    """Read the caller's bearer token.

    Only the header is read. Tokens are no longer accepted from the query
    string; the event stream uses a ticket instead.
    """
    header = request.headers.get(TOKEN_HEADER, "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


class Authenticator:
    """Turns a request into a principal, and gates routes on scopes.

    Every guarded route declares the scope it needs. A token that can read the
    audit log cannot approve a destructive call unless it was issued that
    scope, which is the difference between one shared secret and an identity.
    """

    def __init__(
        self,
        store: TokenStore,
        tickets: StreamTickets | None = None,
        limiter: RateLimiter | None = None,
        chat_limiter: RateLimiter | None = None,
        enabled: bool = True,
    ) -> None:
        self.store = store
        self.tickets = tickets or StreamTickets()
        self.limiter = limiter
        self.chat_limiter = chat_limiter
        self.enabled = enabled

    def identify(self, request: Request) -> Principal:
        if not self.enabled:
            return ANONYMOUS
        secret = presented_token(request)
        if secret is None:
            raise _unauthorized("This API requires a token.")
        principal = self.store.resolve(secret)
        if principal is None:
            raise _unauthorized("That token is not valid.")
        return principal

    def _limit(self, principal: Principal, scope: Scope) -> None:
        limiter = self.chat_limiter if scope is Scope.CHAT else self.limiter
        if limiter is None:
            return
        retry_after = limiter.check(f"{principal.id}:{scope.value}")
        if retry_after is None:
            return
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests.",
            headers={"Retry-After": str(max(1, int(retry_after) + 1))},
        )

    def requires(self, scope: Scope) -> Callable:
        async def dependency(request: Request) -> Principal:
            principal = self.identify(request)
            if self.enabled and not principal.allows(scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"This token lacks the {scope.value} scope.",
                )
            self._limit(principal, scope)
            request.state.principal = principal
            return principal

        return dependency

    def requires_ticket(self) -> Callable:
        async def dependency(request: Request) -> Principal:
            if not self.enabled:
                return ANONYMOUS
            value = request.query_params.get(TICKET_QUERY, "").strip()
            if not value:
                raise _unauthorized("The event stream requires a ticket.")
            principal = self.tickets.redeem(value)
            if principal is None:
                raise _unauthorized("That ticket is expired or already used.")
            if not principal.allows(Scope.READ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="This token lacks the read scope.",
                )
            request.state.principal = principal
            return principal

        return dependency

    def guard(self, scope: Scope) -> Depends:
        return Depends(self.requires(scope))


def binding_is_public(host: str) -> bool:
    return host not in {"127.0.0.1", "::1", "localhost"}
