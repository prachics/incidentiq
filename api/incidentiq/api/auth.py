"""API keys with two roles: `viewer` reads and starts investigations, `approver`
also decides approvals.

The approver recorded on a decision is the user behind the key, never a field
in the request body. Keys are static configuration compared in constant time;
an identity provider in front would only change `_authenticate`.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from incidentiq.config import get_settings

Role = Literal["viewer", "approver"]
_RANK = {"viewer": 0, "approver": 1}

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user: str
    role: Role


def parse_api_keys(spec: str) -> dict[str, Principal]:
    """`key:user:role,key:user:role` -> {key: Principal}. Fails loudly on a
    malformed entry: a silently dropped key is an approver who cannot approve."""
    keys: dict[str, Principal] = {}
    entries = [e.strip() for e in spec.split(",") if e.strip()]
    for position, entry in enumerate(entries, start=1):
        # Errors name the entry's position, never its text: it contains a key.
        parts = entry.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"API_KEYS entry {position} must be key:user:role")
        key, user, role = parts
        if role not in _RANK:
            raise ValueError(f"API_KEYS entry {position} has unknown role {role!r}")
        keys[key] = Principal(user=user, role=role)  # type: ignore[arg-type]
    return keys


def _authenticate(token: str) -> Principal | None:
    for key, principal in parse_api_keys(get_settings().api_keys).items():
        if hmac.compare_digest(key.encode(), token.encode()):
            return principal
    return None


def require(role: Role):
    def dependency(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> Principal:
        principal = _authenticate(creds.credentials) if creds else None
        if principal is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or unknown API key",
                                headers={"WWW-Authenticate": "Bearer"})
        if _RANK[principal.role] < _RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                f"{principal.user} is a {principal.role}; this needs {role}")
        return principal
    return dependency


Viewer = Annotated[Principal, Depends(require("viewer"))]
Approver = Annotated[Principal, Depends(require("approver"))]
