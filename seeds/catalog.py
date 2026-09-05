"""The fictional production estate.

Hand-authored rather than randomly generated, because the *shape* of the
dependency graph is what makes multi-service scenarios meaningful. A cascading
failure is only interesting if the cascade follows real edges: payment-service
degrades, so checkout-service times out waiting on it, so api-gateway starts
returning 502s. Random edges produce cascades that teach nothing.

Every service referenced anywhere in the seed data must appear here. The seed
loader asserts this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["tier-1", "tier-2", "tier-3"]
DepKind = Literal["sync", "async", "datastore"]


@dataclass(frozen=True)
class Service:
    name: str
    tier: Tier
    language: str
    owner_team: str
    replica_count: int
    description: str
    # Service "kind" drives which failure archetypes can apply to it.
    kind: Literal["app", "gateway", "datastore", "cache", "queue", "search", "adapter"] = "app"
    depends_on: tuple[tuple[str, DepKind], ...] = field(default=())


SERVICES: tuple[Service, ...] = (
    # ── Edge ────────────────────────────────────────────────
    Service(
        "web-frontend", "tier-1", "typescript", "storefront", 6,
        "Server-rendered storefront. All customer traffic enters here.",
        kind="app", depends_on=(("api-gateway", "sync"),),
    ),
    Service(
        "api-gateway", "tier-1", "go", "platform", 8,
        "Edge router. Terminates TLS, authenticates, fans out to core services.",
        kind="gateway",
        depends_on=(
            ("auth-service", "sync"), ("checkout-service", "sync"),
            ("catalog-service", "sync"), ("search-service", "sync"),
            ("user-service", "sync"), ("order-service", "sync"),
            ("shipping-service", "sync"), ("recommendation-service", "sync"),
        ),
    ),
    # ── Core (tier-1) ───────────────────────────────────────
    Service(
        "auth-service", "tier-1", "go", "identity", 6,
        "Issues and validates session tokens. Every authenticated request hits it.",
        depends_on=(("user-db", "datastore"), ("redis-sessions", "datastore")),
    ),
    Service(
        "checkout-service", "tier-1", "java", "commerce", 8,
        "Orchestrates the checkout flow: cart -> inventory hold -> payment -> order.",
        depends_on=(
            ("payment-service", "sync"), ("inventory-service", "sync"),
            ("cart-service", "sync"), ("order-service", "sync"),
        ),
    ),
    Service(
        "payment-service", "tier-1", "java", "payments", 6,
        "Authorises and captures payments. PCI scope.",
        depends_on=(
            ("stripe-adapter", "sync"), ("payment-db", "datastore"),
            ("fraud-service", "sync"),
        ),
    ),
    Service(
        "order-service", "tier-1", "java", "commerce", 6,
        "System of record for orders. Emits order events downstream.",
        depends_on=(("order-db", "datastore"), ("kafka-events", "async")),
    ),
    # ── Core (tier-2) ───────────────────────────────────────
    Service(
        "inventory-service", "tier-2", "go", "supply-chain", 4,
        "Stock levels and reservation holds.",
        depends_on=(("inventory-db", "datastore"), ("warehouse-sync", "async")),
    ),
    Service(
        "cart-service", "tier-2", "python", "commerce", 4,
        "Shopping cart state. Read-heavy, Redis-backed.",
        depends_on=(("redis-cart", "datastore"),),
    ),
    Service(
        "catalog-service", "tier-2", "python", "catalog", 5,
        "Product metadata, pricing, and availability.",
        depends_on=(("catalog-db", "datastore"), ("search-service", "sync")),
    ),
    Service(
        "search-service", "tier-2", "java", "catalog", 5,
        "Product search and faceting.",
        kind="search", depends_on=(("elasticsearch-cluster", "datastore"),),
    ),
    Service(
        "user-service", "tier-2", "go", "identity", 4,
        "User profiles, addresses, preferences.",
        depends_on=(("user-db", "datastore"),),
    ),
    Service(
        "fraud-service", "tier-2", "python", "risk", 4,
        "Scores transactions for fraud risk before authorisation.",
        depends_on=(("fraud-db", "datastore"), ("ml-scoring-service", "sync")),
    ),
    Service(
        "shipping-service", "tier-2", "go", "supply-chain", 4,
        "Rate quotes and label generation.",
        depends_on=(("shipping-db", "datastore"), ("carrier-adapter", "sync")),
    ),
    # ── Supporting (tier-3) ─────────────────────────────────
    Service(
        "notification-service", "tier-3", "python", "growth", 3,
        "Transactional email and SMS. Consumes order events.",
        depends_on=(
            ("email-adapter", "sync"), ("sms-adapter", "sync"),
            ("kafka-events", "async"),
        ),
    ),
    Service(
        "ml-scoring-service", "tier-3", "python", "risk", 3,
        "Serves the fraud model. GPU-free, CPU-bound inference.",
        depends_on=(("feature-store", "datastore"),),
    ),
    Service(
        "recommendation-service", "tier-3", "python", "growth", 3,
        "Related-products and homepage recommendations.",
        depends_on=(("feature-store", "datastore"), ("catalog-db", "datastore")),
    ),
    # ── Datastores ──────────────────────────────────────────
    Service("order-db", "tier-1", "postgres", "platform", 3,
            "Primary + 2 replicas. Orders and line items.", kind="datastore"),
    Service("payment-db", "tier-1", "postgres", "payments", 3,
            "Primary + 2 replicas. Payment intents and captures.", kind="datastore"),
    Service("user-db", "tier-1", "postgres", "identity", 3,
            "Primary + 2 replicas. Accounts and credentials.", kind="datastore"),
    Service("inventory-db", "tier-2", "postgres", "supply-chain", 2,
            "Primary + 1 replica. Stock and reservations.", kind="datastore"),
    Service("catalog-db", "tier-2", "postgres", "catalog", 2,
            "Primary + 1 replica. Product records.", kind="datastore"),
    Service("fraud-db", "tier-2", "postgres", "risk", 2,
            "Primary + 1 replica. Risk decisions and rules.", kind="datastore"),
    Service("shipping-db", "tier-2", "postgres", "supply-chain", 2,
            "Primary + 1 replica. Shipments and labels.", kind="datastore"),
    Service("feature-store", "tier-3", "postgres", "risk", 2,
            "Precomputed features for the fraud and recs models.", kind="datastore"),
    Service("redis-sessions", "tier-1", "redis", "identity", 3,
            "Session token cache. 3-node cluster.", kind="cache"),
    Service("redis-cart", "tier-2", "redis", "commerce", 3,
            "Cart state cache. 3-node cluster.", kind="cache"),
    Service("elasticsearch-cluster", "tier-2", "elasticsearch", "catalog", 5,
            "5-node search cluster. Product index.", kind="search"),
    Service("kafka-events", "tier-2", "kafka", "platform", 3,
            "Event bus. 3 brokers, order and inventory topics.", kind="queue"),
    # ── External adapters ───────────────────────────────────
    Service("stripe-adapter", "tier-1", "go", "payments", 4,
            "Wraps the Stripe API. Retries and idempotency keys.", kind="adapter"),
    Service("email-adapter", "tier-3", "go", "growth", 2,
            "Wraps the transactional email provider.", kind="adapter"),
    Service("sms-adapter", "tier-3", "go", "growth", 2,
            "Wraps the SMS provider.", kind="adapter"),
    Service("carrier-adapter", "tier-2", "go", "supply-chain", 3,
            "Wraps carrier rate and label APIs.", kind="adapter"),
    Service("warehouse-sync", "tier-2", "python", "supply-chain", 2,
            "Batch sync of warehouse stock counts. Runs every 5 minutes.", kind="adapter"),
)

BY_NAME: dict[str, Service] = {s.name: s for s in SERVICES}


def dependencies() -> list[tuple[str, str, str]]:
    """Flatten to (from_service, to_service, kind) edges."""
    return [(s.name, target, kind) for s in SERVICES for target, kind in s.depends_on]


def dependents_of(name: str) -> list[str]:
    """Services that call `name` directly. Used to build cascade scenarios."""
    return [s.name for s in SERVICES if any(t == name for t, _ in s.depends_on)]


def upstream_chain(name: str, depth: int = 3) -> list[str]:
    """Walk callers-of-callers, breadth-first. A failure in `name` can surface
    as a symptom in any of these."""
    seen, frontier, out = {name}, [name], []
    for _ in range(depth):
        nxt = []
        for node in frontier:
            for caller in dependents_of(node):
                if caller not in seen:
                    seen.add(caller)
                    out.append(caller)
                    nxt.append(caller)
        frontier = nxt
        if not frontier:
            break
    return out


def validate() -> None:
    """Every dependency target must be a real service, and the graph must be
    acyclic. Called by the seed loader before anything is written."""
    for svc in SERVICES:
        for target, _ in svc.depends_on:
            if target not in BY_NAME:
                raise ValueError(f"{svc.name} depends on unknown service {target!r}")

    colour: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        colour[node] = 1  # grey: on the current path
        for target, _ in BY_NAME[node].depends_on:
            if colour.get(target) == 1:
                cycle = " -> ".join(path + [node, target])
                raise ValueError(f"dependency cycle: {cycle}")
            if colour.get(target, 0) == 0:
                visit(target, path + [node])
        colour[node] = 2  # black: fully explored

    for svc in SERVICES:
        if colour.get(svc.name, 0) == 0:
            visit(svc.name, [])


def is_entry_point(name: str) -> bool:
    """True if nothing calling this service is a bug.

    Two legitimate cases: the public entry point, and event consumers, which are
    driven by a queue rather than by a caller. Anything else with no callers is
    a missing edge in the graph.
    """
    svc = BY_NAME[name]
    if not dependents_of(name):
        if name == "web-frontend":
            return True
        return any(
            kind == "async" and BY_NAME[target].kind == "queue"
            for target, kind in svc.depends_on
        )
    return True
