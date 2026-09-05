"""Tests for the service catalog.

These guard the property the whole corpus rests on: if the dependency graph is
wrong, every "internally consistent" claim about the seed data is wrong too.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "seeds"))

import archetypes as A  # noqa: E402
import catalog as C  # noqa: E402


def test_graph_is_valid_and_acyclic():
    C.validate()  # raises on unknown target or cycle


def test_every_dependency_target_exists():
    for svc in C.SERVICES:
        for target, _ in svc.depends_on:
            assert target in C.BY_NAME, f"{svc.name} -> unknown {target}"


def test_service_names_are_unique():
    names = [s.name for s in C.SERVICES]
    assert len(names) == len(set(names))


def test_datastores_have_no_outbound_dependencies():
    """A datastore that calls an application service would make the cascade
    direction ambiguous, and cascade scenarios depend on it being clear."""
    for svc in C.SERVICES:
        if svc.kind in ("datastore", "cache", "queue"):
            assert not svc.depends_on, f"{svc.name} ({svc.kind}) should be a leaf"


def test_cascade_walk_follows_real_edges():
    chain = C.upstream_chain("payment-db")
    assert "payment-service" in chain      # direct caller
    assert "checkout-service" in chain     # caller of the caller
    assert "payment-db" not in chain       # does not include itself


def test_upstream_chain_terminates_on_leaf():
    assert C.upstream_chain("web-frontend") == []  # nothing calls the frontend


@pytest.mark.parametrize("svc", C.SERVICES, ids=lambda s: s.name)
def test_every_service_is_reachable_or_is_an_entry_point(svc):
    """Every service is either called by something, is the public entry point,
    or is an event consumer driven by a queue. Anything else with no callers is
    a missing edge in the graph, not a design choice."""
    assert C.is_entry_point(svc.name), (
        f"{svc.name} is called by nothing, is not the public entry point, and "
        "does not consume from a queue - the graph is missing an edge"
    )


class TestArchetypes:
    def test_keys_are_unique(self):
        keys = [a.key for a in A.ARCHETYPES]
        assert len(keys) == len(set(keys))

    def test_every_archetype_applies_to_at_least_one_service(self):
        for arch in A.ARCHETYPES:
            matches = [
                s for s in C.SERVICES
                if s.kind in arch.applies_to
                and (not arch.languages or s.language in arch.languages)
            ]
            assert matches, f"archetype {arch.key} matches no service"

    def test_every_service_kind_has_at_least_one_archetype(self):
        kinds = {s.kind for s in C.SERVICES}
        covered = {k for a in A.ARCHETYPES for k in a.applies_to}
        assert kinds <= covered, f"uncovered service kinds: {kinds - covered}"

    def test_templates_render_without_missing_fields(self):
        """Every template must be renderable with the fields the generator
        supplies. A stray {placeholder} would raise at seed time."""
        kw = {"service": "svc", "dep": "dep", "version": "v1.0.0", "team": "team"}
        for arch in A.ARCHETYPES:
            for group in (arch.symptoms, arch.root_cause, arch.resolution):
                for tmpl in group:
                    tmpl.format(**kw)  # raises KeyError / IndexError on a bad template

    def test_dep_referencing_archetypes_have_dep_in_templates(self):
        """If needs_dep is set, at least one template should actually use it -
        otherwise the flag is silently doing nothing."""
        for arch in A.ARCHETYPES:
            if not arch.needs_dep:
                continue
            all_text = " ".join(arch.symptoms + arch.root_cause + arch.resolution)
            assert "{dep}" in all_text, f"{arch.key} sets needs_dep but never uses it"
