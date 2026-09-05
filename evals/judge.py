"""Groundedness scoring by LLM-as-judge.

The question: is every factual claim in the agent's output traceable to a
document or tool result that was actually in its context?

This is standard practice and genuinely imperfect. The limitations are stated
here rather than left to be discovered:

- **Shared blind spots.** A judge and an agent running the same model share
  failure modes, so a claim both find plausible passes regardless of truth. Run
  the judge on a different model where possible; the report records which.
- **Fluency bias.** Judges reward confident, well-structured text - which is
  exactly the failure mode being tested for.
- **Instability.** The same claim can score differently across runs. Temperature
  is pinned low, but this does not eliminate it.

The mitigation that matters is calibration: a manually graded subset, with the
agreement rate reported next to the score. A groundedness number without an
agreement rate is not interpretable, so `calibration.py` exists and the report
says so when it has not been run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from incidentiq.llm.base import LLMProvider, extract_json

JUDGE_SYSTEM = """\
You are grading whether an incident analysis is grounded in its evidence.

You will be given EVIDENCE (documents and tool results that were available) and
an ANALYSIS. Break the analysis into individual factual claims and judge each
one against the evidence only.

A claim is:
  "supported"    - the evidence directly states or clearly implies it
  "unsupported"  - the evidence neither states nor implies it (this includes
                   plausible-sounding claims that simply are not in the evidence)
  "contradicted" - the evidence says otherwise

Do not use outside knowledge. A claim that is true in general but absent from
this evidence is "unsupported". General advice and hedging ("this should be
monitored") is not a factual claim - ignore it.

Return JSON only:

{
  "claims": [
    {"claim": "...", "verdict": "supported|unsupported|contradicted",
     "evidence_ref": "doc id or tool name, or null"}
  ],
  "grounded_fraction": 0.0,
  "notes": "one sentence"
}
"""


@dataclass
class GroundednessResult:
    grounded_fraction: float
    n_claims: int
    n_supported: int
    n_unsupported: int
    n_contradicted: int
    claims: list[dict]
    judge_model: str
    notes: str = ""
    error: str | None = None

    @property
    def fully_grounded(self) -> bool:
        """The headline metric counts a response as grounded only when *every*
        claim is supported. A response that is 80% grounded still contains a
        fabrication, and an engineer cannot tell which fifth to distrust."""
        return self.n_claims > 0 and self.n_supported == self.n_claims


def _render_evidence(state: dict) -> str:
    parts = []
    for d in (state.get("retrieved_docs") or [])[:8]:
        parts.append(f"[{d['parent_doc_id']}] {d.get('title') or ''}\n{d['excerpt'][:700]}")
    for c in (state.get("tool_calls") or []):
        if c["status"] in ("success", "partial") and c.get("result"):
            body = json.dumps(c["result"], default=str)[:900]
            parts.append(f"[tool:{c['tool_name']}] {body}")
    return "\n\n".join(parts) or "(no evidence was available)"


def _render_analysis(state: dict) -> str:
    proposal = state.get("proposal") or {}
    parts = []
    if proposal.get("root_cause"):
        parts.append(f"Root cause: {proposal['root_cause']}")
    if proposal.get("remediation"):
        parts.append(f"Remediation: {proposal['remediation']}")
    if state.get("final_summary"):
        parts.append(f"Summary: {state['final_summary']}")
    return "\n\n".join(parts)


def score_groundedness(llm: LLMProvider, state: dict) -> GroundednessResult:
    analysis = _render_analysis(state)
    proposal = state.get("proposal") or {}

    if proposal.get("abstained") and not analysis.strip():
        # An abstention makes no factual claims, so it cannot be ungrounded.
        # Scoring it as 0 would penalise exactly the behaviour we want.
        return GroundednessResult(
            grounded_fraction=1.0, n_claims=0, n_supported=0, n_unsupported=0,
            n_contradicted=0, claims=[], judge_model=llm.model_name,
            notes="abstained; no factual claims to ground",
        )
    if not analysis.strip():
        return GroundednessResult(
            grounded_fraction=0.0, n_claims=0, n_supported=0, n_unsupported=0,
            n_contradicted=0, claims=[], judge_model=llm.model_name,
            notes="no analysis produced", error="empty analysis",
        )

    user = f"EVIDENCE:\n{_render_evidence(state)}\n\n---\n\nANALYSIS:\n{analysis}"
    resp = llm.complete_json(JUDGE_SYSTEM, [{"role": "user", "content": user}],
                             max_tokens=1600)
    parsed = extract_json(resp.text)

    if not parsed or "claims" not in parsed:
        return GroundednessResult(
            grounded_fraction=0.0, n_claims=0, n_supported=0, n_unsupported=0,
            n_contradicted=0, claims=[], judge_model=llm.model_name,
            error="judge returned no parseable verdict",
        )

    claims = [c for c in parsed["claims"] if isinstance(c, dict)]
    supported = sum(1 for c in claims if c.get("verdict") == "supported")
    unsupported = sum(1 for c in claims if c.get("verdict") == "unsupported")
    contradicted = sum(1 for c in claims if c.get("verdict") == "contradicted")

    # Recompute rather than trusting the model's own arithmetic - judges are
    # unreliable at counting their own output.
    fraction = supported / len(claims) if claims else 0.0

    return GroundednessResult(
        grounded_fraction=round(fraction, 4), n_claims=len(claims),
        n_supported=supported, n_unsupported=unsupported, n_contradicted=contradicted,
        claims=claims, judge_model=llm.model_name, notes=parsed.get("notes", ""),
    )
