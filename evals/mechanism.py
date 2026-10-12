"""Which failure mode a conclusion describes.

Each archetype has phrasings that assert its failure ("the connection pool was
exhausted", "the volume filled", "replication lag"), written from its corpus
narratives and kept specific: naming a component with any failure word ("disk I/O
has high latency", "a disk hardware failure") asserts no failure mode. Only what a
conclusion asserts has failed is read: "the disk is healthy", "no disk pressure" or
"checked the disk" names none, and a denial covers only its own part of a sentence
("the disk is full and cpu is normal" still names a full disk). A conclusion names
an archetype when one of its phrasings matches and no other archetype matches more.
Validated by `tests/test_evals.py`: every corpus narrative as its own archetype and
no other, and answers written apart from the corpus, right and nearly right.

The vocabulary below (words only one archetype's narratives use) is reported beside
a grade, as how many of its terms an answer used; it does not decide the grade.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "seeds"))

import archetypes as A  # noqa: E402

MIN_HITS = 2   # for scenarios graded by their own keyword list

# Common words that carry no failure mode, however rarely the narratives use them.
_STOP = {"service", "services", "requests", "request", "because", "before", "after", "which",
         "while", "their", "there", "every", "other", "under", "about", "where", "would",
         "could", "until", "within", "without", "never", "still", "since", "though", "first",
         "these", "those"}
_WORD = re.compile(r"[a-z]+")

_CLAUSES = re.compile(r"[.;:!?\n]+|\b(?:but|however|although|whereas)\b")
# A clause's parts ("the disk is full, and cpu is normal"), each part's cause segments
# ("X is slow | because | Y has a full disk": the full disk is Y's), and within a
# segment the pieces a denial can cover ("the pool is exhausted with | no disk
# pressure"). A failure is asserted of its segment; a denial drops only its piece.
_PARTS = re.compile(r",|\b(?:and|while)\b")
_CAUSES = re.compile(r"\b(?:because|since|due to|caused by|as a result of)\b")
_PIECES = re.compile(r"\bwith\b")
# A segment ending in a negation denies the cause after it: "not due to disk exhaustion".
_NEGATES_CAUSE = re.compile(r"(?:\bnot|\bno|n't|\bnever)\s*$")
# A part saying something is fine, is not the cause, or did not happen names no failure.
_DENIES = re.compile(
    r"\b(?:no|not|without)\s+(?:[a-z_-]+\s+){0,3}?(?:pressure|exhaust\w*|leak\w*|saturat\w*|"
    r"errors?|failures?|faults?|contention|spikes?|anomal\w*|issues?|problems?|"
    r"backlog|lag|evictions?|throttl\w*|regressions?|outages?|degradation|starvation|"
    r"deadlocks?|bloat|corruption)\b|"
    r"\b(?:healthy|nominal|unaffected|ruled out|within (?:normal )?limits)\b"
    r"|\b(?:is|are|was|were|looks?|looked|appears?|appeared|seems?|seemed|remains?|stayed)"
    r"\s+(?:fine|normal|ok|okay|good|stable)\b"
    r"|\bno (?:sign|signs|evidence|indication|issue|issues|problem|problems|anomal\w*)\b"
    r"|\bnot (?:the|a) (?:cause|problem|issue|culprit)\b"
    r"|\b(?:not|isn't|wasn't|aren't|weren't|never)\s+(?:been\s+)?"
    r"(?:exhausted|full|leaking|saturated|overloaded|failing|degraded|the cause)\b"
    # "does not have a full disk", "is not running out of disk space": a negation with a
    # failure word a few words after it. (A missing timeout or backoff is a failure.)
    r"|(?:\bnot|\bno|n't|\bneither|\bnor)\s+(?:[a-z_-]+\s+){0,3}?(?:full|filled|filling|exhaust\w*|"
    r"running out|ran out|run out|out of (?:disk|space|memory|connections)|leak\w*|saturat\w*|"
    r"lagg\w*|behind|expir\w*|evict\w*|unassigned|offline|down|fail\w*|degrad\w*|slow\w*|"
    r"overload\w*|stuck|blocked|deadlock\w*|throttl\w*|exceed\w*|crash\w*|oom\w*|"
    r"regress\w*|pressure|contention|errors?)\b")
# A clause naming a failure says something is wrong with what it names.
_FAILS = re.compile(
    r"\b(?:full|filled|filling|exhaust\w*|leak\w*|saturat\w*|spik\w*|high|higher|elevated|"
    r"out of|ran out|no space|fail\w*|error\w*|crash\w*|time[ds]?\s?out\w*|evict\w*|"
    r"unassigned|throttl\w*|overload\w*|exceed\w*|degrad\w*|slow\w*|drift\w*|regress\w*|"
    r"oom\w*|killed|stuck|block\w*|refus\w*|reject\w*|broke\w*|lag\w*|backlog\w*|surge\w*|"
    r"starv\w*|deadlock\w*|bloat\w*|corrupt\w*|expir\w*|misconfig\w*|wrong|bad|lost|loss|"
    r"down|unavailable|grow\w*|grew|increas\w*|rising|rose|caus\w*|due to|because|"
    r"pressure|contention|unbounded|retain\w*|thrash\w*|max(?:ed)?\s?out|limit\w*|"
    r"partition\w*|split|outage|incident|issue|problem|fault\w*|unhealthy|invalid|missing|"
    r"never|introduc\w*|exception\w*|dereferenc\w*|null|miss\w*|recomput\w*|stampede|"
    r"could not|couldn't|cannot|can't|unable|stale|empty|tripl\w*|doubl\w*|flush\w*|"
    r"unguarded|mismatch\w*|inconsistent|without|not|behind|too many|429|unwell|struggl\w*|"
    r"affected|impacted|at fault|culprit|responsible|reach\w* 100|at capacity)\b")


@dataclass(frozen=True)
class Segment:
    """A segment of an asserted clause: `text` is what it asserts ("" when all of it
    was denied), `written` the segment as written, and `joined` whether it begins a
    coordinated part (", and disk space issues"), whose subject, if it names none, is
    the one before it."""
    text: str
    written: str
    joined: bool


def clauses(text: str) -> list[tuple[list[Segment], bool]]:
    """Every clause of a conclusion as its segments, and whether it asserts that
    something has failed: a piece that denies a failure is dropped, and so is a cause
    that a negation denies ("not due to ..."); a clause asserts a failure if what is
    left still says something is wrong. Dots inside words ("v2.3.1",
    "min.insync.replicas") are dropped first, so they do not end a clause."""
    out = []
    for clause in _CLAUSES.split(re.sub(r"(?<=\w)\.(?=\w)", "", text.lower())):
        segments = []
        for p, part in enumerate(_PARTS.split(clause)):
            denied = False
            for c, segment in enumerate(_CAUSES.split(part)):
                negates = bool(_NEGATES_CAUSE.search(segment))
                pieces = [] if denied else [
                    piece.strip() for piece in _PIECES.split(segment)
                    if piece.strip() and not _DENIES.search(piece)]
                if segment.strip():
                    segments.append(Segment(" ".join(pieces), segment.strip(),
                                            joined=p > 0 and c == 0))
                denied = negates
        if segments:
            out.append((segments, bool(_FAILS.search(" ".join(s.text for s in segments)))))
    return out


def asserted_segments(text: str) -> list[list[Segment]]:
    """The clauses that assert a failure, as their segments."""
    return [segments for segments, fails in clauses(text) if fails]


def asserted_clauses(text: str) -> list[list[str]]:
    """The asserted clauses, each as the text of its segments that assert something."""
    return [[s.text for s in clause if s.text] for clause in asserted_segments(text)]


def asserted(text: str) -> str:
    """The asserted clauses as one text, the parts of each joined."""
    return " . ".join(" ".join(parts) for parts in asserted_clauses(text))


def _tags(arch: A.Archetype) -> set[str]:
    return {t.lower() for t in arch.tags if len(t) >= 3}


def _words(arch: A.Archetype) -> set[str]:
    text = " ".join([arch.name, *arch.root_cause, *arch.tags]).lower()
    text = re.sub(r"\{[a-z]+\}", " ", text).replace("-", " ")
    return {w for w in _WORD.findall(text) if len(w) > 4 and w not in _STOP} | _tags(arch)


def _vocabularies() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    words = {a.key: _words(a) for a in A.ARCHETYPES}
    others = {key: set().union(*(w for k, w in words.items() if k != key)) for key in words}
    vocabulary = {key: sorted(own - others[key]) for key, own in words.items()}
    defining = {a.key: sorted(_tags(a) - others[a.key]) for a in A.ARCHETYPES}
    return vocabulary, defining


VOCABULARY, TAGS = _vocabularies()


# What asserting each failure mode looks like. Each entry is a set of phrasings; a
# conclusion names the failure mode when its asserted clauses match one of them.
# [^.] keeps a match inside one clause.
_W = r"[^.]{0,%d}"
ASSERTIONS: dict[str, list[str]] = {
    "db_pool_exhausted": [
        r"\b(connection[- ]?pools?|pool of (db |database )?connections)\b" + _W % 40
        + r"\b(exhaust|saturat|full\b|depleted|maxed|ran out|starv|checked out)",
        r"\bpool exhaustion\b",
        r"\b(exhaust\w*|ran out of|no (free|available|idle))\b" + _W % 25 + r"\bconnections?\b",
        r"\bconnections?\b"
        + _W % 40
        + r"\b(leak\w*|never (returned|released|closed)|not (returned|released|closed)|(were|was|"
          r"all) checked out)",
    ],
    "memory_leak_oom": [
        r"\b(memory|heap)\b"
        + _W % 40
        + r"\b(leak\w*|grew|grow\w*|growth|unbounded|climb\w*|exhaust\w*)",
        r"\boom\b|\bout of memory\b",
        r"\b(cache|listeners?|objects?|references?)\b"
        + _W % 40
        + r"\b(retain\w*|never (unregistered|released|freed|removed))",
    ],
    "bad_deploy_regression": [
        r"\b(release|deploy\w*|rollout|version|build)\b"
        + _W % 60
        + r"\b(regress\w*|introduc\w*|broke\w*|bugs?\b|buggy|null|exception|incompatib\w*|changed|"
          r"removed|renamed)",
        r"\bregression\b",
        r"\b(bad|faulty|buggy|broken|defective) (deploy\w*|release|build|version|rollout)",
    ],
    "downstream_timeout_cascade": [
        r"\b(cascad\w*|propagat\w*|knock-on)\b",
        r"\bno timeouts?\b|\bwithout (a |any )?timeouts?\b|\btimeouts? (was |were )?(not|"
        r"never) (configured|set)\b|\bmissing timeouts?\b",
        r"\b(blocked|waiting|hung|stuck)\b"
        + _W % 30
        + r"\b(indefinitely|on (a |the )?(slow|downstream|dependency))|\bvictim\b",
        r"\b(downstream|dependenc\w*|upstream)\b"
        + _W % 50
        + r"\b(slow\w*|timeouts?|timed out|time out|latenc\w*|degrad\w*|unavailable)",
        r"\b(degradation|slowness|latency) (in|of|on)\b" + _W % 30 + r"\bcaus\w*" + _W % 20
        + r"\btime(d)? ?outs?\b",
        r"\bcircuit[- ]breakers? (is |was |were )?(open\w*|tripped)",
        r"\bretr\w*"
        + _W % 60
        + r"\b(tripl\w*|amplif\w*|multipl\w*)"
        + _W % 40
        + r"\b(load|traffic)\b",
    ],
    "cache_stampede": [
        r"\bstampede\b|\bthundering[- ]herd\b|\bdog[- ]?pil\w*",
        r"\b(expir\w*|ttl|evict\w*|flush\w*)\b"
        + _W % 60
        + r"\b(simultaneous\w*|at once|same time|together|all at)",
        r"\bmiss\w*\b" + _W % 40 + r"\b(recomput\w*|regenerat\w*|rebuil\w*|hammer\w*)",
        r"\bcache\b" + _W % 40 + r"\b(flush\w*|cold|emptied|empty|wiped)",
    ],
    "disk_full": [
        r"\b(disks?|volumes?|filesystems?|file system|storage|partitions?|data directory)\b"
        + _W % 40
        + r"\b(full|filled|filling|fills|fill up|exhaust\w*|ran out|out of space|reached 100|"
          r"at 100)",
        r"\bno space left\b|\bout of (disk|space|storage)\b|\bdisk (space )?exhaust\w*|\b(high|"
        r"elevated|critical) disk (usage|utili[sz]ation)\b",
        r"\b(exhaust\w*|filled|full)\b" + _W % 15 + r"\b(disks?|volumes?|storage)\b",
        r"\bwal\b" + _W % 40 + r"\b(accumulat\w*|piled|filled|grew|growth)",
        r"\b(disk|storage) space (issues?|problems?|shortage|pressure)\b"
        r"|\b(low|insufficient|running out of|lack of) (disk|storage) space\b",
    ],
    "replica_lag": [
        r"\breplica(tion|s)?\b" + _W % 40 + r"\b(lag\w*|behind|delay\w*|stale)",
        r"\b(stale|outdated) (reads?|data)\b",
        r"\b(wal|log) replay\b"
        + _W % 30
        + r"\b(block\w*|behind|slow\w*|stall\w*)|\bblock\w*"
        + _W % 20
        + r"\b(wal|log) replay",
    ],
    "cert_expiry": [
        r"\b(certificates?|certs?|tls|ssl)\b" + _W % 40 + r"\b(expir\w*|lapsed|out of date)",
        r"\bexpired (tls |ssl )?(certificate|cert)s?\b",
        r"\brenewal\b" + _W % 30 + r"\b(disabled|failed|never|stopped)",
    ],
    "upstream_rate_limit": [
        r"\brate[- ]?limit\w*"
        + _W % 30
        + r"\b(exceed\w*|hit|breach\w*|reached|throttl\w*)|\b(exceed\w*|hit|breach\w*)"
        + _W % 20
        + r"\brate[- ]?limit",
        r"\b429\b|\btoo many requests\b|\bthrottl\w*",
        r"\bquota\b"
        + _W % 30
        + r"\b(exhaust\w*|exceed\w*|consum\w*|used up)|\bconsum\w*"
        + _W % 30
        + r"\bquota\b",
    ],
    "consumer_lag": [
        r"\bconsumers?\b"
        + _W % 50
        + r"\b(lag\w*|behind|keep (up|pace)|backlog|slow\w*|dropped out|unhealthy)",
        r"\b(consumer|queue|topic|partition) lag\b|\bbacklog\w*\b",
        r"\b(consumption|processing)\b"
        + _W % 40
        + r"\b(below|slower than|behind)"
        + _W % 20
        + r"\b(production|producer|incoming)",
    ],
    "thread_pool_saturation": [
        r"\bthread[- ]?pool\b"
        + _W % 40
        + r"\b(saturat\w*|exhaust\w*|full\b|occupied|consumed|starv\w*|maxed)",
        r"\bthreads?\b" + _W % 30 + r"\b(blocked|stuck|hung)\b",
        r"\bdeadlock\w*\b",
        r"\bconsum\w*" + _W % 10 + r"\b(the )?(thread )?pool\b",
    ],
    "config_drift": [
        r"\b(config\w*|settings?|environment variables?|env vars?|manifest)\b" + _W % 80
        + r"\b(drift\w*|mismatch\w*|inconsisten\w*|missing|absent|stale|not applied|only applied|"
        r"out of sync|old value)",
        r"\b(missing|absent|stale|outdated)\b" + _W % 30
        + r"\b(config\w*|settings?|environment variables?|env vars?)\b",
        r"\bdrift\w*\b",
        r"\b(stale|old) (values?|config\w*|settings?)\b|\bfell back to (a |the )?default",
        r"\bapplied to (only )?(some|part|a subset|two of three|one of)",
        r"\b(absent|missing) from\b"
        + _W % 30
        + r"\b(overlay|manifest|environment|config\w*|canary|replicas?)\b",
    ],
    "search_shard_unassigned": [
        r"\bshards?\b"
        + _W % 40
        + r"\b(unassigned|not assigned|offline|missing|lost|down|unavailable)",
        r"\b(cluster|index)\b" + _W % 20 + r"\b(red|yellow) (status|state|health)",
        r"\b(high|flood[- ]stage) watermark\b",
        r"\bfaster than shard recovery\b|\bshard recovery\b"
        + _W % 40
        + r"\b(slow\w*|behind|incomplete)",
    ],
    "traffic_surge": [
        r"\b(traffic|load|requests?|demand)\b"
        + _W % 40
        + r"\b(surge\w*|spik\w*|exceed\w*|beyond|outgrew)"
        r"|\b(traffic|load|demand) (surge|spike)\w*"
        r"|\b(tripl\w*|doubl\w*|quadrupl\w*)\s+(the )?(traffic|load|requests)\b",
        r"\bunder[- ]?provision\w*|\binsufficient capacity\b|\bexceed\w*"
        + _W % 20
        + r"\bcapacity\b|\bprovisioned for (typical|normal|average)\b|"
          r"\bwithout capacity planning\b",
        r"\bautoscal\w*\b" + _W % 40 + r"\b(slow\w*|lag\w*|late|did not|didn't|failed|not keep)",
    ],
    "cache_node_eviction": [
        r"\b(cache|redis|memcached|node)\b"
        + _W % 40
        + r"\bevict\w*|\bevict\w*"
        + _W % 30
        + r"\b(hot|keys?|entries)\b",
        r"\bmaxmemory\b|\bmemory (limit|pressure)\b"
        + _W % 30
        + r"\b(cache|redis|evict\w*)|\b(cache|redis)\b"
        + _W % 30
        + r"\bmemory (limit|pressure)",
        r"\b(oversized|large|big) key\b|\bworking set\b"
        + _W % 30
        + r"\b(grew|doubl\w*|exceed\w*|larger)",
    ],
    "broker_partition_offline": [
        r"\b(brokers?|partitions?)\b" + _W % 40
        + r"\b(offline|unavailable|went down|leaderless|no leader|without (a )?leader|"
          r"lost leader\w*)",
        r"\b(no|without)\b"
        + _W % 10
        + r"\bin-sync replica|\b(out of|dropped from|left) the isr\b|\bmininsync",
        r"\bleader(ship)?\b"
        + _W % 30
        + r"\b(could not|couldn't|failed to|did not) (move|transfer|elect)",
    ],
}

_ASSERTIONS = {key: [re.compile(p) for p in found] for key, found in ASSERTIONS.items()}


def hits(text: str) -> dict[str, int]:
    """How many of each archetype's phrasings the conclusion asserts."""
    return matches(asserted(text))


def matches(text: str) -> dict[str, int]:
    """How many of each archetype's phrasings occur in text already taken as asserted."""
    return {key: sum(1 for p in found if p.search(text)) for key, found in _ASSERTIONS.items()}


def closest(text: str) -> str | None:
    """The archetype the text asserts most, if it asserts any."""
    found = hits(text)
    best = max(found, key=found.get)
    return best if found[best] else None


def names(archetype: str, text: str) -> bool:
    found = hits(text)
    own = found[archetype]
    return own > 0 and all(h <= own for h in found.values())
