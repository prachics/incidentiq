"""Matching a conclusion's words against a vocabulary, in any form of each word."""

from __future__ import annotations

import os
import re

_WORD = re.compile(r"[a-z0-9]+")


def keyword_hits(keywords: list[str], text: str) -> int:
    """How many keywords the text uses, as whole words in any form: "exhaustion"
    counts for "exhausted", "connection pool" for "connection-pools". Matching
    literally failed a correct answer ("connection pool exhaustion" matched 1 of 16
    terms); matching inside words passed wrong ones ("oom" in "room", "wal" in
    "walked"), so a keyword's words must each be a word of the text, in order."""
    words = _WORD.findall(text.lower().replace("-", " "))
    return sum(1 for keyword in keywords if _uses(words, _WORD.findall(keyword.lower())))


def _uses(words: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return bool(phrase) and any(
        all(_same_word(p, w) for p, w in zip(phrase, words[i:i + n], strict=True))
        for i in range(len(words) - n + 1))


def _same_word(a: str, b: str) -> bool:
    """One word, or two forms of it: a plural ("pool", "pools"), or at least five
    letters in common at the start and no more than three of either left over
    (saturated, saturation)."""
    if a == b or b in (a + "s", a + "es") or a in (b + "s", b + "es"):
        return True
    n = len(os.path.commonprefix([a, b]))
    return n >= 5 and len(a) - n <= 3 and len(b) - n <= 3
