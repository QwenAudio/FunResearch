from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def token_jaccard(left: str, right: str) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


class LexicalDedupIndex:
    """Incremental exact-equivalent implementation of lexical duplicate checks.

    SequenceMatcher's real_quick_ratio and quick_ratio are documented upper
    bounds.  They can therefore skip expensive full ratios without changing
    which pair crosses the configured threshold.
    """

    def __init__(self, texts: list[str] | None = None) -> None:
        self.texts: list[str] = []
        self.normalized: list[str] = []
        self.token_sets: list[set[str]] = []
        for text in texts or []:
            self.add(text)

    def add(self, text: str) -> None:
        normalized = normalize_text(text)
        self.texts.append(text)
        self.normalized.append(normalized)
        self.token_sets.append(set(normalized.split()))

    def find_duplicate(
        self,
        candidate: str,
        sequence_threshold: float,
        jaccard_threshold: float,
    ) -> tuple[bool, int | None, dict[str, float]]:
        candidate_normalized = normalize_text(candidate)
        candidate_tokens = set(candidate_normalized.split())
        for index, (other_normalized, other_tokens) in enumerate(
            zip(self.normalized, self.token_sets)
        ):
            if not candidate_tokens and not other_tokens:
                jaccard = 1.0
            elif not candidate_tokens or not other_tokens:
                jaccard = 0.0
            else:
                jaccard = len(candidate_tokens & other_tokens) / len(
                    candidate_tokens | other_tokens
                )

            matcher = SequenceMatcher(
                None, candidate_normalized, other_normalized
            )
            if jaccard >= jaccard_threshold:
                sequence = matcher.ratio()
                return True, index, {
                    "sequence": sequence,
                    "jaccard": jaccard,
                }
            if matcher.real_quick_ratio() < sequence_threshold:
                continue
            if matcher.quick_ratio() < sequence_threshold:
                continue
            sequence = matcher.ratio()
            if sequence >= sequence_threshold:
                return True, index, {
                    "sequence": sequence,
                    "jaccard": jaccard,
                }
        return False, None, {"sequence": 0.0, "jaccard": 0.0}


def lexical_duplicate(
    candidate: str,
    accepted: list[str],
    sequence_threshold: float,
    jaccard_threshold: float,
) -> tuple[bool, int | None, dict[str, float]]:
    return LexicalDedupIndex(accepted).find_duplicate(
        candidate, sequence_threshold, jaccard_threshold
    )
