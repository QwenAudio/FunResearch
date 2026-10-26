from __future__ import annotations

import re
from typing import Any


# Keep this deterministic gate deliberately conservative. English expressions use
# word boundaries so innocent substrings such as "class" do not match "ass".
_ENGLISH_PATTERNS: dict[str, re.Pattern[str]] = {
    "f_word": re.compile(r"\bf+[\W_]*u+[\W_]*c+[\W_]*k+(?:er|ers|ing|ed|s)?\b", re.I),
    "s_word": re.compile(r"\bsh+i+t+(?:ty|ting|ted|s)?\b", re.I),
    "b_word": re.compile(r"\bb+i+t+c+h+(?:es|y)?\b", re.I),
    "a_word": re.compile(r"\bassh+o+l+e+s?\b", re.I),
    "mf_word": re.compile(r"\bmoth(?:er|a)f+[\W_]*u+[\W_]*c+[\W_]*k+(?:er|ers|ing)?\b", re.I),
    "c_word": re.compile(r"\bc+u+n+t+s?\b", re.I),
    "d_word": re.compile(r"\bd+i+c+k+(?:head|heads|s)?\b", re.I),
    "p_word": re.compile(r"\bp+u+s+s+y+(?:cat|cats|s)?\b", re.I),
}

_CHINESE_TERMS: dict[str, tuple[str, ...]] = {
    "cn_insult": (
        "傻逼", "傻比", "煞笔", "妈的", "他妈的", "你妈的", "操你妈", "草你妈",
        "草泥马", "卧槽", "我操", "我艹", "狗日的", "贱人", "婊子", "王八蛋",
    ),
}


def text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: list[str] = []
        for item in value.values():
            texts.extend(text_values(item))
        return texts
    if isinstance(value, (list, tuple)):
        texts = []
        for item in value:
            texts.extend(text_values(item))
        return texts
    return []


def profanity_categories(value: Any) -> list[str]:
    """Return stable categories, never the matched profanity itself."""
    texts = text_values(value)
    categories = {
        category
        for text in texts
        for category, pattern in _ENGLISH_PATTERNS.items()
        if pattern.search(text)
    }
    for text in texts:
        for category, terms in _CHINESE_TERMS.items():
            if any(term in text for term in terms):
                categories.add(category)
    return sorted(categories)


def contains_profanity(value: Any) -> bool:
    return bool(profanity_categories(value))
