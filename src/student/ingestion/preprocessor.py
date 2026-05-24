"""Text preprocessor for code documents and queries.

Splits snake_case, camelCase, PascalCase, and CONSTANT_CASE identifiers
so BM25 can match query terms against code tokens.

Example:
    >>> preprocess("activation_formats FusedMoEActivationFormat")
    "activation formats FusedMoEActivationFormat fusedmoeactivationformat
     activation_formats Fused Mo EActivation Format"
"""

import re


def _split_camel_case(word: str) -> list[str]:
    """Split camelCase / PascalCase / acronyms into parts.

    Examples:
        FusedMoEActivationFormat -> ['Fused', 'Mo', 'EActivation', 'Format']
        CudaGraph                -> ['Cuda', 'Graph']
        FP8_MIN                  -> ['FP', '8', 'MIN']  (handled by snake split)
    """
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", word)
    parts = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", parts)
    return [p for p in parts.split() if len(p) >= 2]


def _split_snake_case(word: str) -> list[str]:
    """Split snake_case / CONSTANT_CASE into parts."""
    return [p for p in word.split("_") if len(p) >= 2]


def expand_identifiers(text: str) -> str:
    """Return original text with extra space-separated identifier parts appended.

    For each token that looks like an identifier (contains _ or mixed case),
    append its split form so BM25 indexes both the original and the parts.
    """
    tokens = re.findall(r"\b[\w]+\b", text)
    extra: list[str] = []

    for token in tokens:
        parts: list[str] = []

        if "_" in token:
            snake_parts = _split_snake_case(token)
            parts.extend(snake_parts)

        if re.search(r"[A-Z][a-z]|[a-z][A-Z]|[A-Z]{2,}[a-z]", token):
            camel_parts = _split_camel_case(token)
            parts.extend(camel_parts)

        for p in parts:
            pl = p.lower()
            if pl not in text.lower() and pl not in [e.lower() for e in extra]:
                extra.append(p)

    if extra:
        return text + " " + " ".join(extra)
    return text
