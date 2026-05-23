import re


DOMAIN_SYNONYMS = {
    "server": ["api", "endpoint", "service"],
    "api": ["server", "endpoint"],
    "endpoint": ["server", "api", "route"],
    "configure": ["setup", "config", "setting"],
    "config": ["configure", "configuration", "setting"],
    "setup": ["configure", "config", "install"],
    "model": ["llm", "checkpoint"],
    "llm": ["model", "language model"],
    "batch": ["batching"],
    "batching": ["batch", "continuous batching"],
    "quantization": ["quant", "int8", "fp8"],
    "quant": ["quantization"],
    "memory": ["ram", "vram", "gpu memory"],
    "cache": ["caching", "kv cache"],
    "caching": ["cache", "kv cache"],
    "inference": ["generation", "predict"],
    "generation": ["inference", "generate", "decoding"],
    "generate": ["generation", "produce", "decode"],
    "attention": ["self-attention", "mha", "flash attention"],
    "tensor": ["weight", "matrix"],
    "parallel": ["distributed", "multi-gpu"],
    "sampling": ["decoding", "generation strategy"],
    "temperature": ["sampling parameter"],
    "token": ["tokenization", "input id"],
    "tokenize": ["tokenization", "encode"],
    "function": ["method", "def"],
    "method": ["function", "func"],
    "class": ["object", "module"],
    "import": ["importing", "dependency"],
    "error": ["bug", "issue", "exception"],
    "run": ["running", "execute", "launch"],
    "install": ["installation", "setup"],
    "load": ["loading", "initialize", "init"],
    "return": ["output", "result"],
    "parameter": ["argument", "arg", "param", "option"],
    "default": ["default value", "fallback"],
}


class QueryExpander:
    """Expand BM25 queries with domain synonyms to improve recall."""

    def __init__(
        self,
        expand: bool = True,
        max_synonyms_per_word: int = 2,
    ) -> None:
        self.enabled = expand
        self.max_synonyms = max_synonyms_per_word

    def expand(self, query: str) -> str:
        """Return query with synonyms appended."""
        if not self.enabled or not query.strip():
            return query

        words = re.findall(r"\b\w+\b", query.lower())
        extra = []

        for word in words:
            synonyms = DOMAIN_SYNONYMS.get(word, [])
            for syn in synonyms[: self.max_synonyms]:
                if syn not in query.lower() and syn not in extra:
                    extra.append(syn)

        if not extra:
            return query

        return query + " " + " ".join(extra)
