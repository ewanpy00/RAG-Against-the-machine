import re


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase / ABCDef to snake_case."""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s).lower()


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
        """Return query with synonyms and CamelCase expansions appended."""
        if not self.enabled or not query.strip():
            return query

        extra: list[str] = []
        query_lower = query.lower()

        # Domain synonym expansion
        for word in re.findall(r"\b\w+\b", query_lower):
            for syn in DOMAIN_SYNONYMS.get(word, [])[: self.max_synonyms]:
                if syn not in query_lower and syn not in extra:
                    extra.append(syn)

        # CamelCase → snake_case so "ModelRunner" matches "model_runner" in
        # file-path tokens produced by bm25s (e.g. from "model_runner.py").
        for camel in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", query):
            snake = _camel_to_snake(camel)
            if "_" in snake and snake not in query_lower and snake not in extra:
                extra.append(snake)

        return (query + " " + " ".join(extra)) if extra else query
