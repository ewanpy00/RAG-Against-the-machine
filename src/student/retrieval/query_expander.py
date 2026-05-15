from typing import List
import re


# Domain-specific synonyms for vLLM project
DOMAIN_SYNONYMS = {
    # Server / API
    "server": ["api", "endpoint", "service"],
    "api": ["server", "endpoint"],
    "endpoint": ["server", "api"],

    # Configuration
    "configure": ["setup", "config", "setting"],
    "config": ["configure", "configuration", "setting"],
    "setup": ["configure", "config", "install"],

    # Models
    "model": ["llm"],
    "llm": ["model"],

    # Batching
    "batch": ["batching"],
    "batching": ["batch"],

    # Quantization
    "quantization": ["quant", "int8", "fp8"],
    "quant": ["quantization"],

    # Memory
    "memory": ["ram", "vram"],
    "cache": ["caching"],

    # Inference
    "inference": ["generation", "predict"],
    "generation": ["inference", "generate"],
    "generate": ["generation", "produce"],

    # Python / code
    "function": ["method", "def"],
    "method": ["function", "func"],
    "class": ["object"],
    "import": ["importing"],

    # Vocabulary
    "use": ["using", "utilize"],
    "run": ["running", "execute"],
    "install": ["installation", "setup"],

    # Errors
    "error": ["bug", "issue", "problem"]
}


class QueryExpander:
    """Expand queries with synonyms for better BM25 retrieval."""

    def __init__(
        self,
        expand: bool = True,
        max_synonyms_per_word: int = 2,
    ):
        """Initialize expander.

        Args:
            expand: Enable all expansion sources (domain synonyms + WordNet)
            max_synonyms_per_word: Max synonyms to add per query word
        """
        self.use_domain = expand
        self.use_wordnet = expand
        self.max_synonyms = max_synonyms_per_word

        self.wordnet = None
        if self.use_wordnet:
            self._init_wordnet()

    def _init_wordnet(self) -> None:
        """Lazy load WordNet from NLTK."""
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet
            self.wordnet = wordnet
        except ImportError:
            print("Warning: NLTK not installed, WordNet disabled")
            self.use_wordnet = False

    def expand(self, query: str) -> str:
        """Expand query with synonyms.

        Args:
            query: Original query string

        Returns:
            Expanded query with synonyms appended
        """
        if not query.strip():
            return query

        # Tokenize (lowercase, alphanumeric only)
        words = re.findall(r"\b\w+\b", query.lower())

        expanded_terms = list(words)

        for word in words:
            synonyms = self._get_synonyms(word)
            expanded_terms.extend(synonyms[:self.max_synonyms])

        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return " ".join(unique_terms)

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a single word from all sources."""
        synonyms = []

        # Domain-specific
        if self.use_domain and word in DOMAIN_SYNONYMS:
            synonyms.extend(DOMAIN_SYNONYMS[word])

        # WordNet
        if self.use_wordnet and self.wordnet:
            synonyms.extend(self._wordnet_synonyms(word))

        return synonyms

    def _wordnet_synonyms(self, word: str) -> List[str]:
        """Get WordNet synonyms (single-word only)."""
        if self.wordnet is None:
            return []

        synonyms = set()
        for synset in self.wordnet.synsets(word):
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ").lower()

                if name != word and " " not in name and name.isalpha():
                    synonyms.add(name)

        return list(synonyms)
