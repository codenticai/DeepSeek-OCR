# hllset_cortex/domain.py
"""
Domain configuration for ds-ocr encoding tokenization.

This module defines HOW ds-ocr encoding IDs become tokens for the
HLLSet Algebra pipeline. The actual tokenization is performed by the
standard hllset-dsl Tokenizer (exposed via hllset_py.Tokenizer).

Per STANDARD.md §5.7: "What changes is the token definition — what
string you feed to the hash function."

Architecture:
    ds-ocr vision encoder → encoding IDs (bytes)
      → hllset_py.Tokenizer (standard hllset-dsl pipeline)
        → token stream → MurmurHash3 → HLLSet (32,768-bit)
          → ∩ gate_TF HLLSet → TokenLut (TF) → materialize
            → restored encoding IDs → ds-ocr decoder

The Token-LUT maps encoding_id → hash_position (with monotonic TF).
HLLSets are content-addressed structural fingerprints.
The lattice connects pages → chapters → books via union operations.
"""

from __future__ import annotations

import hllset_py


def default_tokenizer() -> hllset_py.Tokenizer:
    """Standard tokenizer: ASCII word pattern + lowercase + 3-gram encoding.

    The 3-gram encoding (unigrams + bigrams + trigrams) provides structural
    fingerprinting that makes the gate intersection effectively deterministic
    (~10^-5 false positive rate for a 32,768-bit space).
    """
    return (
        hllset_py.Tokenizer.word_pattern()
        .lowercase()
        .ngrams(1, 3)
    )


def encoding_tokenizer() -> hllset_py.Tokenizer:
    """Tokenizer for raw ds-ocr encoding IDs (numeric/alphanumeric identifiers).

    Use when ds-ocr produces encoding IDs rather than natural language text.
    Matches alphanumeric identifiers: lowercase letters, digits, underscore, hyphen.
    """
    import string
    allowed = list((string.ascii_lowercase + string.digits + "_-").encode())
    return (
        hllset_py.Tokenizer()
        .pattern(allowed)
        .lowercase()
        .ngrams(1, 3)
    )


def debruijn_tokenizer(
    start: str = "<S>", end: str = "</S>"
) -> hllset_py.Tokenizer:
    """Tokenizer for De Bruijn ordered reconstruction.

    Uses boundary-padded bigrams (no unigrams/trigrams) so that
    materialize_debruijn() can reconstruct token sequence order
    via Eulerian path traversal through the De Bruijn graph.

    Args:
        start: Start boundary marker (default "<S>")
        end: End boundary marker (default "</S>")

    Returns:
        Tokenizer configured for De Bruijn reconstruction.
    """
    return (
        hllset_py.Tokenizer.word_pattern()
        .lowercase()
        .pad(start.encode(), end.encode())
        .ngrams(2, 2)
    )
