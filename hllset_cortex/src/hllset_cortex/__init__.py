# hllset_cortex
"""
HLLSet Cortex — semantic encoding restoration for DeepSeek-OCR.

A reference implementation for HLLSet Algebra applications per the
hllset-next STANDARD.md. Receives encoding IDs from ds-ocr's vision
encoder, processes them through the HLLSet Algebra pipeline, and
returns restored encoding IDs for the decoder.

Architecture:
    ds-ocr encoding IDs → hllset_py.Tokenizer (standard pipeline)
      → MurmurHash3 → HLLSet (32,768-bit bitmap)
        → ∩ gate_TF HLLSet (decoder vocabulary filter)
          → TokenLut (monotonic TF accumulation)
            → TF-ranked materialization → restored IDs → Decoder

Key properties (IICA):
    - Idempotent: same IDs → same HLLSet, every time
    - Immutable: HLLSets never change once created
    - Content-Addressed: HLLSet key = SHA1 of serialized bytes

Scenario: PDF book scanning
    Each page → HLLSet
    Chapter = ∪ page HLLSets
    Book = ∪ chapter HLLSets
    Commit to temporal pyramid → holographic memory

Dependencies:
    hllset-py (Rust PyO3 bindings: hllset-core + hllset-dsl tokenizer)
    Python 3.10+

Reference docs:
    - docs/STANDARD.md (governing development standard)
    - docs/IICA_PRINCIPLES.md (foundational IICA gate definition)
    - DESIGN.md (this module's design)
"""

from hllset_cortex.domain import default_tokenizer, encoding_tokenizer, debruijn_tokenizer
from hllset_cortex.filter import HLLSetFilter, FilterResult, FilterStats
from hllset_cortex.pipeline import OCRPipeline, PipelineResult, GateInfo

__all__ = [
    # Tokenizer config
    "default_tokenizer",
    "encoding_tokenizer",
    # Filter
    "HLLSetFilter",
    "FilterResult",
    "FilterStats",
    # Pipeline
    "OCRPipeline",
    "PipelineResult",
    "GateInfo",
]
