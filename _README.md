# HLLSet Cortex — Encoding Restoration for DeepSeek-OCR

**HLLSet Cortex** is a reference implementation for HLLSet Algebra applications
built on [hllset-next](https://github.com/SGS_lib/fractal_manifold/hllset-next).
It sits between DeepSeek-OCR's vision encoder and language decoder as a
**black box**: it receives *encoding IDs* from the encoder, processes them
through the HLLSet Algebra pipeline, and returns *restored encoding IDs* for
the decoder.

> DeepSeek-OCR and hllset-cortex are **independent modules**. hllset-cortex
> never sees real tokens — only encoding IDs and their hashes.

```text
ds-OCR Encoder                        ds-OCR Decoder
      │                                      ▲
      │ encoding IDs        restored enc IDs │
      ▼                                      │
╔══════════════════════════════════════════════════╗
║              hllset-cortex (black box)           ║
║  encoding IDs → Tokenizer → HLLSet → gate ∩      ║
║    → TokenLut (TF) → materialize → restored IDs  ║
╚══════════════════════════════════════════════════╝
```

## What it does

The pipeline converts opaque encoding-ID streams into content-addressed
structural fingerprints and filters them against the decoder's vocabulary:

1. **Tokenizer** — standard `hllset-dsl` pipeline (pattern → normalize →
   n-grams → boundary pad) turns encoding IDs into tokens.
2. **HLLSet** — `MurmurHash3` hashes each token into a fixed 32,768-bit
   bitmap (1,024 × 32 registers, 4,116 bytes).
3. **Gate ∩** — intersection with the `gate_TF HLLSet` (built once from the
   decoder's BPE vocabulary) removes invalid bit positions.
4. **TokenLut** — a persistent, monotonic lookup table maps bit positions
   back to encoding IDs, accumulating term frequency (TF) across streams.
5. **Materialize** — TF-ranked disambiguation returns the restored encoding
   IDs for the decoder.

## Package structure

| Path | Role |
| -------- | ------ |
| `src/hllset_cortex/domain.py` | Tokenizer configuration (`default_tokenizer`, `encoding_tokenizer`, `debruijn_tokenizer`) |
| `src/hllset_cortex/filter.py` | `HLLSetFilter`, `FilterResult`, `FilterStats` — HLLSet + gate ∩ + LUT + materialize |
| `src/hllset_cortex/pipeline.py` | `OCRPipeline`, `PipelineResult`, `GateInfo` — end-to-end OCR ↔ BPE token IDs |
| `crates/hllset_py/` | PyO3 Rust bindings: `HLLSet`, `TokenLut`, `Tokenizer`, `materialize` |
| `crates/hllset_py/vendor/` | Vendored Rust core: `hllset-core` (algebra), `hllset-dsl` (tokenizer) |

## Key properties (IICA)

Per `hllset_cortex/docs/STANDARD.md` Part I:

- **Idempotent** — same encoding IDs → same HLLSet, every time
- **Immutable** — HLLSets never change once created
- **Content-Addressed** — HLLSet key = SHA1 of serialized bytes

## Quick start

### Simulated encoding IDs (no GPU)

```python
from hllset_cortex import HLLSetFilter

filt = HLLSetFilter()
result = filt.process_text("enc10253 enc18278 enc50690 enc10325 enc1805 enc6579")

print(f"Restored IDs: {result.token_strings}")
print(f"HLLSet key:  {result.hllset.content_key()[:40]}...")
```

### End-to-end with real DeepSeek-OCR (GPU required)

```python
from hllset_cortex import OCRPipeline

pipeline = OCRPipeline()
pipeline.load_tokenizer("path/to/tokenizer.json")
pipeline.set_gate()               # build gate_TF HLLSet
result = pipeline.process(ocr_text)  # → BPE token IDs for the decoder

print(result.compressed_tokens)
print(result.token_ids)
print(result.reconstructed)
```

## Notebooks

| Notebook | Description |
| ---------- | ------------- |
| `01_ocr_hllset_pipeline_real.ipynb` | Validation pipeline + real DeepSeek-OCR integration |
| `02_caal_cortex_unification.ipynb` | Cortex unification experiments |
| `03_recursive_iica_chain.ipynb` | Recursive IICA chain |
| `04_dg_agent_network.ipynb` | DG agent network |
| `08_holographic_memory.ipynb` | Temporal pyramid: pages → chapters → books |

## Documentation

- [`hllset_cortex/README.md`](hllset_cortex/README.md) — full module reference
- [`hllset_cortex/DESIGN.md`](hllset_cortex/DESIGN.md) — design document
- [`hllset_cortex/docs/STANDARD.md`](hllset_cortex/docs/STANDARD.md) — governing standard
- [`hllset_cortex/docs/IICA_PRINCIPLES.md`](hllset_cortex/docs/IICA_PRINCIPLES.md) — IICA gate definition

---

## Related Project: Guru-EWM (Emerging World Models Platform)

**Production URL:** <https://codenticai.com/nanolm>

A self-hosted, **CPU-first** platform that combines NLP question answering,
OCR document analysis, and clinical-report interpretation behind a single
chat UI. It uses an HLLSet (HyperLogLog-set) lattice for content-addressed
knowledge retrieval and IPFS for durable storage — **no external LLM, no GPU
required**.

> ⚠️ **Not a medical device.** The diagnostic features are a research demo and
> do not replace clinician review.

### Highlights

- **NanoLM NLP** — deterministic retrieval-based Q&A over a 55k+ card knowledge
  corpus (keyword index + IDF-weighted cosine + union-of-occurrences replies).
- **Clinical text reports** — ECG / X-ray / CT / knee-MRI / lab reports matched
  against curated finding cards with numeric reference ranges.
- **OCR** — Tesseract-based document/image text extraction.
- **Vision** — BiomedCLIP zero-shot image classification + synthetic knee-MRI
  fingerprint classifier (CPU-only).
- **IPFS** — content-addressed storage for the knowledge snapshot and ingested
  documents.
- **HLLSet lattice** — ingestion and inclusion queries via `hllset-next` (Rust).

### Services

| Service | Port | Technology | Purpose |
|---|---|---|---|
| `ewm-ui` | 8080 | NiceGUI | Chat UI (NLP / OCR / Diagnose modes) |
| `ewm-gateway` | 8001 | FastAPI | Routing, service catalog, IPFS proxy |
| `nlp-model` | 9095 | FastAPI | NanoLM English Q&A + document ingestion |
| `medical-diagnostic` | 9094 | FastAPI | Text lattice + knee-MRI + BiomedCLIP zero-shot |
| `deepseek-ocr` | 9093 | FastAPI + Tesseract | OCR text extraction |
| `hllset-next` | 9090 | Rust (axum) | HLLSet algebra API (ingest + inclusion) |
| `hllset-cortex` | 9092 | Flask | HLLSet semantic compressor |
| `ipfs` | 5001 / 8081 | Kubo | Content-addressed storage |

### Relationship to hllset_cortex

Guru-EWM deploys its **own** `hllset-cortex` service (Flask, port 9092) as the
HLLSet semantic compressor for the platform. It is a separate deployment from
this repository's `hllset_cortex` module — the two share the same underlying
`hllset-next` algebra, but Guru-EWM wires it into its Docker Compose stack
alongside `hllset-next` (Rust) and `ipfs` (Kubo).

### NanoLM (production)

**NanoLM** is the self-contained lattice language model hosting the guru-ewm
diagnostic model — live at <https://codenticai.com/nanolm>. Per its
specification, NanoLM calls **no external LLM**: it is built on probabilistic
set algebra (HLLSet lattice) rather than transformers, running entirely on CPU
with retrieval + deterministic matching instead of generation.