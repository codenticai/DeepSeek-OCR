"""
hllset-cortex HTTP Service — wraps the DeepSeek-OCR pipeline as a microservice.

Endpoints:
  POST /process       — Full pipeline: Tokenize → HLLSet → Gate ∩ → LUT → Materialize
  POST /process/debruijn — Same + De Bruijn ordered reconstruction
  GET  /health        — Health check

Architecture:
  NanoLM Rust API → HTTP POST → this service → HLLSet pipeline → JSON response
"""

import sys
import os
import logging
from pathlib import Path

# Add hllset_cortex to path (works both inside Docker and local dev)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from flask import Flask, request, jsonify

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hllset-cortex-service")

# ── Import hllset_cortex (with graceful fallback) ───────────────────
try:
    import hllset_py
    from hllset_cortex import (
        HLLSetFilter, FilterResult, FilterStats,
        OCRPipeline, PipelineResult, GateInfo,
        default_tokenizer, encoding_tokenizer, debruijn_tokenizer,
    )
    HLLSET_AVAILABLE = True
    logger.info(f"hllset_py loaded: {[x for x in dir(hllset_py) if not x.startswith('_')]}")
except ImportError as e:
    logger.warning(f"hllset_cortex not available: {e}")
    HLLSET_AVAILABLE = False

app = Flask(__name__)

# ── Global pipeline state ───────────────────────────────────────────
_filters: dict[str, HLLSetFilter] = {}  # session_id → filter
_global_lut = None  # Shared LUT for TF accumulation

if HLLSET_AVAILABLE:
    _global_lut = hllset_py.TokenLut()


def get_filter(session_id: str = "default") -> HLLSetFilter:
    """Get or create a filter for a session."""
    if session_id not in _filters:
        _filters[session_id] = HLLSetFilter()
        _filters[session_id].tokenizer = default_tokenizer()
        # LUT is managed internally by HLLSetFilter.
    return _filters[session_id]


# ── Health Check ────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "service": "hllset-cortex",
        "hllset_available": HLLSET_AVAILABLE,
        "active_sessions": len(_filters),
    })


# ── Full Pipeline ───────────────────────────────────────────────────
@app.route('/process', methods=['POST'])
def process():
    """
    Full pipeline: Tokenize → HLLSet → Gate ∩ → LUT → Materialize.

    Request: {
        "text": "encoding IDs or plain text to process",
        "format": "basic" | "debruijn" (default: "basic"),
        "session_id": "optional session for TF accumulation"
    }
    """
    if not HLLSET_AVAILABLE:
        return jsonify({"error": "hllset_cortex not available"}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    fmt = data.get("format", "basic")
    session_id = data.get("session_id", "default")

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        if fmt == "debruijn":
            result = _process_debruijn(text, session_id)
        else:
            result = _process_basic(text, session_id)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _process_basic(text: str, session_id: str) -> dict:
    """Standard TF-ranked materialization (no order)."""
    filt = get_filter(session_id)

    # Set gate from token vocabulary if not already set
    if filt.gate_hllset is None:
        # Build gate from valid tokens (all non-numeric encoding IDs)
        tokens = filt.tokenizer.tokenize_str(text)
        valid = [t.decode() for t in tokens if t.decode().startswith("enc") or t.decode().startswith("tid")]
        if valid:
            filt.gate_hllset = hllset_py.HLLSet.from_tokens(valid)
            logger.info(f"Gate built: {len(valid)} valid tokens")

    result = filt.process_text(text)

    return {
        "tokens": result.token_strings,
        "token_count": len(result.token_strings),
        "stats": {
            "input_tokens": result.stats.input_tokens,
            "hllset_popcount": result.stats.hllset_popcount,
            "gate_popcount": result.stats.gate_popcount,
            "output_tokens": result.stats.output_tokens,
            "filtered": result.stats.hllset_popcount - result.stats.gate_popcount,
        },
        "lut_size": result.lut_size,
    }


def _process_debruijn(text: str, session_id: str) -> dict:
    """De Bruijn ordered reconstruction with START/END markers."""
    # Use boundary-padded bigram tokenizer for De Bruijn
    db_tok = debruijn_tokenizer("<S>", "</S>")
    tokens = db_tok.tokenize_str(text)

    # Build HLLSet + LUT from bigrams
    hllset = hllset_py.HLLSet.from_token_bytes(tokens)
    lut = hllset_py.TokenLut()
    lut.record_all_bytes(tokens)

    # De Bruijn ordered materialization
    ordered = hllset_py.materialize_debruijn(hllset, lut, "<S>", "</S>")

    # Also run through filter for gate filtering
    filt = get_filter(session_id)
    basic_result = filt.process_text(text)

    return {
        "tokens": ordered,
        "token_count": len(ordered),
        "ordered": True,
        "stats": {
            "input_tokens": basic_result.stats.input_tokens,
            "hllset_popcount": basic_result.stats.hllset_popcount,
            "gate_popcount": basic_result.stats.gate_popcount,
            "output_tokens": basic_result.stats.output_tokens,
        },
        "lut_size": lut.len(),
    }


# ── Gate Setup ──────────────────────────────────────────────────────
@app.route('/gate', methods=['POST'])
def set_gate():
    """Set the gate vocabulary for filtering."""
    if not HLLSET_AVAILABLE:
        return jsonify({"error": "hllset_cortex not available"}), 503

    data = request.get_json(silent=True) or {}
    vocab = data.get("vocab", [])
    session_id = data.get("session_id", "default")

    if not vocab:
        return jsonify({"error": "Missing 'vocab' field"}), 400

    filt = get_filter(session_id)
    filt.gate_hllset = hllset_py.HLLSet.from_tokens(vocab)

    return jsonify({
        "status": "ok",
        "vocab_size": len(vocab),
        "gate_popcount": filt.gate_hllset.popcount(),
    })


# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 9092))
    logger.info(f"hllset-cortex service starting on port {port}")
    logger.info(f"hllset_py available: {HLLSET_AVAILABLE}")
    app.run(host="0.0.0.0", port=port, debug=False)
