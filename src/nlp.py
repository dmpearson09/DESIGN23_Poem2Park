"""
nlp.py

SpaCy-Powered Text Processing for Poem2Park.

Main Logic Flow:
- Load a spaCy model
- Extract "units" from text:
  - Tokens: NOUN/PROPN/ADJ/VERB (single word)
  - Chunks: Noun Chunks (multi-word)
  - Clean and Filter tokens and chunks
- Vectorize units
"""

# -----------------------------
# Imports
# -----------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Units:
    """Extracted units from a text."""
    tokens: Tuple[str, ...]           # single-word tokens
    chunks: Tuple[str, ...]           # multi-word chunks
    debug: Dict[str, object]          # Debug info


@dataclass(frozen=True)
class VectorUnit:
    """A unit paired with a vector."""
    text: str                        # extracted text
    unit_type: str                   # token or chunk
    vector: Sequence[float]          # spaCy vector


# -----------------------------
# Configuration defaults
# -----------------------------

DEFAULT_MODEL_NAME = "en_core_web_md"

# POS tags for token extraction
ALLOWED_POS: Set[str] = {"NOUN", "PROPN", "ADJ", "VERB"}


# -----------------------------
# Model loading
# -----------------------------

_NLP_SINGLETON: Optional[Language] = None


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> Language:
    """
    Load and cache a spaCy model.
    Raises a helpful error if the model isn't installed.
    """
    global _NLP_SINGLETON
    if _NLP_SINGLETON is not None:
        return _NLP_SINGLETON

    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise OSError(
            f"Could not load spaCy model '{model_name}'. "
            f"Install it with: python -m spacy download {model_name}"
        ) from e

    _NLP_SINGLETON = nlp
    return nlp


# -----------------------------
# Extraction Helping Functions
# -----------------------------

def _is_noise_token(t: Token) -> bool:
    """Heuristic noise filters for tokens."""
    if t.is_space or t.is_punct:
        return True
    if t.is_stop:
        return True
    if t.like_num:
        return True
    if len(t.text.strip()) <= 1:
        return True
    if t.like_url or t.like_email:
        return True
    return False


def _normalize_token(t: Token) -> str:
    """Normalize a token string for matching (lemma + lowercase)."""
    return t.lemma_.lower().strip()


def _clean_chunk_text(chunk_text: str) -> str:
    """Normalize noun chunk text (lowercase + collapse spaces)."""
    return " ".join(chunk_text.lower().strip().split())


def _filter_chunk(doc: Doc, start: int, end: int) -> bool:
    """
    Keep a noun chunk if it contains at least one non-noise token
    and has a minimal amount of content.
    """
    toks = [doc[i] for i in range(start, end)]
    content = [t for t in toks if not _is_noise_token(t)]
    if not content:
        return False
    joined = "".join(t.text for t in content).strip()
    return len(joined) >= 3


# -----------------------------
# Public API: extract units
# -----------------------------

def extract_units(
    text: str,
    nlp: Optional[Language] = None,
    *,
    allowed_pos: Set[str] = ALLOWED_POS,
    max_tokens: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> Units:
    """
    Extract tokens and noun chunks from the input text.

    Differences from earlier version:
    - No generic filter
    - No deduplication
    - No PROPN flags
    """
    if nlp is None:
        nlp = load_model()

    doc = nlp(text)

    tokens_out: List[str] = []
    removed_counts = {
        "stop_or_noise": 0,
        "pos_not_allowed": 0,
        "empty_lemma": 0,
    }

    # --- Tokens (single word units) ---
    for t in doc:
        if _is_noise_token(t):
            removed_counts["stop_or_noise"] += 1
            continue
        if t.pos_ not in allowed_pos:
            removed_counts["pos_not_allowed"] += 1
            continue

        norm = _normalize_token(t)
        if not norm:
            removed_counts["empty_lemma"] += 1
            continue

        tokens_out.append(norm)

        if max_tokens is not None and len(tokens_out) >= max_tokens:
            break

    # --- Noun chunks (multi-word units) ---
    chunks_out: List[str] = []
    try:
        for chunk in doc.noun_chunks:
            if not _filter_chunk(doc, chunk.start, chunk.end):
                continue

            cleaned = _clean_chunk_text(chunk.text)

            # enforce multi-word chunk
            if " " not in cleaned:
                continue

            chunks_out.append(cleaned)

            if max_chunks is not None and len(chunks_out) >= max_chunks:
                break
    except Exception:
        chunks_out = []

    debug: Dict[str, object] = {
        "removed_counts": removed_counts,
        "token_count": len(tokens_out),
        "chunk_count": len(chunks_out),
    }

    return Units(tokens=tuple(tokens_out), chunks=tuple(chunks_out), debug=debug)


# -----------------------------
# Public API: vectorize units
# -----------------------------

def vectorize_units(
    units: Units,
    nlp: Optional[Language] = None,
) -> List[VectorUnit]:
    """
    Convert extracted Units into VectorUnits with spaCy vectors.
    """
    if nlp is None:
        nlp = load_model()

    vec_units: List[VectorUnit] = []

    # --- Chunks ---
    for ch in units.chunks:
        doc = nlp(ch)
        if not _has_nonzero_vector(doc.vector):
            continue
        vec_units.append(
            VectorUnit(
                text=ch,
                unit_type="chunk",
                vector=doc.vector,
            )
        )

    # --- Tokens ---
    for tok in units.tokens:
        doc = nlp(tok)
        if not _has_nonzero_vector(doc.vector):
            continue
        vec_units.append(
            VectorUnit(
                text=tok,
                unit_type="token",
                vector=doc.vector,
            )
        )

    return vec_units


def _has_nonzero_vector(vec) -> bool:
    """Return True if vector is not all zeros."""
    try:
        return float((vec * vec).sum()) > 1e-12
    except Exception:
        return True