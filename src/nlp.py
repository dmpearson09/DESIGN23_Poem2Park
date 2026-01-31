"""
nlp.py

SpaCy-powered text processing utilities for Poem2Park.

Responsibilities:
- Load a spaCy model once.
- Extract "units" from text:
  - tokens: NOUN/PROPN/ADJ/VERB (minus stopwords, punctuation, etc.)
  - chunks: cleaned noun chunks
- Vectorize units into a consistent structure used by match.py.

Design goals:
- Consistent behavior across poem text, park data, and biome data.
- Debug-friendly outputs.
- Safe defaults (filters that reduce noise in poetry).

Requirements:
- spacy
- en_core_web_md (recommended) OR another model with vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Units:
    """Extracted lexical units from a text."""
    tokens: Tuple[str, ...]           # single-word units (lemmas)
    chunks: Tuple[str, ...]           # multi-word noun chunks
    debug: Dict[str, object]          # optional debug payload (kept lightweight)


@dataclass(frozen=True)
class VectorUnit:
    """A unit paired with a vector and a weight for scoring."""
    text: str
    unit_type: str                   # "token" or "chunk"
    vector: Sequence[float]          # spaCy vector (numpy-like)
    weight: float


# -----------------------------
# Configuration defaults
# -----------------------------

DEFAULT_MODEL_NAME = "en_core_web_md"

# POS tags we keep for single-token extraction
ALLOWED_POS: Set[str] = {"NOUN", "PROPN", "ADJ", "VERB"}

# A small list of generic "nature words" that can cause over-matching.
# Keep this short; tune later from real failures.
GENERIC_FILTER: Set[str] = {
    "thing", "things", "place", "places", "world", "earth", "land",
    "day", "night", "time", "way", "light", "dark", "sound", "silence",
    "wind", "sky",
}

# Defaults for weighting in voting.
DEFAULT_TOKEN_WEIGHT = 1.0
DEFAULT_CHUNK_WEIGHT = 1.6
DEFAULT_PROPN_BOOST = 1.25  # applied on top of token weight for proper nouns


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

    # Disable components you don't need for speed (keep parser for noun_chunks).
    # If you later want NER, remove 'ner' from disable list.
    # NOTE: some pipelines may not have all components; spaCy ignores missing ones.
    # If you remove the parser, noun_chunks won't work.
    # We'll keep the parser enabled by default.
    _NLP_SINGLETON = nlp
    return nlp


# -----------------------------
# Extraction helpers
# -----------------------------

def _is_noise_token(t: Token) -> bool:
    """Heuristic noise filters for tokens."""
    if t.is_space or t.is_punct:
        return True
    if t.is_stop:
        return True
    if t.like_num:
        return True
    # strip tokens that are just punctuation or very short (e.g., "’", "-")
    if len(t.text.strip()) <= 1:
        return True
    # ignore URLs / emails
    if t.like_url or t.like_email:
        return True
    return False


def _normalize_token(t: Token) -> str:
    """
    Convert a token to a normalized string for matching.
    - lemma
    - lowercase
    """
    return t.lemma_.lower().strip()


def _clean_chunk_text(chunk_text: str) -> str:
    """
    Normalize noun chunk text to reduce variability.
    - lowercase
    - trim whitespace
    - collapse multiple spaces
    """
    text = " ".join(chunk_text.lower().strip().split())
    return text


def _filter_chunk(doc: Doc, start: int, end: int) -> bool:
    """
    Decide whether to keep a noun chunk based on its content tokens.
    We remove chunks that are mostly stopwords/punct or too short.
    """
    toks = [doc[i] for i in range(start, end)]
    content = [t for t in toks if not _is_noise_token(t)]
    if len(content) == 0:
        return False
    # Keep chunks that have at least 2 characters total
    joined = "".join(t.text for t in content).strip()
    if len(joined) < 3:
        return False
    return True


# -----------------------------
# Public API: extract units
# -----------------------------

def extract_units(
    text: str,
    nlp: Optional[Language] = None,
    *,
    allowed_pos: Set[str] = ALLOWED_POS,
    remove_generic: bool = True,
    max_tokens: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> Units:
    """
    Extract tokens and noun chunks from the input text.

    Returns:
      Units(tokens=..., chunks=..., debug=...)
    """
    if nlp is None:
        nlp = load_model()

    doc = nlp(text)

    tokens_out: List[str] = []
    propn_flags: List[bool] = []  # align with tokens_out for optional debug

    removed_counts = {
        "stop_or_noise": 0,
        "pos_not_allowed": 0,
        "generic_filtered": 0,
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

        if remove_generic and norm in GENERIC_FILTER:
            removed_counts["generic_filtered"] += 1
            continue

        tokens_out.append(norm)
        propn_flags.append(t.pos_ == "PROPN")

        if max_tokens is not None and len(tokens_out) >= max_tokens:
            break

    # --- Noun chunks (multi-word units) ---
    chunks_out: List[str] = []
    # noun_chunks requires parser; if missing, this will raise.
    # We'll catch and degrade gracefully by returning no chunks.
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

    # Deduplicate while preserving order
    tokens_out = _dedupe_keep_order(tokens_out)
    chunks_out = _dedupe_keep_order(chunks_out)

    debug: Dict[str, object] = {
        "removed_counts": removed_counts,
        "token_count": len(tokens_out),
        "chunk_count": len(chunks_out),
        # helpful if you want to analyze proper-noun influence later
        "propn_flags": tuple(propn_flags),
    }

    return Units(tokens=tuple(tokens_out), chunks=tuple(chunks_out), debug=debug)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------------
# Public API: vectorize units
# -----------------------------

def vectorize_units(
    units: Units,
    nlp: Optional[Language] = None,
    *,
    token_weight: float = DEFAULT_TOKEN_WEIGHT,
    chunk_weight: float = DEFAULT_CHUNK_WEIGHT,
    propn_boost: float = DEFAULT_PROPN_BOOST,
) -> List[VectorUnit]:
    """
    Convert extracted Units into VectorUnits with spaCy vectors.

    Important:
    - Requires a model with vectors. If vectors are missing, unit vectors may be all zeros.
    - This function returns only units with a non-zero vector norm.
    """
    if nlp is None:
        nlp = load_model()

    vec_units: List[VectorUnit] = []

    # --- Chunks first (they typically carry strong place imagery) ---
    for ch in units.chunks:
        doc = nlp(ch)
        # doc.vector is a pooled vector for the phrase
        if not _has_nonzero_vector(doc.vector):
            continue
        vec_units.append(
            VectorUnit(
                text=ch,
                unit_type="chunk",
                vector=doc.vector,
                weight=chunk_weight,
            )
        )

    # --- Tokens ---
    # We'll preserve the order of tokens from extraction.
    # Proper-noun boosting: if original extraction flagged PROPN, apply boost.
    propn_flags = units.debug.get("propn_flags", ())
    for idx, tok in enumerate(units.tokens):
        doc = nlp(tok)
        if not _has_nonzero_vector(doc.vector):
            continue

        w = token_weight
        if idx < len(propn_flags) and bool(propn_flags[idx]):
            w *= propn_boost

        vec_units.append(
            VectorUnit(
                text=tok,
                unit_type="token",
                vector=doc.vector,
                weight=w,
            )
        )

    return vec_units


def _has_nonzero_vector(vec) -> bool:
    """
    Return True if vector is not all zeros.
    Works with spaCy vectors (numpy arrays).
    """
    # Avoid importing numpy explicitly; spaCy vectors support sum of squares.
    # We'll compute a simple norm-like check.
    try:
        # vec is numpy.ndarray
        return float((vec * vec).sum()) > 1e-12
    except Exception:
        # Fallback: treat as truthy
        return True