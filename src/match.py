"""
match.py  (streamlined)

What this file does
-------------------
This is the scoring engine for Poem2Park.

Build-time (once):
1) Vectorize each park's curated terms+chunks into a matrix of vectors.
2) Vectorize each biome's curated terms+chunks and compute ONE mean prototype vector per biome.

Runtime (per poem):
1) Extract + vectorize poem units (tokens + noun chunks) using nlp.py.
2) Infer biome scores by comparing each poem unit to each biome prototype (cosine similarity).
3) Convert biome scores into a per-park "gate" multiplier based on the park's biome tags.
4) Phrase voting: each poem unit votes for the park it matches best (and optionally 2nd best),
   weighted by similarity × unit_weight × gate_weight.
5) Return ranked parks + biome scores + gate weights + word-connection explanations.

IMPORTANT compatibility note
----------------------------
This version is compatible with your streamlined nlp.py:
- No GENERIC_FILTER usage
- No duplicate-removal assumptions
- No proper-noun boosting / propn_flags anywhere

If your results become overly sensitive to repeated words (because you removed dedupe),
tune these knobs in __init__:
- vote_similarity_threshold
- vote_top2 / top2_factor
- park_score_normalize
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import park_data
from .nlp import Language, VectorUnit, Units, extract_units, load_model, vectorize_units


# -----------------------------
# Results data structures
# -----------------------------

@dataclass(frozen=True)
class VoteContribution:
    """One explanation item: how a poem unit contributed to a park's score."""
    poem_unit: str
    poem_unit_type: str
    matched_park_unit: str
    similarity: float
    gated_vote: float


@dataclass(frozen=True)
class MatchResult:
    """Final output of matching a poem."""
    best_park: str
    ranked_parks: List[Tuple[str, float]]                  # [(park, final_score), ...]
    biome_scores: List[Tuple[str, float]]                  # [(biome, score), ...] sorted desc
    gate_weights: List[Tuple[str, float]]                  # [(park, gate_weight), ...] sorted desc
    explanations: Dict[str, List[VoteContribution]]         # per-park top contributions


# -----------------------------
# Vector math helpers
# -----------------------------

def _safe_norm(x: np.ndarray) -> float:
    """Robust norm helper."""
    return float(np.linalg.norm(x))


def _cosine_sim_matrix_vec(
    mat: np.ndarray,          # (n, d)
    mat_norms: np.ndarray,    # (n,)
    vec: np.ndarray,          # (d,)
    vec_norm: float,
) -> np.ndarray:
    """
    Cosine similarity between each row of `mat` and a single `vec`.
    Returns an array (n,) of similarities.
    """
    if vec_norm <= 1e-12:
        return np.zeros((mat.shape[0],), dtype=np.float32)

    denom = mat_norms * vec_norm
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    sims = (mat @ vec) / denom
    return sims.astype(np.float32)


def _topk_weighted_mean(pairs: List[Tuple[float, float]], k: int) -> float:
    """
    pairs: [(score, weight), ...]
    Choose top-k by score, then return weighted mean.
    Used to aggregate biome evidence.
    """
    if not pairs:
        return 0.0

    pairs_sorted = sorted(pairs, key=lambda t: t[0], reverse=True)
    chosen = pairs_sorted[: max(1, k)]

    scores = np.array([s for s, _w in chosen], dtype=np.float32)
    weights = np.array([w for _s, w in chosen], dtype=np.float32)

    wsum = float(weights.sum())
    if wsum <= 1e-12:
        return float(scores.mean())

    return float((scores * weights).sum() / wsum)


# -----------------------------
# ParkMatcher
# -----------------------------

class ParkMatcher:
    """
    Holds precomputed park/biome representations and performs matching.

    Typical usage:
        matcher = ParkMatcher()
        result = matcher.match("your poem text...")
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        *,
        # Biome gating strength
        biome_alpha: float = 0.6,

        # Biome aggregation: how many top unit→biome similarities count
        biome_topk: int = 6,

        # Voting behavior
        vote_similarity_threshold: float = 0.30,  # ignore weak unit->park matches
        vote_top2: bool = True,                   # also give partial credit to runner-up park
        top2_factor: float = 0.5,                 # scale for runner-up vote

        # Gate calculation mix for parks that have multiple biome tags
        park_biome_mix_max: float = 0.6,
        park_biome_mix_mean: float = 0.4,

        # Normalization keeps scores comparable across poems with different lengths
        park_score_normalize: bool = True,

        # Explanation size
        explain_top_parks: int = 3,
        explain_top_contribs: int = 6,
    ) -> None:
        self.nlp = nlp or load_model()

        # Hyperparameters
        self.biome_alpha = float(biome_alpha)
        self.biome_topk = int(biome_topk)
        self.vote_similarity_threshold = float(vote_similarity_threshold)
        self.vote_top2 = bool(vote_top2)
        self.top2_factor = float(top2_factor)
        self.park_biome_mix_max = float(park_biome_mix_max)
        self.park_biome_mix_mean = float(park_biome_mix_mean)
        self.park_score_normalize = bool(park_score_normalize)
        self.explain_top_parks = int(explain_top_parks)
        self.explain_top_contribs = int(explain_top_contribs)

        # Precomputed stores:
        # park -> matrix of vectors (rows are park units)
        self._park_mats: Dict[str, np.ndarray] = {}
        self._park_norms: Dict[str, np.ndarray] = {}
        self._park_unit_texts: Dict[str, List[str]] = {}

        # biome -> prototype vector (mean of biome unit vectors)
        self._biome_proto_vecs: Dict[str, np.ndarray] = {}
        self._biome_proto_norms: Dict[str, float] = {}

        # Build all representations once
        self._build_park_vectors()
        self._build_biome_prototypes()

    # -----------------------------
    # Build-time: parks
    # -----------------------------

    def _build_park_vectors(self) -> None:
        """
        Vectorize curated park terms + chunks into a matrix per park.
        This runs once at initialization.
        """
        for park in park_data.PARKS:
            terms = park_data.PARK_TERMS[park]
            chunks = park_data.PARK_CHUNKS[park]

            # Convert lists -> Units without running spaCy parsing.
            # (They are already curated phrases; we just normalize case/whitespace.)
            units = _units_from_lists(tokens=terms, chunks=chunks)

            # Vectorize into VectorUnits with weights (token_weight/chunk_weight from nlp.py)
            vec_units = vectorize_units(units, self.nlp)

            if not vec_units:
                raise ValueError(
                    f"No vectors produced for park '{park}'. "
                    "Make sure your spaCy model has vectors (e.g., en_core_web_md)."
                )

            # Stack vectors into (n_units, d) matrix for fast similarity lookups
            mat = np.vstack([np.asarray(vu.vector, dtype=np.float32) for vu in vec_units])
            norms = np.linalg.norm(mat, axis=1).astype(np.float32)

            self._park_mats[park] = mat
            self._park_norms[park] = norms

            # Store the original unit text for explanations (index aligns with matrix rows)
            self._park_unit_texts[park] = [vu.text for vu in vec_units]

    # -----------------------------
    # Build-time: biomes
    # -----------------------------

    def _build_biome_prototypes(self) -> None:
        """
        Build ONE mean vector prototype per biome (Option B).
        This runs once at initialization.
        """
        for biome in park_data.BIOMES:
            terms = park_data.BIOME_TERMS[biome]
            chunks = park_data.BIOME_CHUNKS[biome]

            units = _units_from_lists(tokens=terms, chunks=chunks)
            vec_units = vectorize_units(units, self.nlp)

            if not vec_units:
                raise ValueError(
                    f"No vectors produced for biome '{biome}'. "
                    "Make sure your spaCy model has vectors (e.g., en_core_web_md)."
                )

            mat = np.vstack([np.asarray(vu.vector, dtype=np.float32) for vu in vec_units])

            # Prototype = mean vector over the biome's curated units
            proto = mat.mean(axis=0).astype(np.float32)

            self._biome_proto_vecs[biome] = proto
            self._biome_proto_norms[biome] = _safe_norm(proto)

    # -----------------------------
    # Runtime: public API
    # -----------------------------

    def match(self, poem_text: str) -> MatchResult:
        """
        Run the full pipeline and return ranked parks + explanations.
        """
        poem_vec_units = self._poem_vector_units(poem_text)

        biome_scores = self._infer_biome_scores(poem_vec_units)
        gate_weights = self._compute_gate_weights(biome_scores)

        park_scores, contribs = self._vote_over_parks(poem_vec_units, gate_weights)

        ranked = sorted(park_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_park = ranked[0][0] if ranked else "Unknown"

        biome_sorted = sorted(biome_scores.items(), key=lambda kv: kv[1], reverse=True)
        gate_sorted = sorted(gate_weights.items(), key=lambda kv: kv[1], reverse=True)

        # Keep only top contribution lines per top parks for explainability/UI.
        explanations: Dict[str, List[VoteContribution]] = {}
        for park, _score in ranked[: max(1, self.explain_top_parks)]:
            c_list = contribs.get(park, [])
            c_sorted = sorted(c_list, key=lambda c: c.gated_vote, reverse=True)
            explanations[park] = c_sorted[: max(1, self.explain_top_contribs)]

        return MatchResult(
            best_park=best_park,
            ranked_parks=ranked,
            biome_scores=biome_sorted,
            gate_weights=gate_sorted,
            explanations=explanations,
        )

    # -----------------------------
    # Runtime: poem -> vector units
    # -----------------------------

    def _poem_vector_units(self, poem_text: str) -> List[VectorUnit]:
        """
        Convert raw poem text -> vector units (tokens + chunks).
        Uses extract_units + vectorize_units from nlp.py.
        """
        units = extract_units(poem_text, self.nlp)
        return vectorize_units(units, self.nlp)

    # -----------------------------
    # Runtime: biome inference
    # -----------------------------

    def _infer_biome_scores(self, poem_units: List[VectorUnit]) -> Dict[str, float]:
        """
        For each poem unit, measure cosine similarity to each biome prototype.
        Aggregate evidence per biome using top-k weighted mean.

        Output: biome -> score in [-1, 1]
        """
        per_biome_pairs: Dict[str, List[Tuple[float, float]]] = {b: [] for b in park_data.BIOMES}

        for u in poem_units:
            v = np.asarray(u.vector, dtype=np.float32)
            v_norm = _safe_norm(v)
            if v_norm <= 1e-12:
                continue

            for biome in park_data.BIOMES:
                proto = self._biome_proto_vecs[biome]
                proto_norm = self._biome_proto_norms[biome]
                if proto_norm <= 1e-12:
                    continue

                sim = float(np.dot(v, proto) / (v_norm * proto_norm))
                per_biome_pairs[biome].append((sim, float(u.weight)))

        biome_scores: Dict[str, float] = {}
        for biome, pairs in per_biome_pairs.items():
            score = _topk_weighted_mean(pairs, k=self.biome_topk)
            biome_scores[biome] = float(max(-1.0, min(1.0, score)))

        return biome_scores

    # -----------------------------
    # Runtime: biome -> per-park gate weights
    # -----------------------------

    def _compute_gate_weights(self, biome_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Convert biome scores into a per-park multiplier.

        Park can have multiple biome tags. We mix:
          park_biome_score = mix_max * max(tag_scores) + mix_mean * mean(tag_scores)

        Then compute:
          gate_weight = 1 + biome_alpha * park_biome_score

        gate_weight is clamped >= 0.0.
        """
        gate_weights: Dict[str, float] = {}

        for park in park_data.PARKS:
            tags = park_data.PARK_BIOMES[park]
            scores = [float(biome_scores.get(t, 0.0)) for t in tags]

            if not scores:
                park_biome_score = 0.0
            else:
                mx = max(scores)
                mn = float(sum(scores) / len(scores))
                park_biome_score = self.park_biome_mix_max * mx + self.park_biome_mix_mean * mn

            gate = 1.0 + self.biome_alpha * park_biome_score
            gate_weights[park] = max(0.0, float(gate))

        return gate_weights

    # -----------------------------
    # Runtime: phrase voting over parks
    # -----------------------------

    def _vote_over_parks(
        self,
        poem_units: List[VectorUnit],
        gate_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, List[VoteContribution]]]:
        """
        Similarity-weighted voting:

        For each poem unit:
          - For each park, find the best similarity (max over park's unit vectors)
          - Award vote to best park (and optionally second best)
          - vote = similarity * unit_weight * gate_weight
        """
        tallies: Dict[str, float] = {p: 0.0 for p in park_data.PARKS}
        contributions: Dict[str, List[VoteContribution]] = {p: [] for p in park_data.PARKS}

        # Used only for optional normalization
        total_vote_weight = 0.0

        for u in poem_units:
            vec = np.asarray(u.vector, dtype=np.float32)
            vec_norm = _safe_norm(vec)
            if vec_norm <= 1e-12:
                continue

            # For this poem unit, compute its best match in each park.
            park_best: List[Tuple[str, float, str]] = []

            for park in park_data.PARKS:
                mat = self._park_mats[park]
                norms = self._park_norms[park]

                sims = _cosine_sim_matrix_vec(mat, norms, vec, vec_norm)

                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                matched_text = self._park_unit_texts[park][best_idx]

                park_best.append((park, best_sim, matched_text))

            # Sort parks by how well they match this poem unit.
            park_best.sort(key=lambda t: t[1], reverse=True)

            # Skip weak evidence: if the best park match is too low, ignore this unit.
            if park_best[0][1] < self.vote_similarity_threshold:
                continue

            # Top-1 vote
            top1_park, top1_sim, top1_match = park_best[0]
            vote1 = top1_sim * float(u.weight) * float(gate_weights.get(top1_park, 1.0))

            tallies[top1_park] += vote1
            contributions[top1_park].append(
                VoteContribution(
                    poem_unit=u.text,
                    poem_unit_type=u.unit_type,
                    matched_park_unit=top1_match,
                    similarity=top1_sim,
                    gated_vote=vote1,
                )
            )

            total_vote_weight += float(u.weight)

            # Optional runner-up vote (helps if a unit could reasonably indicate two parks)
            if self.vote_top2 and len(park_best) > 1:
                top2_park, top2_sim, top2_match = park_best[1]
                if top2_sim >= self.vote_similarity_threshold:
                    vote2 = (
                        top2_sim
                        * float(u.weight)
                        * float(gate_weights.get(top2_park, 1.0))
                        * self.top2_factor
                    )
                    tallies[top2_park] += vote2
                    contributions[top2_park].append(
                        VoteContribution(
                            poem_unit=u.text,
                            poem_unit_type=u.unit_type,
                            matched_park_unit=top2_match,
                            similarity=top2_sim,
                            gated_vote=vote2,
                        )
                    )

        # Optional normalization makes scores less sensitive to poem length.
        if self.park_score_normalize:
            denom = total_vote_weight if total_vote_weight > 1e-12 else 1.0
            for park in tallies:
                tallies[park] = float(tallies[park] / denom)

        return tallies, contributions


# -----------------------------
# Helper: build Units from curated lists (no spaCy parsing)
# -----------------------------

def _units_from_lists(tokens: Sequence[str], chunks: Sequence[str]) -> Units:
    """
    Create a Units object from curated lists without calling extract_units().

    We normalize:
      - lowercase
      - strip whitespace
      - drop empty strings

    Note: We intentionally do NOT dedupe here, matching your new extraction behavior.
    """
    tokens_clean = tuple(t.strip().lower() for t in tokens if t and t.strip())
    chunks_clean = tuple(c.strip().lower() for c in chunks if c and c.strip())

    debug: Dict[str, object] = {}
    return Units(tokens=tokens_clean, chunks=chunks_clean, debug=debug)