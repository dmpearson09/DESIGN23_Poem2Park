"""
match.py

Core matching logic for Poem2Park:
- Precompute multi-vector representations for each park (from park_data.py)
- Precompute biome prototype vectors (Option B): mean vector per biome
- For a poem:
  1) Extract + vectorize poem units (nlp.py)
  2) Infer biome scores via cosine similarity to biome prototypes
  3) Convert biome scores -> per-park gate weights
  4) Run similarity-weighted "phrase voting" against park vector sets (with gating)
  5) Return ranked parks + debug/explanation payload

Run via:
  python -m src.main
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import park_data
from .nlp import Language, VectorUnit, extract_units, load_model, vectorize_units


# -----------------------------
# Results data structures
# -----------------------------

@dataclass(frozen=True)
class VoteContribution:
    poem_unit: str
    poem_unit_type: str
    matched_park_unit: str
    similarity: float
    gated_vote: float


@dataclass(frozen=True)
class MatchResult:
    best_park: str
    ranked_parks: List[Tuple[str, float]]                 # [(park, final_score), ...]
    biome_scores: List[Tuple[str, float]]                 # [(biome, score), ...] sorted desc
    gate_weights: List[Tuple[str, float]]                 # [(park, gate_weight), ...] sorted desc
    explanations: Dict[str, List[VoteContribution]]        # per-park contributions for top parks


# -----------------------------
# Similarity helpers
# -----------------------------

def _safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _cosine_sim_matrix_vec(
    mat: np.ndarray,          # (n, d)
    mat_norms: np.ndarray,    # (n,)
    vec: np.ndarray,          # (d,)
    vec_norm: float,
) -> np.ndarray:
    """Cosine similarity between each row of mat and vec."""
    if vec_norm <= 1e-12:
        return np.zeros((mat.shape[0],), dtype=np.float32)

    denom = mat_norms * vec_norm
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    sims = (mat @ vec) / denom
    return sims.astype(np.float32)


def _topk_weighted_mean(pairs: List[Tuple[float, float]], k: int) -> float:
    """
    pairs: [(score, weight), ...]
    Select top-k by score, then return weighted mean.
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
    Builds and holds all precomputed representations needed for matching.

    Usage:
        matcher = ParkMatcher()
        result = matcher.match(poem_text)
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        *,
        biome_alpha: float = 0.6,
        biome_topk: int = 6,
        vote_top2: bool = True,
        top2_factor: float = 0.5,
        vote_similarity_threshold: float = 0.30,
        park_score_normalize: bool = True,
        park_biome_mix_max: float = 0.6,
        park_biome_mix_mean: float = 0.4,
        explain_top_parks: int = 3,
        explain_top_contribs: int = 6,
    ) -> None:
        self.nlp = nlp or load_model()

        # Hyperparameters
        self.biome_alpha = float(biome_alpha)
        self.biome_topk = int(biome_topk)
        self.vote_top2 = bool(vote_top2)
        self.top2_factor = float(top2_factor)
        self.vote_similarity_threshold = float(vote_similarity_threshold)
        self.park_score_normalize = bool(park_score_normalize)
        self.park_biome_mix_max = float(park_biome_mix_max)
        self.park_biome_mix_mean = float(park_biome_mix_mean)
        self.explain_top_parks = int(explain_top_parks)
        self.explain_top_contribs = int(explain_top_contribs)

        # Precomputed stores
        self._park_mats: Dict[str, np.ndarray] = {}
        self._park_norms: Dict[str, np.ndarray] = {}
        self._park_unit_texts: Dict[str, List[str]] = {}

        self._biome_proto_vecs: Dict[str, np.ndarray] = {}
        self._biome_proto_norms: Dict[str, float] = {}

        self._build_park_vectors()
        self._build_biome_prototypes()

    # -----------------------------
    # Build steps
    # -----------------------------

    def _build_park_vectors(self) -> None:
        """Vectorize curated park terms+chunks into a matrix per park."""
        for park in park_data.PARKS:
            terms = park_data.PARK_TERMS[park]
            chunks = park_data.PARK_CHUNKS[park]

            units = _units_from_lists(tokens=terms, chunks=chunks)
            vec_units = vectorize_units(units, self.nlp)

            if not vec_units:
                raise ValueError(
                    f"No vectors produced for park '{park}'. "
                    f"Make sure your spaCy model has vectors (e.g., en_core_web_md)."
                )

            mat = np.vstack([np.asarray(vu.vector, dtype=np.float32) for vu in vec_units])
            norms = np.linalg.norm(mat, axis=1).astype(np.float32)

            self._park_mats[park] = mat
            self._park_norms[park] = norms
            self._park_unit_texts[park] = [vu.text for vu in vec_units]

    def _build_biome_prototypes(self) -> None:
        """Option B: build ONE mean vector prototype per biome."""
        for biome in park_data.BIOMES:
            terms = park_data.BIOME_TERMS[biome]
            chunks = park_data.BIOME_CHUNKS[biome]

            units = _units_from_lists(tokens=terms, chunks=chunks)
            vec_units = vectorize_units(units, self.nlp)

            if not vec_units:
                raise ValueError(
                    f"No vectors produced for biome '{biome}'. "
                    f"Make sure your spaCy model has vectors (e.g., en_core_web_md)."
                )

            mat = np.vstack([np.asarray(vu.vector, dtype=np.float32) for vu in vec_units])
            proto = mat.mean(axis=0).astype(np.float32)

            self._biome_proto_vecs[biome] = proto
            self._biome_proto_norms[biome] = _safe_norm(proto)

    # -----------------------------
    # Runtime: public API
    # -----------------------------

    def match(self, poem_text: str) -> MatchResult:
        poem_vec_units = self._poem_vector_units(poem_text)
        biome_scores = self._infer_biome_scores(poem_vec_units)
        gate_weights = self._compute_gate_weights(biome_scores)
        park_scores, contribs = self._vote_over_parks(poem_vec_units, gate_weights)

        ranked = sorted(park_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_park = ranked[0][0] if ranked else "Unknown"

        biome_sorted = sorted(biome_scores.items(), key=lambda kv: kv[1], reverse=True)
        gate_sorted = sorted(gate_weights.items(), key=lambda kv: kv[1], reverse=True)

        explanations: Dict[str, List[VoteContribution]] = {}
        for park, _score in ranked[: max(1, self.explain_top_parks)]:
            c_list = contribs.get(park, [])
            c_sorted = sorted(c_list, key=lambda c: c.gated_vote, reverse=True)[: max(1, self.explain_top_contribs)]
            explanations[park] = c_sorted

        return MatchResult(
            best_park=best_park,
            ranked_parks=ranked,
            biome_scores=biome_sorted,
            gate_weights=gate_sorted,
            explanations=explanations,
        )

    # -----------------------------
    # Runtime: internal helpers
    # -----------------------------

    def _poem_vector_units(self, poem_text: str) -> List[VectorUnit]:
        units = extract_units(poem_text, self.nlp)
        return vectorize_units(units, self.nlp)

    def _infer_biome_scores(self, poem_units: List[VectorUnit]) -> Dict[str, float]:
        """
        For each poem unit:
          similarity to each biome prototype (mean vector)
        Aggregate per biome using top-k weighted mean.
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

    def _compute_gate_weights(self, biome_scores: Dict[str, float]) -> Dict[str, float]:
        """
        park_biome_score = mix(max, mean) over the park's biome tags
        gate_weight = 1 + alpha * park_biome_score
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

    def _vote_over_parks(
        self,
        poem_units: List[VectorUnit],
        gate_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, List[VoteContribution]]]:
        """
        Similarity-weighted voting:
          - For each poem unit, compute best-match similarity to each park (max over park vectors).
          - Apply unit weight and park gate weight.
          - Award vote to top1 (and optionally top2).
          - Normalize totals (optional).
        """
        tallies: Dict[str, float] = {p: 0.0 for p in park_data.PARKS}
        contributions: Dict[str, List[VoteContribution]] = {p: [] for p in park_data.PARKS}

        total_vote_weight = 0.0

        for u in poem_units:
            vec = np.asarray(u.vector, dtype=np.float32)
            vec_norm = _safe_norm(vec)
            if vec_norm <= 1e-12:
                continue

            park_best: List[Tuple[str, float, str]] = []

            for park in park_data.PARKS:
                mat = self._park_mats[park]
                norms = self._park_norms[park]
                sims = _cosine_sim_matrix_vec(mat, norms, vec, vec_norm)

                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                matched_text = self._park_unit_texts[park][best_idx]
                park_best.append((park, best_sim, matched_text))

            park_best.sort(key=lambda t: t[1], reverse=True)

            if park_best[0][1] < self.vote_similarity_threshold:
                continue

            top1_park, top1_sim, top1_match = park_best[0]
            top_vote = top1_sim * float(u.weight) * float(gate_weights.get(top1_park, 1.0))

            tallies[top1_park] += top_vote
            contributions[top1_park].append(
                VoteContribution(
                    poem_unit=u.text,
                    poem_unit_type=u.unit_type,
                    matched_park_unit=top1_match,
                    similarity=top1_sim,
                    gated_vote=top_vote,
                )
            )

            total_vote_weight += float(u.weight)

            if self.vote_top2 and len(park_best) > 1:
                top2_park, top2_sim, top2_match = park_best[1]
                if top2_sim >= self.vote_similarity_threshold:
                    vote2 = top2_sim * float(u.weight) * float(gate_weights.get(top2_park, 1.0)) * self.top2_factor
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

        if self.park_score_normalize:
            denom = total_vote_weight if total_vote_weight > 1e-12 else 1.0
            for park in tallies:
                tallies[park] = float(tallies[park] / denom)

        return tallies, contributions


# -----------------------------
# Helper: build Units from curated lists (no extraction step)
# -----------------------------

def _units_from_lists(tokens: Sequence[str], chunks: Sequence[str]):
    """
    Create a Units object for vectorize_units without running extract_units.
    """
    from .nlp import Units  # package-safe import

    tokens_clean = tuple(t.strip().lower() for t in tokens if t and t.strip())
    chunks_clean = tuple(c.strip().lower() for c in chunks if c and c.strip())

    debug = {"propn_flags": tuple(False for _ in tokens_clean)}
    return Units(tokens=tokens_clean, chunks=chunks_clean, debug=debug)