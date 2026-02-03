"""
match.py

Logic:
1) Extract + vectorize poem units (nlp.py)
2) Infer biome scores using cosine similarity to biome mean prototypes
   + build biome "word connections" by best-matching poem units to biome unit sets
3) Choose TOP biome
4) Match poem ONLY against parks tagged with the top biome
5) Return:
   - best_park
   - ranked_parks (only parks in the top biome)
   - biome_scores (sorted)
   - explanations (word connections for top park and biome)
"""


# -----------------------------
# Imports
# -----------------------------

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
    """Explanation for park voting."""
    poem_unit: str               # poem text
    poem_unit_type: str          # token or chunk
    matched_park_unit: str       # park text
    similarity: float            # cosine similarity
    vote: float                  # vote value


@dataclass(frozen=True)
class BiomeContribution:
    """Explanation for biome selection."""
    poem_unit: str               # poem text
    poem_unit_type: str          # "token" or "chunk"
    matched_biome_unit: str      # biome text
    similarity: float            # cosine similarity


@dataclass(frozen=True)
class MatchResult:
    best_park: str
    ranked_parks: List[Tuple[str, float]]                    # only parks in top biome
    biome_scores: List[Tuple[str, float]]                    # all biomes, sorted desc
    explanations: Dict[str, List[VoteContribution]]          # best park only
    biome_explanations: Dict[str, List[BiomeContribution]]   # top biome only


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


def _topk_mean(scores: List[float], k: int) -> float:
    """Mean of the top-k scores (unweighted)."""
    if not scores:
        return 0.0
    scores_sorted = sorted(scores, reverse=True)
    chosen = scores_sorted[: max(1, k)]
    return float(np.mean(np.array(chosen, dtype=np.float32)))


# -----------------------------
# ParkMatcher (Main Class)
# -----------------------------

class ParkMatcher:
    """
    Hard-biome-filter matcher.

    Usage:
        matcher = ParkMatcher()
        result = matcher.match(poem_text)
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        *,
        # Top-k mean for biome scoring
        biome_topk: int = 5,

        # Poem unit selection behavior
        vote_similarity_threshold: float = 0.35,  # min similarity to award vote
        vote_top2: bool = True,                   # also give partial credit to runner-up park
        top2_factor: float = 0.25,                # scale for runner-up vote

        # Explanation size control
        explain_top_contribs: int = 8,
    ) -> None:
        self.nlp = nlp or load_model()

        # Hyperparameters
        self.biome_topk = int(biome_topk)
        self.vote_similarity_threshold = float(vote_similarity_threshold)
        self.vote_top2 = bool(vote_top2)
        self.top2_factor = float(top2_factor)
        self.explain_top_contribs = int(explain_top_contribs)

        # Precomputed stores (parks)
        self._park_mats: Dict[str, np.ndarray] = {}
        self._park_norms: Dict[str, np.ndarray] = {}
        self._park_unit_texts: Dict[str, List[str]] = {}

        # Precomputed stores (biomes)
        self._biome_unit_mats: Dict[str, np.ndarray] = {}
        self._biome_unit_norms: Dict[str, np.ndarray] = {}
        self._biome_unit_texts: Dict[str, List[str]] = {}
        self._biome_proto_vecs: Dict[str, np.ndarray] = {}
        self._biome_proto_norms: Dict[str, float] = {}

        # Reverse index: biome -> parks (built from PARK_BIOMES)
        self._biome_to_parks: Dict[str, List[str]] = {b: [] for b in park_data.BIOMES}

        self._build_reverse_biome_index()
        self._build_park_vectors()
        self._build_biome_vectors_and_prototypes()

    # -----------------------------
    # Build steps
    # -----------------------------

    def _build_reverse_biome_index(self) -> None:
        """Finds what parks belong to each biome."""
        for park in park_data.PARKS:
            for biome in park_data.PARK_BIOMES.get(park, []):
                if biome in self._biome_to_parks:
                    self._biome_to_parks[biome].append(park)

        # Ensure stable ordering (matches park_data.PARKS order)
        for biome in self._biome_to_parks:
            ordered = [p for p in park_data.PARKS if p in set(self._biome_to_parks[biome])]
            self._biome_to_parks[biome] = ordered

    def _build_park_vectors(self) -> None:
        """Vectorize park units into a matrix per park."""
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

    def _build_biome_vectors_and_prototypes(self) -> None:
        """
        Build:
          - biome unit matrices (for biome word connections)
          - biome mean prototype vectors (for biome scoring)
        """
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
            norms = np.linalg.norm(mat, axis=1).astype(np.float32)
            texts = [vu.text for vu in vec_units]

            self._biome_unit_mats[biome] = mat
            self._biome_unit_norms[biome] = norms
            self._biome_unit_texts[biome] = texts

            proto = mat.mean(axis=0).astype(np.float32)
            self._biome_proto_vecs[biome] = proto
            self._biome_proto_norms[biome] = _safe_norm(proto)

    # -----------------------------
    # Runtime: public API
    # -----------------------------

    def match(self, poem_text: str) -> MatchResult:
        # Extract and vectorize poem units
        poem_units = self._poem_vector_units(poem_text)

        # Infer biomes
        biome_scores, biome_contribs = self._infer_biome_scores_and_connections(poem_units)
        biome_sorted = sorted(biome_scores.items(), key=lambda kv: kv[1], reverse=True)

        # Choose top biome (hard filter)
        top_biome = biome_sorted[0][0] if biome_sorted else None
        candidate_parks = self._biome_to_parks.get(top_biome, []) if top_biome else []

        # Vote only over candidate parks
        park_scores, park_contribs = self._vote_over_candidate_parks(poem_units, candidate_parks)
        ranked_parks = sorted(park_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_park = ranked_parks[0][0] if ranked_parks else "Unknown"

        # Top Biome and Park Explanations
        explanations: Dict[str, List[VoteContribution]] = {}
        if ranked_parks and best_park != "Unknown":
            c_list = park_contribs.get(best_park, [])
            c_sorted = sorted(c_list, key=lambda c: c.vote, reverse=True)[: max(1, self.explain_top_contribs)]
            explanations[best_park] = c_sorted

        biome_explanations: Dict[str, List[BiomeContribution]] = {}
        if top_biome:
            c_list = biome_contribs.get(top_biome, [])
            c_sorted = sorted(c_list, key=lambda c: c.similarity, reverse=True)[: max(1, self.explain_top_contribs)]
            biome_explanations[top_biome] = c_sorted

        return MatchResult(
            best_park=best_park,
            ranked_parks=ranked_parks,
            biome_scores=biome_sorted,
            explanations=explanations,
            biome_explanations=biome_explanations,
        )

    # -----------------------------
    # Runtime: internal helpers
    # -----------------------------

    def _poem_vector_units(self, poem_text: str) -> List[VectorUnit]:
        units = extract_units(poem_text, self.nlp)
        return vectorize_units(units, self.nlp)

    def _infer_biome_scores_and_connections(
        self, poem_units: List[VectorUnit]
    ) -> Tuple[Dict[str, float], Dict[str, List[BiomeContribution]]]:
        """
        Biome scoring:
          - similarity of each poem unit to biome prototype (mean vector)
          - aggregate per biome using top-k mean

        Biome connections:
          - for each poem unit and biome, find best-matching biome unit
        """
        per_biome_sims: Dict[str, List[float]] = {b: [] for b in park_data.BIOMES}
        biome_contribs: Dict[str, List[BiomeContribution]] = {b: [] for b in park_data.BIOMES}

        for u in poem_units:
            v = np.asarray(u.vector, dtype=np.float32)
            v_norm = _safe_norm(v)
            if v_norm <= 1e-12:
                continue

            for biome in park_data.BIOMES:
                # Score vs mean prototype
                proto = self._biome_proto_vecs[biome]
                proto_norm = self._biome_proto_norms[biome]
                if proto_norm <= 1e-12:
                    continue

                sim_proto = float(np.dot(v, proto) / (v_norm * proto_norm))
                per_biome_sims[biome].append(sim_proto)

                # Connection: best match vs biome unit set (for explanations)
                mat = self._biome_unit_mats[biome]
                norms = self._biome_unit_norms[biome]
                sims = _cosine_sim_matrix_vec(mat, norms, v, v_norm)

                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                matched_text = self._biome_unit_texts[biome][best_idx]

                biome_contribs[biome].append(
                    BiomeContribution(
                        poem_unit=u.text,
                        poem_unit_type=u.unit_type,
                        matched_biome_unit=matched_text,
                        similarity=best_sim,
                    )
                )

        biome_scores: Dict[str, float] = {}
        for biome, sims in per_biome_sims.items():
            score = _topk_mean(sims, k=self.biome_topk)
            biome_scores[biome] = float(max(-1.0, min(1.0, score)))

        return biome_scores, biome_contribs

    def _vote_over_candidate_parks(
        self,
        poem_units: List[VectorUnit],
        candidate_parks: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, List[VoteContribution]]]:
        """
        Similarity voting ONLY over the candidate parks from the top biome.

        For each poem unit:
          - compute best-match similarity per candidate park (max over that park's vectors)
          - award vote to top1 (and optionally top2) candidate parks
          - vote = similarity

        Returns:
          tallies: {park: score} only for candidate parks
          contributions: {park: [VoteContribution, ...]}
        """
        # If top biome maps to 0 parks, fall back to all parks
        parks = candidate_parks if candidate_parks else list(park_data.PARKS)

        tallies: Dict[str, float] = {p: 0.0 for p in parks}
        contributions: Dict[str, List[VoteContribution]] = {p: [] for p in parks}

        for u in poem_units:
            vec = np.asarray(u.vector, dtype=np.float32)
            vec_norm = _safe_norm(vec)
            if vec_norm <= 1e-12:
                continue

            park_best: List[Tuple[str, float, str]] = []

            for park in parks:
                mat = self._park_mats[park]
                norms = self._park_norms[park]
                sims = _cosine_sim_matrix_vec(mat, norms, vec, vec_norm)

                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                matched_text = self._park_unit_texts[park][best_idx]
                park_best.append((park, best_sim, matched_text))

            park_best.sort(key=lambda t: t[1], reverse=True)

            # ignore very weak matches (noise words)
            if park_best[0][1] < self.vote_similarity_threshold:
                continue

            # Top 1
            p1, s1, m1 = park_best[0]
            vote1 = s1
            tallies[p1] += vote1
            contributions[p1].append(
                VoteContribution(
                    poem_unit=u.text,
                    poem_unit_type=u.unit_type,
                    matched_park_unit=m1,
                    similarity=s1,
                    vote=vote1,
                )
            )

            # Top 2
            if self.vote_top2 and len(park_best) > 1:
                p2, s2, m2 = park_best[1]
                if s2 >= self.vote_similarity_threshold:
                    vote2 = s2 * self.top2_factor
                    tallies[p2] += vote2
                    contributions[p2].append(
                        VoteContribution(
                            poem_unit=u.text,
                            poem_unit_type=u.unit_type,
                            matched_park_unit=m2,
                            similarity=s2,
                            vote=vote2,
                        )
                    )

        return tallies, contributions


# -----------------------------
# Builds Units (tokens+chunks) from curated lists (no SpaCy extraction)
# -----------------------------

def _units_from_lists(tokens: Sequence[str], chunks: Sequence[str]) -> Units:
    """
    Create a Units object for vectorize_units without running extract_units.
    Normalizes case/whitespace; no dedupe.
    """
    tokens_clean = tuple(t.strip().lower() for t in tokens if t and t.strip())
    chunks_clean = tuple(c.strip().lower() for c in chunks if c and c.strip())
    return Units(tokens=tokens_clean, chunks=chunks_clean, debug={})