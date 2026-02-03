"""
src/app.py

FastAPI backend for Poem2Park.
- Serves the frontend from / (static/index.html)
- Serves images from /images/... (static/images/...)
- Exposes POST /match which returns:
    best_park, ranked_parks, biomes, explanations, biome_explanations, image_url
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .match import ParkMatcher

app = FastAPI(title="Poem2Park")

# Park Image Path
PARK_IMAGE: Dict[str, str] = {
    "Channel Islands": "/images/channel-island.jpg",
    "Death Valley": "/images/death-valley.jpg",
    "Joshua Tree": "/images/joshua-tree.jpg",
    "Kings Canyon": "/images/kings-canyon.jpg",
    "Lassen Volcanic": "/images/lassen-volcanic.jpg",
    "Pinnacles": "/images/pinnacles.jpg",
    "Redwood": "/images/redwood.jpg",
    "Sequoia": "/images/sequoia.jpg",
    "Yosemite": "/images/yosemite.jpg",
}

# Build matcher once at startup. All tuning defaults live in match.py.
matcher = ParkMatcher()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/match")
def match_poem(payload: Dict[str, Any]) -> Dict[str, Any]:
    poem = str(payload.get("poem", "")).strip()
    if not poem:
        return {"error": "Missing 'poem' text"}

    result = matcher.match(poem)

    # Top Park explanations (JSON)
    explanations: Dict[str, Any] = {}
    if result.explanations:
        best_park = next(iter(result.explanations.keys()))
        contribs = result.explanations[best_park]
        explanations[best_park] = [
            {
                "poem_unit": c.poem_unit,
                "poem_unit_type": c.poem_unit_type,
                "matched_park_unit": c.matched_park_unit,
                "similarity": float(c.similarity),
                "vote": float(c.vote),
            }
            for c in contribs
        ]

    # Top Biome explanations (JSON)
    biome_explanations: Dict[str, Any] = {}
    if result.biome_explanations:
        top_biome = next(iter(result.biome_explanations.keys()))
        contribs = result.biome_explanations[top_biome]
        biome_explanations[top_biome] = [
            {
                "poem_unit": c.poem_unit,
                "poem_unit_type": c.poem_unit_type,
                "matched_biome_unit": c.matched_biome_unit,
                "similarity": float(c.similarity),
            }
            for c in contribs
        ]

    return {
        "best_park": result.best_park,
        "ranked_parks": result.ranked_parks,
        "biomes": result.biome_scores,
        "explanations": explanations,
        "biome_explanations": biome_explanations,
        "image_url": PARK_IMAGE.get(result.best_park, ""),
    }


# Serve site + images from the "static" folder at project root.
app.mount("/", StaticFiles(directory="static", html=True), name="static")