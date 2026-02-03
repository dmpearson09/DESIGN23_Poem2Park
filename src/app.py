"""
src/app.py

FastAPI backend for Poem2Park.

Responsibilities:
- Serve the static frontend from "/" (static/index.html)
- Serve images from "/images/..." (static/images/...)
- Provide JSON API:
    POST /match  -> best park, ranked parks, top biomes, explanations, image_url
    GET  /health -> basic liveness check
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .match import ParkMatcher

app = FastAPI(title="Poem2Park")

# Park name -> image URL (served from static/images/)
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

# Build matcher once at startup (precomputes park + biome vectors).
# All tuning defaults should live in match.py now.
matcher = ParkMatcher()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/match")
def match_poem(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Read poem from request JSON
    poem = str(payload.get("poem", "")).strip()
    if not poem:
        return {"error": "Missing 'poem' text"}

    # Run matching
    result = matcher.match(poem)

    # Convert explanation dataclasses -> JSON-friendly dicts
    explanations: Dict[str, Any] = {}
    for park, contribs in result.explanations.items():
        explanations[park] = [
            {
                "poem_unit": c.poem_unit,
                "poem_unit_type": c.poem_unit_type,
                "matched_park_unit": c.matched_park_unit,
                "similarity": float(c.similarity),
                "gated_vote": float(c.gated_vote),
            }
            for c in contribs
        ]

    # Keep response shaped exactly how your frontend expects
    return {
        "best_park": result.best_park,
        "ranked_parks": result.ranked_parks[:3],
        "biomes": result.biome_scores[:3],
        "explanations": explanations,
        "image_url": PARK_IMAGE.get(result.best_park, ""),
    }


# Serve the website + images from the "static" folder at repo root.
# "/" -> static/index.html
# "/images/..." -> static/images/...
app.mount("/", StaticFiles(directory="static", html=True), name="static")