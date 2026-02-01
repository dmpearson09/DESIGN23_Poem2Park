"""
src/app.py

FastAPI backend for Poem2Park.
- Serves the frontend from / (static/index.html)
- Serves images from /images/... (static/images/...)
- Exposes POST /match which returns:
    best_park, ranked_parks, biomes, explanations, image_url
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .match import ParkMatcher

app = FastAPI(title="Poem2Park")

# Map park -> image path (you said your images are JPG and named like channel-island.jpg)
PARK_IMAGE = {
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

# Build matcher once at startup (precomputes park + biome vectors)
# Tune biome_alpha if you want biome gating to be stronger/weaker.
matcher = ParkMatcher(
    biome_alpha=2.0,
    explain_top_parks=3,
    explain_top_contribs=10,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match_poem(payload: dict):
    poem = (payload.get("poem") or "").strip()
    if not poem:
        return {"error": "Missing 'poem' text"}

    result = matcher.match(poem)

    explanations = {
        park: [
            {
                "poem_unit": c.poem_unit,
                "poem_unit_type": c.poem_unit_type,
                "matched_park_unit": c.matched_park_unit,
                "similarity": c.similarity,
                "gated_vote": c.gated_vote,
            }
            for c in contribs
        ]
        for park, contribs in result.explanations.items()
    }

    return {
        "best_park": result.best_park,
        "ranked_parks": result.ranked_parks[:3],   # top 3 for your UI format
        "biomes": result.biome_scores[:3],         # top 3 for your UI format
        "explanations": explanations,              # connections for best/top parks
        "image_url": PARK_IMAGE.get(result.best_park, ""),
    }


# Serve your simple site + images from the "static" folder at project root.
# - "/" loads static/index.html
# - "/images/..." loads static/images/...
app.mount("/", StaticFiles(directory="static", html=True), name="static")