"""
park_data.py

Hardcoded databases for:
1) Park language profiles (terms + noun chunks)
2) Park -> biome tags
3) Biome language profiles (terms + noun chunks)

This file is intentionally "data-first" so you can iterate on lists without
touching matching logic.

Notes:
- Keep items lowercase.
- Terms should be single words (no spaces).
- Chunks should be multi-word phrases (contain spaces).
- You can start small and expand over time.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# -----------------------------
# Canonical park list (9 CA NPs)
# -----------------------------
PARKS: Tuple[str, ...] = (
    "Channel Islands",
    "Death Valley",
    "Joshua Tree",
    "Kings Canyon",
    "Lassen Volcanic",
    "Pinnacles",
    "Redwood",
    "Sequoia",
    "Yosemite",
)

# -----------------------------
# Biome taxonomy (your updated list)
# -----------------------------
BIOMES: Tuple[str, ...] = (
    "Marine_Costal",
    "Dessert",
    "Forrest",
    "Mountains_Hills",
    "Rivers_Lakes",
    "Meadows_Flowers",
    "Volcanic_Fire",
)

# --------------------------------------
# Park -> biomes (your updated mapping)
# --------------------------------------
PARK_BIOMES: Dict[str, List[str]] = {
    "Channel Islands": ["Marine_Costal"],
    "Death Valley": ["Dessert"],
    "Joshua Tree": ["Dessert"],
    "Pinnacles": ["Dessert", "Meadows_Flowers"],
    "Redwood": ["Forrest"],
    "Sequoia": ["Forrest"],
    "Yosemite": ["Mountains_Hills", "Rivers_Lakes", "Meadows_Flowers"],
    "Kings Canyon": ["Mountains_Hills", "Rivers_Lakes"],
    "Lassen Volcanic": ["Mountains_Hills", "Volcanic_Fire"],
}

# ============================================================
# 1) PARK LANGUAGE DATABASE
# ============================================================
# Each park has:
# - "terms": single-word NOUN/PROPN/ADJ/VERB-like terms (no stopwords)
# - "chunks": multi-word noun chunks / phrases (2+ words)
#
# Keep these lists curated and expandable. Start modest, then add more as you test.
# ============================================================

PARK_TERMS: Dict[str, List[str]] = {
    "Channel Islands": [
        "island",
        "channel",
        "coast",
        "ocean",
        "sea",
        "kelp",
        "tide",
        "surf",
        "shore",
        "harbor",
        "seal",
        "sea lion",
        "whale",
        "dolphin",
        "gull",
        "pelican",
        "brine",
        "spray",
        "cliff",
        "bluff",
        "cove",
        "anchorage",
        "marine",
        "fog",
        "wind",
        "sandstone",
    ],
    "Death Valley": [
        "desert",
        "dune",
        "sand",
        "salt",
        "basin",
        "badlands",
        "canyon",
        "butte",
        "mesa",
        "mirage",
        "heat",
        "sun",
        "arid",
        "dry",
        "cracked",
        "alkali",
        "scorch",
        "barren",
        "gully",
        "ravine",
        "windswept",
        "sage",
        "creosote",
        "pan",
    ],
    "Joshua Tree": [
        "desert",
        "cactus",
        "yucca",
        "joshua",
        "boulder",
        "granite",
        "sand",
        "basin",
        "sky",
        "star",
        "moon",
        "arid",
        "dry",
        "spine",
        "thorn",
        "ocotillo",
        "cholla",
        "scrub",
        "ridge",
        "hike",
        "climb",
        "sunrise",
        "twilight",
    ],
    "Kings Canyon": [
        "canyon",
        "river",
        "creek",
        "waterfall",
        "granite",
        "pine",
        "fir",
        "cedar",
        "forest",
        "ridge",
        "peak",
        "alpine",
        "snow",
        "meadow",
        "valley",
        "trail",
        "glacier",
        "lake",
        "basin",
        "gorge",
        "cascade",
        "roar",
    ],
    "Lassen Volcanic": [
        "volcano",
        "lava",
        "basalt",
        "cinder",
        "crater",
        "caldera",
        "steam",
        "fumarole",
        "sulfur",
        "geyser",
        "thermal",
        "ash",
        "pumice",
        "obsidian",
        "fire",
        "burn",
        "mountain",
        "snow",
        "lake",
        "pine",
        "ridge",
        "eruption",
    ],
    "Pinnacles": [
        "pinnacle",
        "spire",
        "talus",
        "cave",
        "chaparral",
        "oak",
        "grass",
        "wildflower",
        "meadow",
        "ridge",
        "canyon",
        "rock",
        "crag",
        "condor",
        "hawk",
        "sun",
        "heat",
        "dry",
        "scrub",
        "trail",
        "hike",
        "bloom",
    ],
    "Redwood": [
        "redwood",
        "coast",
        "fog",
        "fern",
        "moss",
        "grove",
        "canopy",
        "forest",
        "creek",
        "river",
        "salmon",
        "trail",
        "damp",
        "shade",
        "cathedral",
        "ancient",
        "tower",
        "giant",
        "mist",
        "coastal",
        "spruce",
    ],
    "Sequoia": [
        "sequoia",
        "giant",
        "grove",
        "tree",
        "trunk",
        "bark",
        "forest",
        "granite",
        "dome",
        "canyon",
        "alpine",
        "snow",
        "meadow",
        "ridge",
        "peak",
        "trail",
        "high",
        "thin",
        "pine",
        "bear",
    ],
    "Yosemite": [
        "yosemite",
        "granite",
        "dome",
        "cliff",
        "wall",
        "valley",
        "river",
        "creek",
        "waterfall",
        "cascade",
        "pine",
        "fir",
        "meadow",
        "wildflower",
        "alpine",
        "snow",
        "peak",
        "ridge",
        "glacier",
        "trail",
        "mist",
        "thunder",
    ],
}

PARK_CHUNKS: Dict[str, List[str]] = {
    "Channel Islands": [
        "kelp forest",
        "rocky shore",
        "sea cave",
        "tide pool",
        "marine terrace",
        "coastal bluff",
        "salt spray",
        "ocean swell",
        "harbor seal",
        "sea lion rookery",
    ],
    "Death Valley": [
        "salt flats",
        "sand dunes",
        "dry lakebed",
        "desert basin",
        "badlands ridge",
        "scorched earth",
        "cracked clay",
        "wind carved canyon",
        "sun baked rock",
    ],
    "Joshua Tree": [
        "joshua tree",
        "granite boulders",
        "desert night sky",
        "cactus garden",
        "rocky wash",
        "high desert",
        "hidden valley",
        "boulder field",
    ],
    "Kings Canyon": [
        "deep canyon",
        "granite gorge",
        "alpine lake",
        "river canyon",
        "pine forest",
        "high country",
        "glacial valley",
        "roaring waterfall",
        "wide meadow",
    ],
    "Lassen Volcanic": [
        "cinder cone",
        "lava field",
        "sulfur vents",
        "boiling springs",
        "steam plume",
        "volcanic crater",
        "obsidian flow",
        "thermal basin",
        "ash covered slope",
    ],
    "Pinnacles": [
        "rock spires",
        "talus cave",
        "chaparral hills",
        "spring wildflowers",
        "rolling grassland",
        "condor flight",
        "sunlit ridges",
    ],
    "Redwood": [
        "redwood grove",
        "coastal fog",
        "fern lined trail",
        "mossy forest",
        "cathedral canopy",
        "ancient giants",
        "misty creek",
    ],
    "Sequoia": [
        "giant sequoias",
        "sequoia grove",
        "granite domes",
        "high sierra",
        "alpine meadow",
        "snowy peaks",
        "deep canyon",
    ],
    "Yosemite": [
        "granite walls",
        "glacial valley",
        "towering waterfall",
        "misty cascade",
        "wildflower meadow",
        "river bend",
        "pine scented air",
        "granite dome",
    ],
}

# ============================================================
# 2) BIOME LANGUAGE DATABASE (for Option B biome prototypes)
# ============================================================
# These are NOT parks — they're biome concepts used to infer the poem's biome
# using embedding similarity.
# ============================================================

BIOME_TERMS: Dict[str, List[str]] = {
    "Marine_Costal": [
        "ocean",
        "sea",
        "marine",
        "coast",
        "coastal",
        "island",
        "kelp",
        "tide",
        "surf",
        "shore",
        "spray",
        "brine",
        "harbor",
        "seal",
        "gull",
        "whale",
        "fog",
        "salt",
        "cove",
    ],
    "Dessert": [
        "desert",
        "arid",
        "dry",
        "dune",
        "sand",
        "cactus",
        "yucca",
        "heat",
        "sun",
        "mirage",
        "barren",
        "scrub",
        "mesa",
        "butte",
        "salt",
        "basin",
    ],
    "Forrest": [
        "forest",
        "grove",
        "canopy",
        "tree",
        "redwood",
        "sequoia",
        "fern",
        "moss",
        "shade",
        "damp",
        "mist",
        "loam",
        "understory",
        "trail",
    ],
    "Mountains_Hills": [
        "mountain",
        "hill",
        "ridge",
        "peak",
        "summit",
        "alpine",
        "granite",
        "cliff",
        "crag",
        "snow",
        "glacier",
        "basin",
        "high",
    ],
    "Rivers_Lakes": [
        "river",
        "creek",
        "stream",
        "waterfall",
        "cascade",
        "rapids",
        "lake",
        "shoreline",
        "pool",
        "confluence",
        "ford",
        "wet",
        "current",
    ],
    "Meadows_Flowers": [
        "meadow",
        "wildflower",
        "bloom",
        "petal",
        "grass",
        "prairie",
        "spring",
        "field",
        "blossom",
        "pollen",
        "nectar",
        "butterfly",
    ],
    "Volcanic_Fire": [
        "volcano",
        "lava",
        "basalt",
        "cinder",
        "crater",
        "caldera",
        "obsidian",
        "ash",
        "pumice",
        "fumarole",
        "sulfur",
        "steam",
        "thermal",
        "fire",
        "ember",
    ],
}

BIOME_CHUNKS: Dict[str, List[str]] = {
    "Marine_Costal": [
        "kelp forest",
        "tide pool",
        "rocky shore",
        "coastal bluff",
        "salt spray",
        "ocean swell",
        "sea cave",
    ],
    "Dessert": [
        "sand dunes",
        "salt flats",
        "dry lakebed",
        "high desert",
        "sun baked rock",
        "arid basin",
    ],
    "Forrest": [
        "old growth forest",
        "redwood grove",
        "mossy trail",
        "fern covered floor",
        "cathedral canopy",
    ],
    "Mountains_Hills": [
        "granite ridge",
        "alpine peak",
        "snowy summit",
        "glacial valley",
        "rocky cliffs",
    ],
    "Rivers_Lakes": [
        "roaring waterfall",
        "river bend",
        "mountain creek",
        "glacial lake",
        "rushing rapids",
    ],
    "Meadows_Flowers": [
        "wildflower meadow",
        "spring bloom",
        "grassy field",
        "flowered hills",
    ],
    "Volcanic_Fire": [
        "lava field",
        "cinder cone",
        "sulfur vents",
        "steam plume",
        "volcanic crater",
        "thermal basin",
    ],
}

# ============================================================
# Basic integrity checks (optional but helpful during dev)
# ============================================================

def validate_databases() -> None:
    """Raise ValueError if the hardcoded data has obvious issues."""
    # Parks present
    for park in PARKS:
        if park not in PARK_TERMS:
            raise ValueError(f"Missing PARK_TERMS for park: {park}")
        if park not in PARK_CHUNKS:
            raise ValueError(f"Missing PARK_CHUNKS for park: {park}")
        if park not in PARK_BIOMES:
            raise ValueError(f"Missing PARK_BIOMES for park: {park}")

    # Biomes present
    for biome in BIOMES:
        if biome not in BIOME_TERMS:
            raise ValueError(f"Missing BIOME_TERMS for biome: {biome}")
        if biome not in BIOME_CHUNKS:
            raise ValueError(f"Missing BIOME_CHUNKS for biome: {biome}")

    # Park biomes must be valid
    biome_set: Set[str] = set(BIOMES)
    for park, tags in PARK_BIOMES.items():
        for tag in tags:
            if tag not in biome_set:
                raise ValueError(f"Unknown biome tag '{tag}' in PARK_BIOMES for park '{park}'")

    # Basic lowercase enforcement
    def _check_lower(items: List[str], label: str) -> None:
        for s in items:
            if s != s.lower():
                raise ValueError(f"{label} item not lowercase: '{s}'")

    for park in PARKS:
        _check_lower(PARK_TERMS[park], f"PARK_TERMS[{park}]")
        _check_lower(PARK_CHUNKS[park], f"PARK_CHUNKS[{park}]")

    for biome in BIOMES:
        _check_lower(BIOME_TERMS[biome], f"BIOME_TERMS[{biome}]")
        _check_lower(BIOME_CHUNKS[biome], f"BIOME_CHUNKS[{biome}]")


# Run validations during import in dev (comment out if you prefer)
validate_databases()