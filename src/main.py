"""
main.py

Terminal-based entry point for Poem2Park.

Run from project root:
  python -m src.main

Paste a poem (multi-line supported), then press ENTER twice.
"""

from __future__ import annotations

from .match import ParkMatcher


def read_poem_from_terminal() -> str:
    """
    Read a multi-line poem from stdin.
    End input by submitting an empty line.
    """
    print("\nPaste your poem below.")
    print("Press ENTER on an empty line to finish.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "":
            break
        lines.append(line)

    return "\n".join(lines)


def print_results(result) -> None:
    print("\n==============================")
    print("        Poem2Park Result      ")
    print("==============================\n")

    print(f"🏞  Best match: {result.best_park}\n")

    print("Top parks:")
    for i, (park, score) in enumerate(result.ranked_parks[:3], start=1):
        print(f"  {i}. {park:15s}  score = {score:.3f}")

    print("\nTop inferred biomes:")
    for biome, score in result.biome_scores[:4]:
        print(f"  - {biome:18s}  {score:.3f}")

    print("\nWhy this match (top contributing phrases):")
    for park, _score in result.ranked_parks[:2]:
        print(f"\n{park}:")
        contribs = result.explanations.get(park, [])
        if not contribs:
            print("  (no strong contributors)")
            continue

        for c in contribs[:6]:
            print(
                f"  • '{c.poem_unit}' ({c.poem_unit_type}) "
                f"→ '{c.matched_park_unit}' "
                f"[sim={c.similarity:.2f}, vote={c.gated_vote:.2f}]"
            )


def main() -> None:
    matcher = ParkMatcher(biome_alpha=3, vote_similarity_threshold=0.4, park_biome_mix_max=0.9, park_biome_mix_mean=0.1)

    poem = read_poem_from_terminal()
    if not poem.strip():
        print("No poem entered. Exiting.")
        return

    result = matcher.match(poem)
    print_results(result)


if __name__ == "__main__":
    main()