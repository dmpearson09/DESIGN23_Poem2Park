"""
main.py

Terminal Based Run for Poem2Park.

Returns
  - Best Match 
  - Top parks
  - Top Biomes
  - Word Connections (Best Park)
  - Word Connections (Best Biome)

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
    print("          Poem2Park")
    print("==============================\n")

    print(f"Best Match:\n1. {result.best_park}\n")

    print("Top Parks:")
    for i, (park, score) in enumerate((result.ranked_parks or [])[:3], start=1):
        print(f"{i}. {park}  (score={score:.3f})")

    print("\nTop Biomes:")
    for i, (biome, score) in enumerate((result.biome_scores or [])[:3], start=1):
        print(f"{i}. {biome}  (score={score:.3f})")

    # --- Park connections (best park only) ---
    print("\nWord Connections (Best Park):")
    park_contribs = []
    if getattr(result, "explanations", None):
        park_contribs = result.explanations.get(result.best_park, [])

    if not park_contribs:
        print("(no strong contributors)")
    else:
        for c in park_contribs[:10]:
            print(
                f"• '{c.poem_unit}' ({c.poem_unit_type}) → '{c.matched_park_unit}' "
                f"[sim={c.similarity:.2f}, vote={c.vote:.2f}]"
            )

    # --- Biome connections (top biome only) ---
    print("\nWord Connections (Best Biome):")
    top_biome = result.biome_scores[0][0] if result.biome_scores else ""
    biome_contribs = []
    if getattr(result, "biome_explanations", None) and top_biome:
        biome_contribs = result.biome_explanations.get(top_biome, [])

    if not biome_contribs:
        print("(no strong contributors)")
    else:
        for c in biome_contribs[:10]:
            print(
                f"• '{c.poem_unit}' ({c.poem_unit_type}) → '{c.matched_biome_unit}' "
                f"[sim={c.similarity:.2f}]"
            )


def main() -> None:
    matcher = ParkMatcher()

    poem = read_poem_from_terminal()
    if not poem.strip():
        print("No poem entered. Exiting.")
        return

    result = matcher.match(poem)
    print_results(result)


if __name__ == "__main__":
    main()