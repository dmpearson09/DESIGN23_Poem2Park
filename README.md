# Poem2Park (DESIGN23)

Poem2Park takes a poem as input and returns the California national park (among 9) it most resembles.
It uses spaCy embeddings, multi-vector voting, and biome gating.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md