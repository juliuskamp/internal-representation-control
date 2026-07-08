# internal-representation-control

Replication of the intentional-control experiment from *Emergent Introspective
Awareness in Large Language Models* on Gemma 3 27B-it, with Gemma Scope 2 SAEs.

## Interactive viewer

`docs/` is a static site with a per-token representation-strength viewer
(served via GitHub Pages: https://juliuskamp.github.io/internal-representation-control/).

- Locally: `python -m http.server -d docs`, then open http://localhost:8000
  (the page fetches its data, so it will not work over `file://`).
- Embed elsewhere:
  `<iframe src="https://juliuskamp.github.io/internal-representation-control/?embed=1" width="100%" height="900"></iframe>`
- Regenerate the data from a pipeline run:
  `uv run python scripts/export_viz_data.py --run-id run1-core`

To publish: enable Pages in the repo settings (Settings → Pages → Deploy from
branch → `main` / `docs/`) and push.
