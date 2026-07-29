# internal-representation-control

Replication of the _Intentional Control of Internal States_ section of Anthropic's _Emergent Introspective Awareness in Large Language Models_ ([Lindsey, 2025](https://transformer-circuits.pub/2025/introspection/index.html#control)) on Gemma 3 27B-it extended by also measuring the internal representation of concepts with Gemma Scope 2 SAEs and Natural Language Autoencoders.

The model is asked to write a fixed sentence while *thinking* / *not thinking* about a concept word, and must reproduce the sentence exactly. On the response tokens we then measure how strongly the concept is represented internally, three ways:

- **concept vectors** — cosine with a mean-difference vector for the word, against a null band from 100 control words
- **SAE latents** — activation of concept-selective Gemma Scope 2 latents, selected contrastively and labelled via Neuronpedia
- **NLA** — decoding the activations back into text with a Natural Language Autoencoder, scored by an LLM judge

Full write-up on my [website](https://juliuskamp.com/research/intentional-control-of-internal-representations/).


## Interactive viewer

Four pages, served via GitHub Pages: **https://juliuskamp.github.io/internal-representation-control/**

| Page | Shows |
| --- | --- |
| `index.html` | per-token strength for one word × sentence, all conditions |
| `aggregate.html` | mean ±1 std across words, per sentence |
| `layers.html` | strength vs layer, collapsed over tokens and sentences |
| `forest.html` | per-word paired Δ vs the no-mention baseline, at one layer |

Locally: `python -m http.server -d docs`, then open http://localhost:8000. The pages fetch their data, so `file://` will not work.

Every page mirrors its state into the URL, so any view can be linked or embedded. Add `?embed=1` to hide the page chrome:

```html
<iframe src="https://juliuskamp.github.io/internal-representation-control/?embed=1"
        width="100%" height="900"></iframe>
```

<details>
<summary>Embedding with automatic height (e.g. LessWrong)</summary>

Embedded pages post their content height to the parent, so the iframe can resize itself:

```html
<style>
html, body { margin: 0; padding: 0; }
#viz { display: block; width: 100%; border: 0; height: 620px; }
</style>

<iframe id="viz"
        src="https://juliuskamp.github.io/internal-representation-control/layers.html?embed=1"
        title="Concept-vector strength vs layer"
        scrolling="no"></iframe>

<script>
(function () {
  var f = document.getElementById("viz");
  window.addEventListener("message", function (e) {
    if (e.source !== f.contentWindow) return;
    var d = e.data || {}, h = d["irc-height"];
    if (typeof h !== "number" || !isFinite(h)) return;
    f.style.height = Math.ceil(h) + "px";
  });
})();
</script>
```

</details>

## Running it

Needs a GPU with ~55 GB of VRAM, [uv](https://docs.astral.sh/uv/), and access to the gated `google/gemma-3-27b-it` repo.

```bash
cp .env.example .env        # then fill in HF_TOKEN
```

Set `HF_HOME` in `.env` too if your home volume has no room for ~55 GB of weights.

### Pipeline

One command runs all four stages. A *run* is an incrementally grown dataset — re-invoking with the same `--run-id` extends it, and every stage is cached, so re-running is cheap and safe:

```bash
uv run python scripts/run_pipeline.py --run-id myrun
```

| Stage | Does | Needs GPU |
| --- | --- | --- |
| `vectors` | mean-difference concept vectors for 50 concept + 100 control words | yes |
| `generate` | generate + compliance-check each (word, sentence, condition), capture all-layer residuals | yes |
| `latents` | pick concept-selective SAE latents per word, label them via Neuronpedia | yes |
| `measure` | cosines and SAE stats from the stored activations | no |

Output lands in `artifacts/runs/{run_id}/`: `generations.jsonl`, `acts/*.pt` (the captured residuals), `results/*.parquet`, and `invocations.jsonl` recording the config, versions and git commit of every invocation.

### Viewer data and figures

```bash
uv run python scripts/export_viz_data.py --run-id myrun   # -> docs/data/
uv run python scripts/export_agg_data.py                  # -> docs/data/agg/
python -m http.server -d docs
```

`export_agg_data.py` derives its input from the word chunks, so run it after every export. Both write into `docs/data/`, which is committed.

Any viewer chart can be rendered to a static image (format from the `--out` extension: PNG, SVG or PDF), either from flags or by pasting a viewer URL:

```bash
uv run python scripts/render_viewer_figure.py --word Dust --sent 27 --meas sae --layer 40 --out dust.svg
uv run python scripts/render_viewer_figure.py "http://localhost:8000/layers.html?meas=word_tokens&delta=1"
```

### Smoke tests

Four standalone scripts that each check one piece end to end and print what they found — model loading and instruction-following (`a`), SAE wiring (`b`), concept-vector extraction (`c`), latent selection (`d`):

```bash
uv run python scripts/smoke_a_generate.py
```

## Layout

```
irc/            library: prompts, model hooks, concept vectors, pipeline stages
scripts/        entry points (pipeline, exports, figures, smoke tests, NLA)
docs/           the published static site, including its committed data
nla_server/     locked SGLang serving environment for the NLA checkpoint
artifacts/      generated data store (gitignored)
```

To publish the site: enable Pages in the repo settings (Settings → Pages → Deploy from branch → `main` / `docs/`) and push.

## License

MIT — see [LICENSE](LICENSE). `irc/vendor/` and `nla_server/patches/` are vendored third-party code under Apache-2.0, with provenance and licenses alongside.
