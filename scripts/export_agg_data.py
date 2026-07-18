"""Aggregate the per-word viewer data into per-sentence mean/std series.

Reads docs/data/index.json + docs/data/words/*.json.gz (written by
scripts/export_viz_data.py) and writes docs/data/agg/{si}.json.gz for the
aggregate viewer (docs/aggregate.html): for every sentence x measurement x
layer x condition x token, the mean and ±1 std across concept words.

Two series families are precomputed, each for two word sets:

  conds  — raw levels per condition
  deltas — paired per-word delta vs the no_mention condition (think and
           dont_think only). Word-level offsets (the shared generic cosine
           direction) cancel per word before aggregating, so the std here
           measures effect variability, not between-word level spread.

  "all"      — per condition, every word whose completion was exact (for
               deltas: both the condition and no_mention exact)
  "complete" — only words where all three conditions are exact for this
               sentence (same word set for every condition, avoids the
               exclusion bias of the think condition)

Also writes agg/layers.json.gz for the layer-curve viewer (docs/layers.html):
concept-vector measurements collapsed over tokens and sentences, mean ±1 std
across words per layer (two-stage: token-mean per word x sentence, then
sentence-mean per word, then across words — words are the replicates).
SAE/NLA are omitted there (4 layers / 1 layer make no curve).

Also writes agg/words.json.gz for the per-word forest view (docs/forest.html):
per word x layer, the sentence-mean paired delta vs no_mention (±1 std
across sentences).

Model-free and fast (reads only the exported chunks, not artifacts/);
re-run after every export_viz_data.py run. The agg files are committed
derived data, like the rest of docs/data/.
"""

from irc import env  # noqa: F401

import gzip
import json

import numpy as np

from irc.paths import DOCS_DATA

CONDS = ["think", "dont_think", "no_mention"]
DELTA_CONDS = ["think", "dont_think"]
CV_VARIANTS = ["word_tokens", "paper"]
SAE_VERSIONS = ["sae", "sae_v2"]
SAE_AGGS = ["sum", "mean", "max"]
SAE_FN = {"sum": np.sum, "mean": np.mean, "max": np.max}


def rounded(a: np.ndarray, nd: int) -> list:
    return np.round(a, nd).tolist()


def mean_std(stack: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = np.stack(stack)
    return a.mean(0), a.std(0)


def cond_words(records: dict, words: list[str], cid: str, mode: str,
               delta: bool) -> list[str]:
    if mode == "complete":
        return [w for w in words
                if all(records[w]["exact"].get(c) for c in CONDS)]
    need = [cid, "no_mention"] if delta else [cid]
    return [w for w in words if all(records[w]["exact"].get(c) for c in need)]


def sae_series(entry: dict, ver: str, li: int, agg: str) -> np.ndarray | None:
    """Latent-aggregated (T,) series for one word/condition/layer, or None."""
    x = entry.get(ver)
    if not x or x[li] is None:
        return None
    return SAE_FN[agg](x[li], axis=0)


def build_block(records: dict, ws: list[str], cid: str, delta: bool) -> dict | None:
    """mean/std over words for one condition (or its delta vs no_mention)."""
    if not ws:
        return None
    block: dict = {"n": len(ws)}
    for var in CV_VARIANTS:
        stack = []
        for w in ws:
            a = records[w][cid].get(var)
            b = records[w]["no_mention"].get(var) if delta else None
            if a is None or (delta and b is None):
                continue
            stack.append(a - b if delta else a)
        if stack:
            m, s = mean_std(stack)  # (layers, T)
            # summary column: per-word mean over tokens, then across words —
            # dispersion of the word-level effect (words are the replicates)
            tm, ts = mean_std([a.mean(1) for a in stack])  # (layers,)
            block[var] = {"n": len(stack), "mean": rounded(m, 4),
                          "std": rounded(s, 4),
                          "summary": {"mean": rounded(tm, 4),
                                      "std": rounded(ts, 4)}}
    for ver in SAE_VERSIONS:
        if not any(ver in records[w][cid] for w in ws):
            continue
        n_l = len(next(records[w][cid][ver] for w in ws
                       if ver in records[w][cid]))
        vb: dict = {"n": [], **{agg: {"mean": [], "std": [],
                                      "summary": {"mean": [], "std": []}}
                                for agg in SAE_AGGS}}
        for li in range(n_l):
            per_agg: dict[str, list[np.ndarray]] = {agg: [] for agg in SAE_AGGS}
            for w in ws:
                for agg in SAE_AGGS:
                    a = sae_series(records[w][cid], ver, li, agg)
                    if a is None:
                        break
                    if delta:
                        b = sae_series(records[w]["no_mention"], ver, li, agg)
                        if b is None:
                            break
                        a = a - b
                    per_agg[agg].append(a)
            vb["n"].append(len(per_agg["sum"]))
            for agg in SAE_AGGS:
                if not per_agg[agg]:
                    vb[agg]["mean"].append(None)
                    vb[agg]["std"].append(None)
                    vb[agg]["summary"]["mean"].append(None)
                    vb[agg]["summary"]["std"].append(None)
                    continue
                m, s = mean_std(per_agg[agg])  # (T,)
                vb[agg]["mean"].append(rounded(m, 3))
                vb[agg]["std"].append(rounded(s, 3))
                tm, ts = mean_std([a.mean() for a in per_agg[agg]])
                vb[agg]["summary"]["mean"].append(round(float(tm), 3))
                vb[agg]["summary"]["std"].append(round(float(ts), 3))
        block[ver] = vb
    nla = []
    for w in ws:
        a = records[w][cid].get("nla")
        b = records[w]["no_mention"].get("nla") if delta else None
        if a is None or (delta and b is None):
            continue
        nla.append(a - b if delta else a)  # NaN where either token unjudged
    if nla:
        a = np.stack(nla)  # (words, T)
        n_tok = np.sum(~np.isnan(a), axis=0)
        with np.errstate(invalid="ignore"):
            m, s = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
        block["nla"] = {
            "n": n_tok.tolist(),
            "mean": [None if k == 0 else round(float(v), 2)
                     for v, k in zip(m, n_tok)],
            "std": [None if k == 0 else round(float(v), 2)
                    for v, k in zip(s, n_tok)],
        }
        with np.errstate(invalid="ignore"):
            tmeans = np.nanmean(a, axis=1)  # per-word mean over judged tokens
        tmeans = tmeans[~np.isnan(tmeans)]
        if tmeans.size:
            block["nla"]["summary"] = {
                "n": int(tmeans.size),
                "mean": round(float(tmeans.mean()), 2),
                "std": round(float(tmeans.std()), 2)}
    return block


def layer_curves(per_word: dict) -> dict:
    """Per measurement variant x word-set mode: mean/std across words of the
    per-word layer profile (token-mean per sentence, then sentence-mean)."""
    words = sorted(per_word)
    out: dict = {}
    for var in CV_VARIANTS:
        out[var] = {}
        for mode in ("all", "complete"):
            vm: dict = {"conds": {}, "deltas": {}}
            for delta, family, cids in ((False, "conds", CONDS),
                                        (True, "deltas", DELTA_CONDS)):
                for cid in cids:
                    vecs = []
                    for w in words:
                        arrs = []
                        for rec in per_word[w].values():
                            need = CONDS if mode == "complete" else (
                                [cid, "no_mention"] if delta else [cid])
                            if not all(rec["exact"].get(c) for c in need):
                                continue
                            a = rec[cid].get(var)
                            b = rec["no_mention"].get(var) if delta else None
                            if a is None or (delta and b is None):
                                continue
                            arrs.append((a - b if delta else a).mean(1))
                        if arrs:
                            vecs.append(np.mean(arrs, axis=0))  # (layers,)
                    if vecs:
                        m, s = mean_std(vecs)
                        vm[family][cid] = {"n": len(vecs),
                                           "mean": rounded(m, 4),
                                           "std": rounded(s, 4)}
            out[var][mode] = vm
    return out


def eligible(rec: dict, cid: str, mode: str) -> bool:
    """Does this word x sentence record enter the paired-delta stats?"""
    need = CONDS if mode == "complete" else [cid, "no_mention"]
    return all(rec["exact"].get(c) for c in need)


def word_delta_stats(per_word: dict) -> dict:
    """agg/words.json.gz concept part: per word, the (layers,) sentence-mean
    paired delta vs no_mention with ±1 std across sentences."""
    words = sorted(per_word)
    out: dict = {"words": words}
    for var in CV_VARIANTS:
        out[var] = {}
        for mode in ("all", "complete"):
            vm: dict = {}
            for cid in DELTA_CONDS:
                mean, std, n = [], [], []
                for w in words:
                    arrs = []
                    for rec in per_word[w].values():
                        if not eligible(rec, cid, mode):
                            continue
                        a, b = rec[cid].get(var), rec["no_mention"].get(var)
                        if a is None or b is None:
                            continue
                        arrs.append((a - b).mean(1))  # (layers,)
                    if arrs:
                        m, s = mean_std(arrs)
                        mean.append(rounded(m, 4))
                        std.append(rounded(s, 4))
                    else:
                        mean.append(None)
                        std.append(None)
                    n.append(len(arrs))
                vm[cid] = {"mean": mean, "std": std, "n": n}
            out[var][mode] = vm
    return out


def collect(chunk: dict) -> dict[str, dict]:
    """Extract compact numpy records per sentence index from one word chunk."""
    out = {}
    for si, slot in chunk["slots"].items():
        rec: dict = {"exact": {}, "tokens": slot["tokens"],
                     "sentence": slot["sentence"]}
        for cid in CONDS:
            c = slot["conditions"].get(cid)
            rec["exact"][cid] = bool(c and c["exact"])
            entry: dict = {}
            if c and c["exact"]:
                for var in CV_VARIANTS:
                    if c.get(var):
                        entry[var] = np.asarray(c[var]["target"], dtype=np.float32)
                for ver in SAE_VERSIONS:
                    if c.get(ver):
                        entry[ver] = [np.asarray(l, dtype=np.float32)
                                      if l else None for l in c[ver]]
                nla = c.get("nla")
                if nla and any(s is not None for s in nla["score"]):
                    entry["nla"] = np.asarray(
                        [np.nan if s is None else s for s in nla["score"]],
                        dtype=np.float32)
            rec[cid] = entry
        out[si] = rec
    return out


def main() -> None:
    index = json.loads((DOCS_DATA / "index.json").read_text())
    per_word = {}
    for f in sorted((DOCS_DATA / "words").glob("*.json.gz")):
        per_word[f.name.removesuffix(".json.gz")] = collect(
            json.loads(gzip.decompress(f.read_bytes())))
    words = sorted(per_word)

    out_dir = DOCS_DATA / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)
    sis = sorted({si for recs in per_word.values() for si in recs}, key=int)
    sentences = {}
    for si in sis:
        records = {w: recs[si] for w, recs in per_word.items() if si in recs}
        ws = sorted(records)
        tokens = records[ws[0]]["tokens"]
        for w in ws:
            assert records[w]["tokens"] == tokens, \
                f"token mismatch for sentence {si}, word {w}"
        payload = {
            "sentence": records[ws[0]]["sentence"],
            "tokens": tokens,
            "n_words": len(ws),
            "conds": {
                cid: {mode: build_block(
                    records, cond_words(records, ws, cid, mode, False),
                    cid, False) for mode in ("all", "complete")}
                for cid in CONDS},
            "deltas": {
                cid: {mode: build_block(
                    records, cond_words(records, ws, cid, mode, True),
                    cid, True) for mode in ("all", "complete")}
                for cid in DELTA_CONDS},
        }
        (out_dir / f"{si}.json.gz").write_bytes(
            gzip.compress(json.dumps(payload).encode(), mtime=0))
        sentences[si] = payload["sentence"]
    # sentence texts for the dropdown, so labels don't need the full chunks
    (out_dir / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False))
    (out_dir / "layers.json.gz").write_bytes(
        gzip.compress(json.dumps(layer_curves(per_word)).encode(), mtime=0))

    (out_dir / "words.json.gz").write_bytes(
        gzip.compress(json.dumps(word_delta_stats(per_word)).encode(), mtime=0))
    print(f"wrote {len(sis)} sentence aggregates "
          f"({len(words)} words, run {index['run_id']}) to {out_dir}")


if __name__ == "__main__":
    main()
