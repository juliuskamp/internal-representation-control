"""Score NLA explanations against concept words with an LLM judge.

For every (word, sentence) pair with NLA rows, each explanation from the
think / dont_think / no_mention conditions is judged against the concept
word. no_mention explanations are word-free, so they are the control: they
are judged against every concept word that has its own NLA rows on the same
sentence (same pairing as export_viz_data.py).

Scoring follows Betley et al. (2025) / the persona-vector paper: the judge
(GPT-4.1 via OpenRouter) writes an Evidence quote, then a bare integer score
0-100. We store the sampled score AND the logit-weighted expectation over
integer tokens in the top-20 logprobs at the score position (plus the
renormalized distribution), so the two readouts can be compared.

    uv run python scripts/nla_judge.py --run-id run1-core --pilot   # one call, no write
    uv run python scripts/nla_judge.py --run-id run1-core

Requires OPENROUTER_API_KEY in .env. Resumable: judgments already present in
the output file (same explanation row, target word, prompt version, model)
are skipped on re-run. The prompt template lives in irc/nla_judge_prompt.py
and is versioned; the version is stored with every judgment.
"""

from irc import env  # noqa: F401  (must be first: loads .env, sets HF_HOME)

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import httpx

from irc.nla_judge_prompt import JUDGE_PROMPT_VERSION, JUDGE_PROMPTS

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

EVIDENCE_RE = re.compile(r"Evidence:\s*(.*?)\s*$", re.MULTILINE)
SCORE_RE = re.compile(r"Score:\s*(\d{1,3})")


def judge_call(client: httpx.Client, model: str, prompt: str,
               max_tokens: int, retries: int = 3):
    """One judge call. Returns (text, logprob_content, provider)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 20,
        # Pin to OpenAI: it returns logprobs (verified) but doesn't advertise
        # them to the router, so require_parameters would 404. Other
        # providers (e.g. Azure) may silently drop logprobs.
        "provider": {"order": ["openai"], "allow_fallbacks": False},
    }
    for attempt in range(retries):
        try:
            resp = client.post(OPENROUTER_URL, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            lp = (choice.get("logprobs") or {}).get("content") or []
            return choice["message"]["content"], lp, data.get("provider")
        except (httpx.HTTPError, KeyError, ValueError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  judge call failed ({e}), retrying in {wait}s")
            time.sleep(wait)


def score_from_logprobs(logprob_content: list) -> tuple[int, float, list] | None:
    """Find the score token (last generated integer 0-100) and compute the
    probability-weighted expectation over integer candidates in its top-20
    logprobs. Returns (sampled, expected, distribution) or None if no score
    token was found."""

    def as_score(token: str) -> int | None:
        s = token.strip()
        if s.isdigit() and 0 <= int(s) <= 100:
            return int(s)
        return None

    for tok in reversed(logprob_content):
        sampled = as_score(tok["token"])
        if sampled is None:
            continue
        probs: dict[int, float] = {}
        for cand in tok.get("top_logprobs", []):
            v = as_score(cand["token"])
            if v is not None:  # " 85" and "85" both count toward 85
                probs[v] = probs.get(v, 0.0) + math.exp(cand["logprob"])
        if not probs:
            return sampled, float(sampled), [[sampled, 1.0]]
        total = sum(probs.values())
        expected = sum(v * p for v, p in probs.items()) / total
        dist = sorted(([v, p / total] for v, p in probs.items()),
                      key=lambda x: -x[1])
        return sampled, expected, dist
    return None


def parse_evidence(text: str) -> str | None:
    m = EVIDENCE_RE.search(text)
    if not m:
        return None
    ev = m.group(1).strip().strip('"')
    return None if ev.lower() in ("none", "") else ev


def iter_tasks(expl_rows: list[dict], words: list[str] | None):
    """Yield (row, target_word). Worded rows are judged against their own
    word; no_mention rows against every word with rows on that sentence."""
    words_by_sentence: dict[int, set[str]] = {}
    for r in expl_rows:
        if r["word"]:
            words_by_sentence.setdefault(r["sentence_idx"], set()).add(r["word"])
    for r in expl_rows:
        targets = [r["word"]] if r["word"] else \
            sorted(words_by_sentence.get(r["sentence_idx"], ()))
        for t in targets:
            if words and t not in words:
                continue
            yield r, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--words", nargs="*", default=None,
                    help="Target concept words (Capitalized). Default: all.")
    ap.add_argument("--sentences", nargs="*", type=int, default=None,
                    help="Restrict to these sentence indices (matches "
                         "nla_explain.py --sentences). Default: all.")
    ap.add_argument("--conditions", nargs="*",
                    default=["think", "dont_think", "no_mention"])
    ap.add_argument("--agg", choices=["mean", "token"], default="token",
                    help="Which explanations file to judge (matches "
                         "nla_explain.py --agg).")
    ap.add_argument("--layer", type=int, default=41)
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max judgments per (target word, condition).")
    ap.add_argument("--pilot", action="store_true",
                    help="Judge only the first task, print the full exchange "
                         "(prompt, response, logprob readout), write nothing.")
    ap.add_argument("--out", default=None,
                    help="Output jsonl (default: results/nla_judgments_"
                         "{agg}_L{layer}.jsonl in the run dir). Appends.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — add it to .env")

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id
    expl_path = run_dir / "results" / \
        f"nla_explanations_{args.agg}_L{args.layer}.jsonl"
    out_path = Path(args.out) if args.out else (
        run_dir / "results" / f"nla_judgments_{args.agg}_L{args.layer}.jsonl"
    )

    with open(expl_path) as f:
        expl_rows = [json.loads(line) for line in f]
    expl_rows = [r for r in expl_rows if r["condition"] in args.conditions]
    # Restrict the row set before iter_tasks so no_mention rows are also judged
    # only against words present on the kept sentences (words_by_sentence is
    # rebuilt from this filtered set).
    if args.sentences is not None:
        keep = set(args.sentences)
        expl_rows = [r for r in expl_rows if r["sentence_idx"] in keep]

    sentences: dict[int, str] = {}
    with open(run_dir / "generations.jsonl") as f:
        for line in f:
            row = json.loads(line)
            sentences[row["sentence_idx"]] = row["sentence"]

    done: set[tuple] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line from an interrupted write
                done.add((r["key"], r["position"], r["target_word"],
                          r["prompt_version"], r["judge_model"]))

    template = JUDGE_PROMPTS[JUDGE_PROMPT_VERSION]
    client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})

    def iter_pending():
        """Yield (row, target) still needing a judgment: not already in the
        output file (resume) and within the per-(word, condition) --limit.
        Iterates the in-memory expl_rows, so it is cheap to run twice — once
        to size the progress indicator, once to drive the judging."""
        per_slot: dict[tuple[str, str], int] = {}
        for row, target in iter_tasks(expl_rows, args.words):
            ident = (row["key"], row["position"], target,
                     JUDGE_PROMPT_VERSION, args.model)
            if ident in done:
                continue
            slot = (target, row["condition"])
            per_slot[slot] = per_slot.get(slot, 0)
            if args.limit is not None and per_slot[slot] >= args.limit:
                continue
            per_slot[slot] += 1
            yield row, target

    # Cheap first pass (no network) to size the progress indicator.
    total = sum(1 for _ in iter_pending())
    if total == 0:
        print("nothing to judge — all explanations already judged (or filtered "
              "out). Bump the prompt version or pass a different --model to redo.")
        return
    if not args.pilot:
        print(f"judging {total} explanation(s) with {args.model}")

    n_done = 0
    out = None if args.pilot else open(out_path, "a")
    try:
        for row, target in iter_pending():
            prompt = template.format(
                word=target.lower(),  # concept words are stored Capitalized
                sentence=sentences[row["sentence_idx"]],
                explanation=row["explanation"],
            )
            text, lp, provider = judge_call(
                client, args.model, prompt, args.max_tokens)

            scored = score_from_logprobs(lp)
            text_score = SCORE_RE.search(text)
            if scored is None:
                print(f"  WARNING no integer score token found for "
                      f"{row['key']} pos={row['position']} vs {target}; "
                      f"text score: {text_score and text_score.group(1)}")
                sampled, expected, dist = (
                    int(text_score.group(1)) if text_score else None, None, None)
            else:
                sampled, expected, dist = scored
                if text_score and int(text_score.group(1)) != sampled:
                    print(f"  WARNING score token {sampled} != text score "
                          f"{text_score.group(1)} for {row['key']}")

            rec = {
                "key": row["key"],
                "condition": row["condition"],
                "word": row["word"],
                "target_word": target,
                "sentence_idx": row["sentence_idx"],
                "layer": row["layer"],
                "position": row["position"],
                "evidence": parse_evidence(text),
                "score": sampled,
                "score_expected": expected,
                "score_distribution": dist,
                "judge_model": args.model,
                "judge_provider": provider,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "raw": text,
            }

            if args.pilot:
                print("=== PROMPT ===\n" + prompt)
                print("\n=== RESPONSE ===\n" + text)
                print(f"\n=== SCORING (provider: {provider}) ===")
                print(f"logprob positions returned: {len(lp)}")
                print(f"sampled score: {sampled}   expectation: {expected}")
                print(f"distribution: {dist}")
                return

            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_done += 1
            exp_str = f"{expected:.1f}" if expected is not None else "?"
            print(f"[{n_done}/{total} {100 * n_done // total}%] "
                  f"{row['key']} pos={row['position']} vs "
                  f"{target}: {sampled} (E={exp_str})")
    finally:
        if out is not None:
            out.close()

    print(f"\nwrote {n_done} judgments to {out_path}")


if __name__ == "__main__":
    main()
