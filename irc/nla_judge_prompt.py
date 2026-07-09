"""Judge prompt for scoring NLA explanations against a concept word.

Versioned like the word lists (see CLAUDE.md): never edit a released version
in place — add a V2 and bump JUDGE_PROMPT_VERSION. The version string is
stored with every judgment so scores from different prompts are never mixed.

Placeholders: {word} (lowercased concept word), {sentence} (the sentence the
model was writing), {explanation} (the NLA-decoded explanation to score).

Output contract (parsed by scripts/nla_judge.py): an `Evidence:` line
followed by a `Score:` line whose value is a bare integer 0-100 — integers in
this range are single tokens in OpenAI's tokenizer, so the score token's
top-20 logprobs give the full score distribution (Betley et al. 2025 /
persona-vector methodology). Evidence comes first deliberately: the score
token is generated conditioned on the quote, not the other way around.
"""

JUDGE_PROMPT_VERSION = "v1"

JUDGE_PROMPT_V1 = """\
You are scoring the output of an interpretability tool. The tool decodes a \
language model's internal activations into a natural-language description \
(an "explanation") of what the model was representing at one moment while \
it wrote a sentence.

The model was writing this sentence: "{sentence}"

Explanation decoded from the model's activations:
<explanation>
{explanation}
</explanation>

Question: beyond representing the writing task and the sentence itself, does \
the explanation show that the model was also thinking about the concept \
"{word}"?

Note: explanations usually describe the writing task and quote or paraphrase \
the sentence being written. That alone is NOT evidence about "{word}" — look \
for content related to "{word}" that goes beyond the task and the sentence.

Score from 0 to 100:
- 0 — nothing in the explanation relates to "{word}"
- 25 — a faint or incidental association
- 50 — clearly related themes or associations
- 75 — strong semantic presence: synonyms or close associates of "{word}"
- 100 — "{word}" itself (or a direct variant of it) appears, or the \
explanation is explicitly about it

First quote your evidence, then give the score. Reply in exactly this format:

Evidence: "<the shortest quote from the explanation that best supports your score>"
Score: <integer between 0 and 100>

If nothing in the explanation relates to "{word}", write `Evidence: none`.
The score must be a bare integer with nothing after it."""

JUDGE_PROMPTS = {"v1": JUDGE_PROMPT_V1}
