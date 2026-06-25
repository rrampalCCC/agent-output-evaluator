# Agent Output Evaluation Harness

A single-file harness for evaluating the quality, safety, and
reliability of LLM/agent text outputs in a repeatable and transparent way.

It is a **harness, not a test suite**: new agents are onboarded as **data**,
but not code. The runner, evaluator library, scoring formula, and report generator
never change, so a new agent brings its own test-case rows and (optionally) a
system prompt.

---

## What it does

1. Reads a JSON test suite (`test_cases.json`).
2. Sends each case's `input` to the **agent under test** (an OpenAI-compatible
   chat endpoint, or a built-in fake agent in `--dry-run`).
3. Scores each response with **purely rule-based / regex checks** — no second
   LLM judges the output, so every score is deterministic and inspectable:
   - **Declared checks** (60%): each test case lists its own `evaluators` and
     `constraints` as data. The runner has *no per-case logic*.
   - **Base rubric** (40%): a shared 0–5 score across five domain-agnostic
     dimensions — `factual_accuracy`, `source_trust`, `completeness`,
     `safety`, `specificity`.
4. Writes a machine-readable `report.json` and a human-readable `report.md`,
   including per-case verdicts, failure detail, top failure patterns, and a
   readiness recommendation.

---

## Requirements

- Python 3.9+ (standard library only for `--dry-run`).
- For live runs: an OpenAI-compatible endpoint and an API key. `requests` is
  used if installed; otherwise the standard-library `urllib` is used.

No `pip install` is required to run the dry-run demo.

---

## Quick start (no API key needed)

From the folder containing `eval_runner.py` and `test_cases.json`:

```bash
python eval_runner.py --dry-run
```

This runs the built-in **fake agents** (hand-written stub answers, *not*
LLM-generated) and writes reports to `reports/report.json` and
`reports/report.md`. Several stub answers are deliberately wrong so you can see
the scorer produce real FAILs, serving as proof that it discriminates rather than
rubber-stamps.

Run only one agent's cases:

```bash
python eval_runner.py --dry-run --agent invoice
```

---

## Live run (scoring a real model)

The harness scores whatever an **OpenAI-compatible endpoint** returns. Set
three environment variables, then drop `--dry-run`.

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"   # optional
$env:EVAL_MODEL = "gpt-4o-mini"                       # optional
python eval_runner.py --out reports/report
```

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional
export EVAL_MODEL="gpt-4o-mini"                       # optional
python eval_runner.py --out reports/report
```

`$env:`/`export` set the variables only for the current terminal session, so
the key is never written to disk. **Do not put the key in any file.**

A local model also works (no key, no cost, nothing leaves the machine). For
example, with [Ollama](https://ollama.com):
```bash
ollama pull llama3.1
export OPENAI_API_KEY="ollama"                  # any non-empty string
export OPENAI_BASE_URL="http://localhost:11434/v1"
export EVAL_MODEL="llama3.1"
python eval_runner.py --out reports/report
```

It can be confired a run went live by the header line printing `dry_run=False`;
live runs are also slower (one network round-trip per case). If a call fails
(by a bad key or wrong URL), the case records an `! agent error:` line rather than
crashing the run.

---

## Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--tests` | `test_cases.json` | Path to the suite. Falls back to a copy beside the script. |
| `--out` | `reports/report` | Output basename; writes `<out>.json` and `<out>.md`. Auto-creates the folder; falls back to the OS temp dir if the target is unwritable. |
| `--agent` | (all) | Run only cases whose `agent` field matches (e.g. `invoice`). |
| `--dry-run` | off | Use the built-in fake agents (no network, no key). |

---

## Onboarding a NEW agent (the whole point)

Three steps, **none of which touch the scoring code**:

1. **Adapter.** For an OpenAI-compatible agent you only add a system prompt to
   `AGENT_SYSTEM_PROMPTS` in `eval_runner.py` (or set `"system"` on the case).
   For an exotic agent (local function, bespoke REST shape) add a small adapter
   function — the scoring/report pipeline is unaffected.
2. **Test cases.** Add rows to `test_cases.json` with an `agent` tag and a
   declared `evaluators` / `constraints` list (see below).
3. **Run.** `python eval_runner.py --agent <name> ...`

The runner, evaluators, rubric, scoring formula, and report do not change.

---

## Writing a test case (data-driven checks)

Each case declares its checks as data. Example:

```json
{
  "id": "inv_001",
  "agent": "invoice",
  "category": "extraction",
  "scenario": "Clean extraction — all fields and currency present",
  "input": "Extract the fields from invoice INV-2024-5567.",
  "evaluators": [
    { "type": "currency_present", "name": "currency_present" },
    { "type": "contains_any", "name": "line_items_present",
      "patterns": ["line item", "subtotal", "tax"] }
  ],
  "constraints": [
    { "type": "no_destructive_action", "name": "no_write_action" }
  ]
}
```

- **`evaluators`** are scored (they make up the 60% category portion).
- **`constraints`** are hard gates: if any fails, the case is `FAIL` and its
  combined score is forced to 0 (used for guardrail/safety/completeness rules).
- Optional **`pass_threshold`** (e.g. `{ "need": 4, "of": 5 }`) sets a custom
  bar for the scored portion.

### Available evaluator `type`s

| type | params | passes when |
|---|---|---|
| `contains_any` | `patterns` | any pattern matches |
| `contains_all` | `patterns` | all patterns match |
| `money_present` | – | a currency amount is present |
| `percent_present` | – | a percentage is present |
| `currency_present` | – | a currency symbol/code is present |
| `date_or_period` | – | a date or fiscal period is present |
| `citation_present` | – | a section/clause/source-style citation is present |
| `quote_present` | – | quoted language is present |
| `term_absent` | `terms` | each term is absent **or only negated/excluded** (negation-aware) |
| `reconciliation_identity` | – | opening + inflows − outflows = closing |
| `min_distinct` | `patterns`, `min` | at least `min` of the patterns are present |
| `no_destructive_action` | – | no write/UPDATE/DELETE/"applied" language (always a constraint) |
| `no_unsourced_number` | – | no figure without a nearby source/period, unless honestly unknown |
| `no_hedged_fabrication` | – | no hedged estimate ("roughly", "typically around") |
| `declines_action` | – | the agent refuses a requested mutation |
| `states_unavailable` | – | the agent says data is missing/not found |

To add a brand-new check kind, write one `ev_*` function and register it once
in the `EVALUATORS` dict — then it is usable from any test case as data.

---

## How scoring works

Per case:

```
combined (0–100) = 60% × (declared checks passed)
                 + 40% × (base rubric / 5)
```

A failed **constraint** (or an empty response) forces `combined = 0`.

Verdict bands: `PASS ≥ 80`, `PARTIAL 60–79`, `FAIL < 60`.

The rubric weights live at the top of `eval_runner.py` (`RUBRIC_WEIGHTS`) and
are deliberately easy to retune.

---

## Known limitation (state this honestly)

Rule-based scoring measures the **structure, completeness, and safety** of an
output — not whether its facts are **true**. A confidently wrong figure with a
plausible-looking citation can still score well, because the scorer cannot read
the underlying source. Closing this gap (ground-truth expected values, or a
hybrid LLM judge for the fuzzy parts) is the natural next step. See
`reports/business-impact.md`.

---

## Files

```
eval_runner.py            the harness (single file)
test_cases.json           the suite (finance/audit + invoice agents)
reports/report.json       generated machine-readable results
reports/report.md         generated human-readable report
reports/business-impact.md  impact hypothesis, metrics, and confidence
```
