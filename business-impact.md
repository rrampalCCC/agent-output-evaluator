# Business Impact Hypothesis

The value hypothesis for the Agent Output Evaluation
Harness consists of the metrics chosen, a baseline estimate, a target estimate, and a
confidence level for each. It is important to note that the estimates are **hypotheses to validate**, not
measured results and the confidence reflects evidence quality.

---

## What the harness is for

It converts a slow, subjective and inconsistent human task, "is this agent's
output good enough to ship?", into a fast, repeatable, documented and reported one. It is a **development-time and pre-release quality gate.**

Three uses drive the value:

1. **Triage / screening**: auto-checks every output for structural,
   completeness, and safety properties so a human reviews only the FAILs and
   PARTIALs instead of re-reading every answer from scratch.
2. **Regression detection**: re-running the suite after a prompt or model
   change instantly surfaces a previously-passing behavior that now breaks.
3. **Evidence-based readiness**: replaces "the agent seems fine" with
   "14/17 pass, 0 unexplained hard failures, readiness: …".

The distinctive (niche) value for a finance/audit context is the **hard-fail
constraint mechanism**: some failures are not "lower quality," they are
disqualifying regardless of how good the rest of the answer is (such as an agent
that writes a clean reconciliation and executes a ledger `UPDATE`). Encoding
those as automatic zeros mirrors how a compliance reviewer actually thinks, 
something a generic, gradient-only eval tool would miss.

---

## Metrics, baselines, and targets

### Metric 1 — Manual review time per evaluation run

- **Definition:** analyst minutes to assess a fixed batch of agent outputs.
- **Baseline (hypothesis):** ~10 min/output reading and checking by hand →
  ~170 min for the current 17-case suite.
- **Target (hypothesis):** human reviews only flagged cases (FAIL/PARTIAL,
  ~20–30%) plus a spot-check sample → **~50–60% reduction** in review minutes
  per run.
- **Confidence: Medium.** The mechanism (screen-then-review) is sound and the
  automated pass is near-instant, but the baseline is an estimate, not a timed
  measurement. Validate by timing one real reviewer with vs. without the
  harness.

### Metric 2 — Failure modes detected before release

- **Definition:** count of distinct defect types (fabrication, guardrail
  breach, missing currency/period, incomplete reconciliation, etc.) caught
  pre-release.
- **Baseline (hypothesis):** ad-hoc manual review catches the obvious cases but
  misses subtle ones (e.g. an unlabeled accounting basis); inconsistent across
  reviewers.
- **Target (hypothesis):** the declared checks + rubric catch a **consistent,
  named set** every run. In the current suite the harness deterministically
  catches hallucination (`inv_004`), a source-system write (`inv_005`), and an
  incomplete/currency-less answer (`inv_006`) — the same way every time.
- **Confidence: Medium-High.** Directly demonstrated on the seeded fail cases
  and fully reproducible. Lower confidence on coverage of *unseen* real-model
  failure types until run against a live model.

### Metric 3 — Repeatability across runs

- **Definition:** variance in the score/verdict for the same output across
  repeated evaluations.
- **Baseline (hypothesis):** human review varies reviewer-to-reviewer and
  day-to-day; an LLM-judge would add sampling noise.
- **Target (hypothesis):** **zero variance** — scoring is pure rule-based
  regex + arithmetic, so identical input yields identical output every time.
- **Confidence: High.** This is a property of the design (deterministic and no
  model in the scoring path), not an estimate. Verifiable by running the same
  suite twice and diffing the reports.

---

## Summary table

| Metric | Baseline (hyp.) | Target (hyp.) | Confidence |
|---|---|---|---|
| Manual review time / run | ~170 min (17 cases) | ~50–60% reduction | Medium |
| Failure modes detected pre-release | Ad-hoc, inconsistent | Consistent, named set every run | Medium-High |
| Repeatability across runs | Reviewer/day variance | Zero variance | High |

---

## Honest limitations (what the numbers do *not* claim)

- **Structure, not truth.** Rule-based checks verify that an output *has* the
  right shape, completeness, and safety properties — not that its facts are
  correct. A confidently wrong figure with a plausible citation can still score
  well. This is the inherent ceiling of rule-based judging without a
  ground-truth key.
- **Validated machinery, pending live data.** The pipeline is proven
  end-to-end, but scores to date are against hand-written stub answers. The
  first real-model results require a live endpoint; the review-time and
  detection-rate numbers should be re-measured then.
- **Pre-release scope.** This evaluates a fixed suite offline. It is not a
  production traffic monitor.

---

## Highest-value next steps (to raise confidence and close the gap)

1. **Ground-truth expected values** in each test case → upgrades
   `factual_accuracy` from "internally consistent" to "actually correct." This
   is the single most important limitation to close.
2. **Hybrid LLM-judge fallback** for the fuzzy dimensions regex can't assess
   (semantic faithfulness of a paraphrase, whether a quoted clause means what
   the agent claims), while keeping the deterministic checks as the backbone.
3. **Regression tracking over time** — store each run and diff against the
   last ("`grd_001` was PASS yesterday, now FAIL"), turning the gate into a
   trend monitor and directly evidencing Metric 3.
4. **Latency / cost capture** — the adapter already records `latency_ms`;
   surfacing it (plus token cost) adds cheap operational signal.
5. **Calibration against human graders** — with a set of expert-graded
   examples, tune the weights/thresholds and report the agreement rate as a
   confidence metric for the harness itself.
