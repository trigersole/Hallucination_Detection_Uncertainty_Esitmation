# TriviaQA independent replication

This mode repeats the detector experiment within TriviaQA. It answers whether
second-pass versus first-pass improvements recur when detectors are trained and
evaluated within a second dataset. The older `run_triviaqa.sh` default tests
TruthfulQA-trained detector transfer, a different question.

## Fixed protocol before seeing results

- Deterministic 500-question subset of TriviaQA `unfiltered.nocontext` validation,
  seed 42 hash ranking, deduplicated and excluding TruthfulQA question overlap.
  Dataset revision and exported question hashes are recorded.
- Same pinned Llama model/tokenizer revision and runtime as the completed source.
- Initial instruction: **Answer the question concisely.** Same chat template,
  token ceilings (256 initially, retry up to 1024), five sampled answers at
  temperature 0.7, generation seed 42 and answer-state extraction position.
- Same second-pass factual verification, formatting control and none prompts
  from assessment-v3. Extract at the assessment cue before any assessment output.
  No references or answer aliases enter these prompts.
- Same layers `[-1, -4, -8, 15]` and per-layer L2 normalization.
- Primary labels: reviewed factual correctness, using the same answer-bound
  `export_review` / `apply_review` functions as TruthfulQA. Extraction labels are
  deferred. Every answer must receive a judgment and reviewer before fitting.
  Only correct/incorrect judgments without generation-quality flags enter the
  binary detector experiment; ambiguous, refusal and exclude are reported and
  omitted. This matches the TruthfulQA implementation, including quality exclusions.
- Official whole-answer exact match is a supplementary generator benchmark score
  in `supplementary_exact_match.json`. It never supplies primary training labels
  and is not prefilled into review judgments. Correct prose can fail exact match.
- Independently fit on TriviaQA: stratified train/dev/test with source test size
  0.25, dev fraction of remaining rows 0.1633986928, split seeds 42 through 46.
  Same source C (0.1), balanced logistic regression, train-only scaler and
  dev-only Platt calibration. No TruthfulQA-trained detector used in this mode.
- Repeat original-three, layer-15 and all-layer baseline/combined analyses.
  The primary matched comparison fixes all four layers: first-pass state,
  native none, control+none concatenation and fixed native probability ensemble.
  No choice based on TriviaQA test results. Other configurations are descriptive.
- Report AUROC, Brier, fixed-0.5 accuracy, balanced accuracy and selective risk;
  paired 1,000-draw bootstrap intervals and mean/SD across the five split seeds.
  The seeds are overlapping partitions, not five independent generation runs.

## Cluster commands

```bash
git pull --ff-only
unset CVERIFY_TRIVIAQA_RUN CVERIFY_TRIVIAQA_SIZE
bash scripts/run_triviaqa_replication.sh --background
```

Default separate output: `runs/triviaqa_validation500_replication_reviewed`.
The controller uses one CPU per task, one GPU per extraction shard and CPU-only
final analysis; at most two submitted jobs. Extraction and analysis checkpoints
resume when rerunning the command. It deliberately pauses at `needs_review`.
Do not update code midway through extraction: extraction settings/code are immutable.

```bash
bash scripts/run_triviaqa_replication.sh --status
tail -n 80 logs/triviaqa-controller.log
cat runs/triviaqa_validation500_replication_reviewed/finalize_status.json
```

## Review and resume

When extraction completes, review this file on the cluster:

```text
runs/triviaqa_validation500_replication_reviewed/review.jsonl
```

For each row, set `judgment` to one of `correct`, `incorrect`, `ambiguous`,
`refusal`, or `exclude`; enter the reviewer identifier in `reviewer` and useful
evidence/rationale in `notes`. Do not change IDs, question/answer text, references
or sample hashes. Keep answers, references and judgments separate from detector
scores; review before inspecting detector results or split assignments.

Judge whether the complete answer factually answers the question. A truthful
sentence containing the right entity can be correct even if it fails EM. A
correct entity accompanied by a materially false claim is not automatically
correct. Treat references as guidance; record independent evidence for uncertain
cases. Use ambiguous for unresolved factual correctness, refusal for abstention,
and exclude for unusable items. Do not convert uncertainty into an arbitrary
binary label. Record who reviewed and whether independent agreement was assessed.

After all rows are reviewed, resume with exactly the same command:

```bash
bash scripts/run_triviaqa_replication.sh --background
```

This keeps existing generated answers and states, applies reviewed labels and
runs the complete detector comparison. The reviewed data are an indexed view of
the original features rather than another large copy. Counts and exclusions are
saved in `review.summary.json`; remaining review issues are in `finalize_status.json`.

Final result to share:

```bash
cat runs/triviaqa_validation500_replication_reviewed/dataset_replication/summary.txt
```

The summary exists only after all requested analyses complete. Full target
metrics, predictions and split indices are under `audited/<review-hash>/results_seed_42_*`,
`audited/<review-hash>/combined_detectors` and `audited/<review-hash>/matched_pass_detection`.
Preserve the `full`, shard and checkpoint directories.

If an old frozen TriviaQA run was already started, do not launch a competing
controller for it or overwrite its configuration. This default replication uses
a different directory and does not reuse or alter its extraction checkpoints.

## Thesis decision

Compare the predeclared ensemble-minus-first AUROC/Brier effects on both datasets,
including interval widths and individual splits, regardless of outcome. If the
direction recurs, describe a similar observed pattern on two datasets under the
same factual-review protocol. If it does not, report dataset dependence. Shared
review criteria do not establish independent reviewer agreement or eliminate
label error. Neither outcome establishes
universality, a verification-specific benefit, equal-compute superiority or an
improvement in the generated answers themselves.

Dataset/scoring sources and exact prompts are documented in
`triviaqa_frozen_protocol.md` and the recorded run manifest.
