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
- Automatic official whole-answer normalized exact-match labels, as previously
  selected by the user. All selected questions retained; generation flags counted.
  A correct entity embedded in prose can fail exact match. These labels differ
  from reviewed TruthfulQA factual judgments; do not call the label protocols identical.
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

Default separate output: `runs/triviaqa_validation500_replication_em`.
The controller uses one CPU per task, one GPU per extraction shard and CPU-only
final analysis; at most two submitted jobs. Extraction and analysis checkpoints
resume when rerunning the command. No manual review gate is introduced.
Do not update code midway through extraction: extraction settings/code are immutable.

```bash
bash scripts/run_triviaqa_replication.sh --status
tail -n 80 logs/triviaqa-controller.log
cat runs/triviaqa_validation500_replication_em/dataset_replication/summary.txt
```

The summary exists only after all requested analyses complete. Full target
metrics, predictions and split indices are under `full/results_seed_42_*`,
`full/combined_detectors` and `full/matched_pass_detection` in that run.
Preserve the `full`, shard and checkpoint directories.

If an old frozen TriviaQA run was already started, do not launch a competing
controller for it or overwrite its configuration. This default replication uses
a different directory and does not reuse or alter its extraction checkpoints.

## Thesis decision

Compare the predeclared ensemble-minus-first AUROC/Brier effects on both datasets,
including interval widths and individual splits, regardless of outcome. If the
direction recurs, describe a similar observed pattern under the two stated label
protocols. If it does not, report dataset dependence. Neither outcome establishes
universality, a verification-specific benefit, equal-compute superiority or an
improvement in the generated answers themselves.

Dataset/scoring sources and exact prompts are documented in
`triviaqa_frozen_protocol.md` and the recorded run manifest.
