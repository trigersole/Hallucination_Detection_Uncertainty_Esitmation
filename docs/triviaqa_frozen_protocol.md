# Frozen TriviaQA evaluation protocol

This is a fixed external evaluation following the TruthfulQA exploratory study.
The target dataset uses automatic official exact-match labels at the user's
explicit request. It is not a second manual factual-correctness audit.

## Predeclared dataset

- Hugging Face dataset: `mandarjoshi/trivia_qa`.
- Configuration: `unfiltered.nocontext`; split: `validation`.
- Default subset: 500 unique questions, chosen before generation by a deterministic
  SHA-256 ranking of `42|question_id`.
- Deduplicate normalised question text and exclude normalised overlap with the
  complete source TruthfulQA question set, including excluded source answers.
- Pin the dataset repository revision on first export and record its revision,
  dataset fingerprint, selected-ID hash and JSONL hash in the export manifest.
- No evidence passages, reference answers or answer aliases are supplied to the
  generating LLM or to the assessment prompts.

The dataset schema supplies answer aliases; this run uses `answer.normalized_aliases`.
See the [dataset card](https://huggingface.co/datasets/mandarjoshi/trivia_qa/blob/main/README.md).

## Target labels

The complete generated answer is compared with the official normalized aliases
using the authors' normalization: lower case, underscores to spaces, specified
punctuation to spaces, removal of English articles, and whitespace collapse.
There is no substring acceptance, BLEURT threshold or Unicode casefold/accent
expansion beyond the official rule. See the
[authors' evaluator](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py).

A normalized match is labelled 0 (no benchmark-matching error); a non-match is
labelled 1. A factual sentence containing the right entity can fail this rule.
Conversely, the benchmark compares aliases rather than adjudicating all possible
semantic interpretations. Therefore target labels measure benchmark answer
matching, not comprehensive factual correctness. All 500 exported questions are
retained, including refusals or generation-flagged answers; counts of such flags
are reported rather than silently excluded. If only one class occurs, AUROC is
reported as undefined, not as a meaningful detection result.

## Frozen representation and detectors

- Reuse the source model/tokenizer revision, concise generation instruction,
  chat formatting, assessment-v3 prompts, normalization and extraction positions.
- Use the source initial/retry token ceilings (the completed source run used
  256 and 1024), five sampled answers, temperature 0.7 and seed 42.
- Extract the same layers [-1, -4, -8, 15]; second-pass states remain at the
  assessment cue before generating an assessment.
- Predeclare the all-layer configuration and source seed-42 train/dev split.
- Fit five source-only probes: original-answer state, none, verification, control,
  and control+none concatenation. Reproduce their original source-test AUROC and
  Brier within numerical tolerance before evaluating transfer.
- Fit scalers/probes only on TruthfulQA train and calibrators only on TruthfulQA
  development data. The ensemble is the fixed mean of the three native scores.
- No TriviaQA-based fitting, calibration, model selection or threshold tuning.
- Record source artifact hashes, target artifact hashes and runtime agreement.
  Changed source model/runtime/representation settings are rejected.

This is a test of frozen cross-dataset and label-protocol transfer. The change
from reviewed TruthfulQA labels to TriviaQA exact match confounds a pure
dataset-shift interpretation and must remain explicit in the thesis. TriviaQA
is unseen by detector fitting, but may have appeared during LLM pretraining.

## Execution

From the existing cluster repository:

```bash
git pull --ff-only
bash scripts/run_triviaqa.sh --background
```

The script sets its own separate run name and uses the existing Python/HF cache.
The controller submits at most two jobs, with one CPU per task. GPU extraction
uses one GPU per shard; final evaluation uses CPU only. Question checkpoints
allow interrupted extraction to resume. It bypasses the manual review gate for
this explicitly automatic-label protocol. Source TruthfulQA results are not
modified and no large audited feature copy is created.

```bash
bash scripts/run_triviaqa.sh --status
tail -n 80 logs/triviaqa-controller.log
```

Final outputs:

```text
runs/triviaqa_validation500_frozen_em/external_validation/summary.txt
runs/triviaqa_validation500_frozen_em/external_validation/report.json
runs/triviaqa_validation500_frozen_em/external_validation/predictions.json
```

The evaluation records AUROC, Brier, risk at 80% coverage, fixed-0.5 accuracy and
other available uncertainty measures. Paired 1,000-draw question bootstraps compare
none, concatenation and ensemble against the first-pass detector. Intervals are
conditional and unadjusted for multiple comparisons. Multiple assessment
conditions add forward passes, so the methods are not equal-compute alternatives.

Frozen probe parameters and per-model predictions are saved as small resumable
artifacts. If orchestration code must be updated after every extraction shard
is complete, use the existing finalization-only option with this script:

```bash
bash scripts/run_triviaqa.sh --background --finalize-existing
```

To change subset size before starting, set `CVERIFY_TRIVIAQA_SIZE` and a distinct
`CVERIFY_TRIVIAQA_RUN` name. Once extraction starts, settings and code are immutable
for that run. The default source is `runs/truthfulqa_concise_chat_audit_l15`;
`CVERIFY_TRUTHFULQA_RUN` can explicitly select another completed source workflow.

## Thesis interpretation

Report the exact subset and label definition. Compare source and target results
without choosing a new winner on target data. A successful result supports
transfer to TriviaQA benchmark-matching errors; a weak result documents a limit
under dataset/label shift. Neither outcome justifies a claim that factual
verification is uniquely beneficial or that the generator becomes more truthful.
