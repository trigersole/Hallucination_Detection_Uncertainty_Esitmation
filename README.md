# Controlled Verification for Hallucination Detection

This repository implements the pilot proposed in **“Beyond Generation
Confidence: Controlled Verification for Uncertainty-Based Hallucination
Detection.”** It keeps a causal language model frozen and extracts signals from:

- the original answer (confidence, consistency, and lexical-cluster entropy);
- three factual verification prompts;
- three length-matched non-factual control prompts;
- normalized hidden-state differences `verify - control` at selected layers.

It then trains calibrated linear detectors and compares generation-only,
P(True), verification-state, control-state, concatenation, and difference-state
feature sets.

## Install

Python 3.10+ and a CUDA-capable environment are recommended for 7B models.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Input data

Use JSON Lines with one object per question:

```json
{"id":"q1","question":"What is the capital of France?","answers":["Paris"]}
```

`answers` contains acceptable aliases. Automatic exact-match labeling is useful
for a pilot but must be manually audited for thesis results. Refusals and
ambiguous answers should be handled separately.

## Smoke-test extraction

```powershell
cverify extract --data data/example.jsonl --model Qwen/Qwen2.5-7B-Instruct --output runs/smoke --samples 2 --layers -1 -4 -8
```

For a real pilot, provide at least 20 questions with at least five correct and
five incorrect generated answers, then train the detectors:

```powershell
cverify extract --data data/pilot.jsonl --model Qwen/Qwen2.5-7B-Instruct --output runs/qwen_pilot --limit 1000 --samples 5 --layers -1 -4 -8
cverify train --run runs/qwen_pilot --output runs/qwen_pilot/results --bootstrap 1000
```

Add `--trust-remote-code` only for a model repository you trust. On limited
hardware, choose a smaller causal instruction model; the protocol is unchanged.

The extractor writes:

- `features.npz`: numeric features and labels;
- `metadata.jsonl`: prompts, answers, samples, probabilities, and correctness;
- `manifest.json`: model, layer, seed, and run settings.

The trainer uses a question-level stratified split, fits preprocessing only on
the training portion, calibrates on the development portion, and reports AUROC,
hallucination-positive AUPRC, Brier score, and bootstrap confidence intervals on
the held-out test set. It also reports error at 80% answer coverage and the
prespecified confident/consistent subgroup (defaults: mean token log-probability
at least `-0.5`, consistency at least `0.8`). Override those thresholds with
`--confidence-logprob` and `--consistency`; choose them before inspecting the
test set. Never select layers or prompts using the test set.

The comparison includes an original-answer hidden-state probe, making the key
incremental test `generation features + generation state` versus the full
controlled verifier. HaloScope, TSV, PRISM, and SSP are research baselines with
their own training protocols; integrate their published implementations against
the cached `metadata.jsonl` samples instead of relabeling or regenerating data.

## Important scope notes

- The included entropy groups normalized answer strings. It is a transparent
  **lexical-cluster proxy**, not full semantic entropy. Replace the clusterer
  with bidirectional NLI entailment for a publication-grade Semantic Entropy
  baseline.
- The verifier score is self-verification, not a ground-truth label.
- Hidden coordinates are model-specific; train separate probes per model.
- An equal-cost comparison should account for all generated and scored tokens.

## Full TruthfulQA experiment on Slurm

Export the 817-question generation split once, without the legacy HaloScope
dependency overlay:

```bash
cd /home/msai/siddhart022/Hallucination_Detection_Uncertainty_Esitmation
/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python \
  -m cverify.cli prepare-truthfulqa --output data/truthfulqa.jsonl
```

The included array launcher uses nine independent GPU shards of at most 100
questions, so no single extraction job approaches the six-hour cluster limit.
It uses the existing HaloScope BLEURT-20 checkpoint to label generated answers;
exact match is not appropriate for TruthfulQA's sentence-style answers. Because
this cluster permits only two submitted jobs at once, submit the shards in waves
and wait for each wave to finish before submitting the next:

```bash
sbatch --array=0-1%2 scripts/slurm_truthfulqa_array.sbatch
# Wait until both tasks finish successfully.
sbatch --array=2-3%2 scripts/slurm_truthfulqa_array.sbatch
# Wait, then repeat:
sbatch --array=4-5%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --array=6-7%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --array=8 scripts/slurm_truthfulqa_array.sbatch
```

After shard 8 finishes successfully, submit the CPU merge/training job:

```bash
sbatch scripts/slurm_truthfulqa_train.sbatch
```

Monitor all shards with:

```bash
squeue -u "$USER"
tail -f logs/cverify-tqa-ARRAY_JOB_0.out
```

The merge step rejects duplicate question IDs and mismatched model, layer,
sampling, or labeling settings. Final metrics are written to
`runs/truthfulqa/full/results/metrics.json`. Preserve all nine shard directories
until the merged file has been verified.

Do not queue the next wave or the training job early: pending tasks count toward
the two-job submission limit.

### Complete-sentence answer condition

Keep the original brief-answer run unchanged. After pulling the latest code,
export a separate run name and the new generation prompt in the shell from which
you submit the jobs:

```bash
export CVERIFY_RUN_NAME=truthfulqa_sentence
export CVERIFY_GENERATION_PROMPT=complete-sentence
```

Run shard 0 first as a pilot. Check `runs/truthfulqa_sentence/shard_0/metadata.jsonl`
for complete answers and truncation before committing to the other shards. Keep
the same settings for every shard; the merge command rejects mismatched prompt
profiles and maximum token limits.

```bash
sbatch --export=ALL --array=0 scripts/slurm_truthfulqa_array.sbatch
```

If the pilot looks good, submit the remaining shards in waves, waiting for each
wave to finish successfully before the next one:

```bash
sbatch --export=ALL --array=1 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=2-3%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=4-5%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=6-7%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=8 scripts/slurm_truthfulqa_array.sbatch
```

After shard 8 finishes, merge and train on the prespecified seed-42 split:
512 training, 100 development, and 205 test questions.

```bash
sbatch --export=ALL scripts/slurm_truthfulqa_train.sbatch
```

Results go to `runs/truthfulqa_sentence/full/results_seed_42_val100_test25/`.
The original `runs/truthfulqa/` directory is untouched. The sentence prompt
encourages, but cannot guarantee, complete sentences. BLEURT is still an
imperfect correctness labeler: audit changed labels and ambiguous answers
before treating a detector score as factual hallucination performance.
