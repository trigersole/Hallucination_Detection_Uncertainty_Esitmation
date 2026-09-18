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
