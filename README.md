# Model Compression I Labs: PTQ, QAT, Pruning

Four labs, all NumPy/scikit-learn/ONNX Runtime -- no deep-learning framework needed.
All four were executed end to end during preparation; outputs below are from that run.

| Lab | File | Lecture | What it teaches |
|---|---|---|---|
| 2 | `lab2_quantization.py` | 1 (PTQ) | Affine INT8 from scratch: outliers, per-channel, INT32 accumulate |
| 3 | `lab3_pruning_and_benchmark.py` | 2 (Pruning) | Unstructured vs structured proxy (SQNR); correct benchmarking |
| 6 | `lab6_qat_from_scratch.py` | 1 (QAT) | Fake-quant + Straight-Through Estimator; PTQ vs QAT head-to-head |
| 7 | `lab7_pruning_deep.py` | 2 (Pruning) | Magnitude vs random criteria, one-shot vs iterative, Lottery Ticket |

Labs 2 and 3 are shared with the prior Edge Computing lecture package (same files,
reused here because they are directly relevant to this deeper two-lecture unit).

## Setup
```
pip install numpy scikit-learn onnx onnxruntime
```
Python 3.10+. No GPU needed. Each lab runs in under a minute on a laptop CPU.

## Key results (from the authoring run)
- **Lab 6 (QAT):** at 8-6 bits, PTQ and QAT are statistically indistinguishable (~97%).
  At 2 bits, QAT reaches 89.3% vs PTQ's 47.0% -- a +42 point gap.
- **Lab 7 (criteria):** at 95% sparsity with fine-tuning, magnitude pruning reaches 91.0%
  vs random pruning's 42.8%.
- **Lab 7 (schedule):** at 95% sparsity with the SAME total fine-tuning budget (75 epochs),
  iterative pruning (95.0%) beats one-shot (92.96%).
- **Lab 7 (Lottery Ticket):** winning ticket (mask + original init) reaches 96.67% vs
  95.74% for the same mask with a random re-init -- a real but modest gap at this scale,
  smaller than the dramatic effect reported in the original paper on larger networks.

## Known limitations (share with students)
- All four labs use small models and small data (sklearn digits, a small heavy-tailed
  synthetic layer). The qualitative patterns are robust; the exact percentages are not
  meant to generalize to ImageNet-scale models.
- Lab 6's QAT implementation uses a FIXED clip range (set once from initial weights).
  An earlier version recomputed the range from the current max weight every step, which
  made 2-bit QAT training unstable and gave a nonsensical result (QAT worse than PTQ).
  This is documented in the file's header as a worked lesson in verifying surprising
  results before trusting them.
- Lab 7's iterative-pruning schedule similarly had a sparsity-accounting bug during
  development (each round's threshold compounded against the wrong population, causing
  sparsity to overshoot every round). Also documented in the file header.
- The Lottery Ticket effect size in Lab 7 is small and single-seed; run it over several
  seeds (Exercise in Lecture 4) before treating the direction as conclusive at this scale.
