# SALSA_SAFE
### Learning Modular Arithmetic with Universal Transformers

An educational, research-quality PyTorch project that reproduces the **machine
learning and software engineering machinery** of *"SALSA: Attacking Lattice
Cryptography with Transformers"* — Universal Transformers, sequence-to-sequence
learning, tokenization, training infrastructure, evaluation, and visualization —
applied to **generic, benign modular-arithmetic tasks**.

> **Scope note.** This repository does **not** implement, reproduce, or enable
> any cryptanalytic attack, LWE secret-recovery algorithm, or lattice-hardness
> exploit. Every dataset here is a deterministic function of publicly known
> inputs (`(a + b) mod n`, `(a * b) mod n`, etc.) — there is no hidden secret
> to recover, no error distribution hiding a cryptographic key, and no attack
> pipeline anywhere in this codebase. See `data/generators/modular_tasks.py`
> for an explicit discussion of why each task is structurally unrelated to
> LWE, even the "noisy" variant. `ModularInverseTask` additionally enforces a
> hard-coded small-modulus ceiling in code (not just documentation) to keep
> it a toy group-theory exercise.

---

## What's inside

| Module | Contents |
|---|---|
| **1. Data Generation** | 6 synthetic modular-arithmetic tasks (addition, subtraction, multiplication, exponentiation, toy inverse, noisy-label robustness), fully deterministic and reproducible |
| **2. Tokenizer & Preprocessing** | Digit-level vocabulary, PyTorch `Dataset`/`DataLoader`, padding + attention-mask construction |
| **3. Universal Transformer** | Multi-head attention, positional + timestep encoding, shared-weight recurrent encoder/decoder, optional Adaptive Computation Time (ACT), greedy decoding |
| **4. Training Pipeline** | AdamW + warmup/inverse-sqrt LR schedule, mixed precision (AMP), gradient clipping, checkpointing, early stopping, TensorBoard + JSON metrics logging |
| **5. Evaluation & Visualization** | Token/exact-sequence accuracy, operand-magnitude error breakdown, matplotlib training-curve plots |
| **6. Notebooks, Docker, Docs** | 4 runnable Jupyter notebooks, Dockerfile + docker-compose, this README |

Every module has an accompanying `tests/` file; **42 tests pass** across the whole repository.

---

## Project structure

```
SALSA_SAFE/
├── configs/                       # ExperimentConfig dataclasses + YAML configs
│   ├── config.py
│   ├── mod_addition.yaml
│   └── smoke_test.yaml
├── data/
│   ├── generators/                 # BaseModularTask + 6 concrete task generators
│   ├── tokenizer/                  # Vocabulary (token <-> ID)
│   ├── preprocessing/              # Dataset, collate_fn, DataLoader factory
│   └── processed/                  # Generated .jsonl datasets (gitignored)
├── models/
│   ├── layers/                     # PositionalEncoding, MultiHeadAttention, FeedForward, Embedding
│   └── universal_transformer/      # ACT, Encoder, Decoder, full Seq2Seq model
├── training/                       # Trainer, LR scheduler, checkpointing, logging
├── evaluation/                     # evaluate_model, operand-magnitude error analysis
├── metrics/                        # accuracy metrics, resource tracking, metrics history
├── visualization/                  # matplotlib plotting functions + saved plots
├── notebooks/                      # 4 runnable demo notebooks
├── scripts/                        # CLI entry points (generate_data / train / evaluate / visualize_training)
├── tests/                          # pytest suite (47 tests)
├── docker-compose.yml / Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart

### Option A: local Python environment

```bash
git clone <this-repo> && cd SALSA_SAFE
pip install -r requirements.txt

# 1. Generate datasets for every task
python scripts/generate_data.py --task all --modulus 97 --num_samples 50000

# 2. Train a Universal Transformer on modular addition
python scripts/train.py --config configs/mod_addition.yaml

# 3. Evaluate the best checkpoint on the held-out test set
python scripts/evaluate.py \
    --checkpoint checkpoints/salsa_safe_mod_addition/checkpoint_best.pt \
    --config configs/mod_addition.yaml

# 4. Plot training curves
python scripts/visualize_training.py \
    --history checkpoints/salsa_safe_mod_addition/metrics_history.json \
    --output_dir visualization/plots/salsa_safe_mod_addition

# 5. Watch training live
tensorboard --logdir logs

# 6. Run the test suite
python -m pytest tests/ -v
```

### Option B: Docker

```bash
docker compose run generate     # generate all datasets
docker compose run train        # train on mod_addition
docker compose run evaluate     # evaluate the best checkpoint
docker compose run notebook     # Jupyter Lab -> http://localhost:8888
docker compose run tensorboard  # TensorBoard -> http://localhost:6006
```

### Option C: Notebooks

Open `notebooks/01_dataset_generation.ipynb` through `04_visualization.ipynb` in
order — each one is runnable top-to-bottom and was executed end-to-end while
building this repo (outputs are saved in the `.ipynb` files themselves).

---

## Theory summary

**Universal Transformer (Dehghani et al., 2019).** Instead of stacking `L`
distinct layers, a Universal Transformer applies **one shared transition
function** repeatedly across `T` recurrent depth steps, injecting a
sinusoidal *timestep encoding* at each step (analogous to positional
encoding, but indexed by iteration instead of sequence position). This gives
parameter efficiency and, with **Adaptive Computation Time (ACT)**, lets
different token positions use different amounts of computation — harder
positions (e.g. a multiplication digit needing several carries) can "think
longer" than easy ones.

**Modular arithmetic as a seq2seq task.** Each integer is tokenized as a
fixed-width sequence of digit tokens in a configurable base; operands are
joined with a `<sep>` token into one source sequence, and the model is
trained with teacher forcing to autoregressively emit the digits of the
result. This mirrors "transformers learn arithmetic" literature (Lample &
Charton; Nogueira et al.) — a well-established, purely ML line of research.

**Why modular arithmetic specifically?** It gives a *deterministic ground
truth* (unlike NLP tasks), *tunable difficulty* (modulus size, operation
type), and lets us measure **exact-sequence accuracy** — did the model get
the *whole* answer right, not just some digits — which is a much more honest
signal than token-level accuracy alone.

Full theory write-ups (with the specific mathematical formulas) are in the
docstrings at the top of each module — see especially
`data/generators/base_task.py`, `models/layers/positional_encoding.py`,
`models/universal_transformer/act.py`, and `training/scheduler.py`.

---

## Metrics explained

| Metric | Meaning |
|---|---|
| `token_accuracy` | % of individual digit tokens predicted correctly (teacher-forced) — generous, partial credit |
| `exact_sequence_accuracy` | % of *entire* answers correct under greedy decoding — the metric that actually matters |
| `loss` | Cross-entropy over non-pad target tokens |
| `learning_rate` | Warmup + inverse-sqrt schedule value |
| `runtime_seconds` / `gpu_peak_memory_mb` / `cpu_memory_mb` | Wall-clock and memory cost per epoch — diagnostics, not correctness |

A gap between high token accuracy and low exact-sequence accuracy is
expected early in training (getting most digits right ≠ getting the whole
number right) and should close as training progresses.

---

## Configuration

All hyperparameters live in YAML files under `configs/`, loaded via
`ExperimentConfig` (`configs/config.py`). Key knobs:

```yaml
model:
  d_model: 128        # embedding/hidden dimension
  num_heads: 8         # attention heads
  d_ff: 512            # feed-forward hidden dimension
  num_steps: 6         # Universal Transformer recurrent depth
  use_act: false        # enable Adaptive Computation Time

train:
  num_epochs: 20
  learning_rate: 0.001  # peak-LR multiplier for the warmup schedule
  warmup_steps: 1000
  use_amp: true          # mixed precision (CUDA only)
  early_stopping_patience: 5
```

`configs/smoke_test.yaml` is a fast, tiny-model config for pipeline
verification; `configs/mod_addition.yaml` is a more realistic starting point
for actually training a model to convergence.

---

## Extending the project

- **New task:** subclass `BaseModularTask` in `data/generators/modular_tasks.py`,
  implement `sample_operands()` and `compute()`, register it in `TASK_REGISTRY`.
- **New architecture variant:** the `UniversalTransformerEncoder`/`Decoder`
  classes accept `use_act` as a constructor flag — fixed-depth and ACT modes
  share the same underlying layer, so experimenting with recurrence depth or
  halting thresholds doesn't require touching the layer code itself.
- **Multi-GPU:** wrap `model` in `torch.nn.parallel.DistributedDataParallel`
  before constructing `Trainer`, and launch with `torchrun` — the training
  loop itself is DDP-agnostic since all reductions happen inside the wrapped
  module.

---

## Requirements

- Python 3.11+
- PyTorch 2.2+
- See `requirements.txt` for the full list (matplotlib, tensorboard, pyyaml,
  psutil, jupyterlab, pytest, etc.)

## Running tests

```bash
python -m pytest tests/ -v
```

Expected: **42 passed**, covering dataset correctness (mathematical spot
checks against Python's own arithmetic), tokenizer/padding/masking
correctness (including tests that actually perturb masked values to prove
masks work, not just check shapes), model gradient flow, ACT diagnostics,
checkpoint round-tripping, and plot generation.
