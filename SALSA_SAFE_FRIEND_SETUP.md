# SALSA_SAFE.zip — Run on a Friend's Laptop

Your zip already contains generated datasets (`data/processed/`) and one
trained checkpoint (`checkpoints/smoke_test_mod_addition/`), so your friend
can skip straight to evaluation if they just want to see results quickly —
or retrain from scratch if they want the full experience.

---

## Step 0: Send the file

Send `SALSA_SAFE.zip` to your friend (USB drive, cloud link, chat app — any
of these work, it's just a folder of code + small data files).

---

## Step 1: Unzip

### Windows (PowerShell)
```powershell
Expand-Archive SALSA_SAFE.zip -DestinationPath D:\code\SALSA -Force
cd D:\code\SALSA\SALSA_SAFE
```

### macOS / Linux
```bash
unzip SALSA_SAFE.zip -d ~/code/SALSA
cd ~/code/SALSA/SALSA_SAFE
```

---

## Step 2: Install Python 3.11+ (if not already installed)

Check first:
```bash
python --version
```
If missing or older than 3.11, download from https://www.python.org/downloads/
(Windows: tick "Add Python to PATH" during install).

---

## Step 3: Create and activate a virtual environment

### Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
You should see `(venv)` appear at the start of the terminal prompt.

---

## Step 4: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This takes a few minutes (PyTorch is the largest download, ~700MB–2GB
depending on OS/CUDA).

### Optional — GPU (CUDA) build instead of CPU-only
Only if their laptop has an NVIDIA GPU with drivers installed:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 5: Verify the install — run the test suite

```bash
python -m pytest tests/ -v
```
Expected: **`42 passed`**. If this fails, something went wrong in Step 4 —
re-check the venv is activated and requirements installed cleanly.

---

## Step 6A: Fastest path — just evaluate the checkpoint already in the zip

Since `checkpoints/smoke_test_mod_addition/checkpoint_best.pt` is already
included, they can evaluate it immediately without training anything:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/smoke_test_mod_addition/checkpoint_best.pt \
    --config configs/smoke_test.yaml
```

Then plot its training curves:
```bash
python scripts/visualize_training.py \
    --history checkpoints/smoke_test_mod_addition/metrics_history.json \
    --output_dir visualization/plots/smoke_test_mod_addition
```
PNG plots will appear in `visualization/plots/smoke_test_mod_addition/`.

---

## Step 6B: Full path — regenerate data and train from scratch

If they want to reproduce everything themselves (or try a different task):

```bash
# Generate all 6 modular-arithmetic datasets
python scripts/generate_data.py --task all --modulus 97 --num_samples 50000

# Train (quick smoke test, few minutes)
python scripts/train.py --config configs/smoke_test.yaml

# OR a fuller training run
python scripts/train.py --config configs/mod_addition.yaml
```

---

## Step 7: Watch training live with TensorBoard (optional, separate terminal)

```bash
tensorboard --logdir logs
```
Open `http://localhost:6006` in a browser.

---

## Step 8: Run the notebooks (optional)

```bash
jupyter lab notebooks
```
Open in order: `01_dataset_generation.ipynb` → `02_training.ipynb` →
`03_evaluation.ipynb` → `04_visualization.ipynb`.

---

## Step 9: Docker route (alternative to Steps 2–4)

Only if they have Docker Desktop installed:
```bash
docker compose run generate
docker compose run train
docker compose run evaluate
docker compose run notebook      # -> http://localhost:8888
docker compose run tensorboard   # -> http://localhost:6006
```

---

## Step 10: When finished

```bash
deactivate
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `requirements.txt` not found | Make sure they `cd`'d into the `SALSA_SAFE` folder (Step 1) |
| `ModuleNotFoundError` | venv not activated — repeat Step 3, then Step 4 |
| `.jsonl` file not found | Only needed if they skip Step 6A and want to retrain — run Step 6B first |
| Old checkpoint / history mismatch errors | Delete the folder and regenerate: `rm -rf checkpoints/<name> logs/<name>` then retrain |
| Training slow on CPU-only laptop | Normal — `configs/smoke_test.yaml` is designed to finish in ~1 minute on CPU |
| `python` not recognized (Windows) | Reinstall Python and check "Add Python to PATH", or use `python3`/`py` instead |
