# Deterministic Right-to-Forget (RTF) Microbenchmark

This repository contains a minimal, deterministic pipeline that demonstrates “right to forget” (RTF) mechanics via micro-batch logging (WAL), deterministic replay with forget filtering, a dense-delta ring buffer demo, and an audit harness. Running one script produces a small set of CSV results that are cited in the paper.

- Script: `rtf_pipeline.py`
- Outputs (under `results/`):  
  - `exactness.csv` — columns: `max_abs_diff`, `exact_pass`  
  - `wal_overhead.csv` — columns: `bytes_per_record`, `records`, `total_bytes`  
  - `ring_reverts.csv` — columns: `dense_delta_per_step_bytes`, `window_N`, `compression_ratio_hint`  
  - `audits.csv` — columns: `model`, `ppl_retain`, `mia_auc`, `canary_exposure_mu`, `canary_exposure_sd`, `targeted_extract_pct`  
- Artifacts (under `artifacts/`): checkpoints and the binary WAL (`train.wal`).

## 1) Quickstart

### Prereqs
- OS: Windows 10 or Linux/macOS
- Python: **3.8–3.10** (tested on 3.8.18)
- CPU-only is fine; no GPU required.

### Create env and install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Install pinned deps
pip install --upgrade pip
# Install PyTorch CPU wheel first (important for Windows/CPU):
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
# Then the rest:
pip install -r requirements.txt
```

### Run
```bash
# Default: synthetic data, 300 steps, checkpoint at 150
python rtf_pipeline.py

# Faster smoke test (smaller run):
python rtf_pipeline.py --fast

# Optional: tiny real data (requires internet to fetch a wikitext-2 slice)
python rtf_pipeline.py --no-synthetic
```

On success you’ll see:
```
=== DONE ===
- results written under: results
- artifacts under: artifacts
Files you’ll cite in Results: exactness.csv, wal_overhead.csv, ring_reverts.csv, audits.csv
```

## 2) What each CSV means

- `results/exactness.csv`  
  - `max_abs_diff`: maximum absolute parameter difference between replay-filtered model and oracle retrain.  
  - `exact_pass`: boolean; `True` when byte-equal in training dtype on many setups, else `False` with small epsilon.

- `results/wal_overhead.csv`  
  - Fixed-width 32-byte records per microbatch and aggregate size.

- `results/ring_reverts.csv`  
  - Illustrative dense-delta per-step size (training dtype), `window_N`, and a typical compression ratio hint.

- `results/audits.csv`  
  - Utility + leakage metrics used in the paper: retain PPL, membership-inference AUC, canary exposure stats, and a simple targeted extraction rate.

## 3) Reproducibility & determinism

This pipeline hard-enables deterministic algorithms, fixes all RNGs, and logs per-microbatch seeds/LR/schedule digests. You don’t need to set env vars manually; the script sets the relevant flags at startup. *(See `set_determinism()` and the WAL record format in the code.)*

**Known-good environment**  
`env_pins.json` captures a working configuration (Windows 10, Python 3.8.18, Torch 2.2.1+cpu, Transformers 4.46.3, etc.) and its SHA256:

- `env_pins.json` SHA256: `1b6b4cc915cd20d77cd354319931d1d488e3c194a40624c66a7144f5ac29f4f0`

Reviewers can use this file to sanity-check their setup before running.

**Determinism smoke test (optional)**
```bash
# run twice with same seed, compare the CSVs
python rtf_pipeline.py --fast --seed 1337
cp -r results results_run1   # use 'copy' on Windows
python rtf_pipeline.py --fast --seed 1337
# Compare
diff -u results_run1/exactness.csv results/exactness.csv || true
diff -u results_run1/wal_overhead.csv results/wal_overhead.csv || true
diff -u results_run1/audits.csv results/audits.csv || true
```

## 4) Folder layout

```
.
├─ rtf_pipeline.py
├─ requirements.txt
├─ LICENSE
├─ README.md
├─ env_pins.json              # environment snapshot (optional)
├─ artifacts/                 # created on first run
│  ├─ ckpt_k.pth
│  ├─ ckpt_T.pth
│  └─ train.wal
└─ results/                   # created on first run
   ├─ exactness.csv
   ├─ wal_overhead.csv
   ├─ ring_reverts.csv
   └─ audits.csv
```

## 5) Troubleshooting

- **Torch install on CPU/Windows:** Use the `--index-url` command above to grab the CPU wheel.
- **No internet:** Leave `--synthetic` as default; tiny real data requires internet for the `datasets` hub.
- **Version drift:** If results differ, check your Python/Torch/Transformers versions and compare to `env_pins.json`.

## 6) How to cite
If you use this code or the audit artifacts, please cite the accompanying paper. A `CITATION.cff` can be added on request.

## 7) License
Apache-2.0 (see `LICENSE`).
