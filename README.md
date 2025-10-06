# DT-BIM: Digital Twin-driven BIM Simulation

Reproducible reference implementation accompanying the paper **"Digital-Twin Driven BIM Simulation: Real-time Data for Construction Process Optimization"**.

This repository provides a minimal-yet-complete end-to-end pipeline that mirrors the paper’s framework:
1) **Data Layer** – Synthetic multi-source data generation (IoT sensors, simple "vision" features, mobile inputs).  
2) **Fusion Layer** – Per-source **Extended Kalman Filters (EKF)** and a **federated information fusion** step.  
3) **Decision Layer** – A **TD3** deep reinforcement learning agent for long-horizon multi-objective control.  
4) **Optimization Layer** – A **Mixed-Integer Program (MIP)** for resource allocation (using `pulp`).  
5) **Evaluation** – Metrics for SPA, RUE, CSR, etc., and simple plots/CSV outputs.

> Everything is lightweight and CPU-friendly, using only common open-source packages. No external services or datasets are required.

---

## 1. Repository Structure

```text
dt-bim/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ environment.yml
├─ reproduce.sh
├─ scripts/
│  ├─ run_all.py
│  ├─ simulate_and_fuse.py
│  ├─ train_td3.py
│  └─ run_mip.py
├─ src/dt_bim/
│  ├─ __init__.py
│  ├─ configs/default.yaml
│  ├─ data/synth_dataset.py
│  ├─ fusion/ekf.py
│  ├─ fusion/federated.py
│  ├─ rl/td3.py
│  ├─ rl/replay_buffer.py
│  ├─ mip/resource_mip.py
│  ├─ evaluation/metrics.py
│  └─ utils/{logging_utils.py, seed.py}
├─ data/                      # generated at runtime
│  └─ synthetic/              # CSVs produced by scripts
└─ notebooks/                 # (optional) empty placeholder
```

---

## 2. Installation

### 2.1 Quickstart (pip)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Conda
```bash
conda env create -f environment.yml
conda activate dt-bim
```

---

## 3. How to Reproduce Experiments

End-to-end (sensible defaults, CPU-friendly):
```bash
bash reproduce.sh
```
This runs:
1. `scripts/simulate_and_fuse.py` – generates synthetic data and fuses with EKF + federated algebra.
2. `scripts/train_td3.py` – trains a TD3 policy on the fused states (few thousand steps by default).
3. `scripts/run_mip.py` – solves a small resource allocation instance guided by the learned policy.
4. Writes CSV logs under `data/synthetic/` and prints summary metrics.

To run each stage manually:

```bash
# 1) Data + fusion
python scripts/simulate_and_fuse.py

# 2) RL training (TD3)
python scripts/train_td3.py --steps 5000 --seed 0

# 3) Resource MIP
python scripts/run_mip.py --seed 0
```

Outputs:
- `data/synthetic/fused_states.csv` (state estimates after EKF + fusion)
- `data/synthetic/rl_train_log.csv` (episodic rewards and key metrics)
- `data/synthetic/mip_solution.csv` (resource assignment and schedule)
- `data/synthetic/summary.json` (SPA / RUE / CSR / QCR estimates from the run)

---

## 4. Configuration

A single YAML file controls sizes, noise, EKF covariances, TD3 hparams, reward weights, and MIP sizes:
```bash
cat src/dt_bim/configs/default.yaml
```
Override any value via CLI flags (scripts expose common knobs) or by editing the YAML.

---

## 5. Expected Results (small demo settings)

With default CPU-friendly settings (tiny synthetic world):
- SPA ~ 0.80–0.90 (short horizon)
- RUE ~ 0.70–0.85
- CSR ~ 0.10–0.25 (vs. a simple baseline)
These numbers vary by random seed but should be within these ballparks.

---

## 6. Troubleshooting

- **Slow `torch` install**: try `pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision` or use Conda.
- **`pulp` solver**: the default CBC solver ships with `pulp`. If it fails, set `PULP_CBC_CMD(msg=True)` for diagnostics.
- **Numerical instability in EKF**: reduce process/measurement noise (`q`, `r`) in the YAML.
- **TD3 training plateaus**: increase steps, batch size, or exploration noise; or reduce state/action dims to stabilize.

---

## 8. License

Released under the MIT License (see `LICENSE`).

