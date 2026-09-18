# SparsePOD-Sim2Real

[English](README.md) | [中文说明](README_zh.md)

Few-shot Extreme-Sparse to Full-field Sim2Real PDE Neural Operator Learning Framework with Physics Modal Priors (Targeting ICLR 2027).

---

## 1. Overview & Motivation

Existing Scientific Machine Learning (SciML) and neural operator benchmarks (such as RealPDEBench) predominantly assume full-field dense observations on regular Cartesian grids, supported by abundant real-world training trajectories.

However, in realistic engineering deployments (wind tunnels, aerospace structures, marine vehicles):
* **Extreme Spatial Sparsity**: Full-field optical diagnostics (PIV) are costly and restricted by optical access. In-situ experiments rely on only **16 to 64 discrete probes (0.195% to 0.781% of grid locations)**.
* **Few-shot Scarcity**: Real-world experimental runs are expensive, yielding only **1 to 3 trajectories**.
* **Ill-posed Sparse Forecasting**: Sparse probes leave most spatial degrees of freedom unobserved, so full-field forecasting requires explicit priors and careful empirical comparison.

**SparsePOD-Sim2Real** bridges this gap by decoupling modal geometry from temporal dynamics:
* Extracts global spatial orthonormal coherent structures from dense numerical simulation (CFD).
* Employs **Differentiable Gappy-POD** and **Grassmannian Subspace Warping ($\mathbf{W}_{\text{align}}$)** to lift sparse sensor readings into continuous, physically sound full-field flows.
* Integrates a 3D-UNet residual backbone to model high-frequency nonlinear structure. Accuracy and stability are established by the benchmark experiments rather than assumed by the implementation.

---

## 2. Installation

```bash
conda create -n sparse_pod python=3.10 -y
conda activate sparse_pod

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install package
pip install -r requirements.txt
pip install -e .
```

---

## 3. Data Preprocessing & Fast Tensor Caching

```bash
# Preprocess all 99+99 trajectories, freeze official splits and sensor layouts,
# and build separate development and all-Sim final POD bases.
bash scripts/prepare_server_data.sh \
    data/foil/hf_dataset/real \
    data/foil/hf_dataset/sim \
    data/foil/tensor_cache_64x128 \
    manifests
```

---

## 4. Benchmark Models & Configurations

The main model and baseline configurations are in `configs/yaml_main_v1`; controlled variants are in `configs/yaml_ablations`. The main set includes Masked-UNet3D, Masked-FNO3D, Classical Gappy-POD, Gappy-LinearAR, POD-ResUNet3D-Sparse, and MISF-NO.

---

## 5. Training & Evaluation

### Batch Training
```bash
# Linux / Server:
bash scripts/train_all.sh configs/yaml_main_v1 0 data/foil

# Windows PowerShell:
.\scripts\train_all.ps1 -ConfigDir configs/yaml_main_v1 -Gpu 0 -DataRoot data/foil
```

### Batch Evaluation & Excel Export
```bash
# Linux / Server:
bash scripts/evaluate_all.sh best_checkpoints data/foil 0

# Windows PowerShell:
.\scripts\evaluate_all.ps1 -CkptDir best_checkpoints -DataRoot data/foil -Gpu 0
```

Evaluation reports ID and OOD fluid, sensor, unobserved-region, channel, vorticity and RMSE metrics, frame accounting, and three-round rollout results. Batch mode writes `evaluation_results_*.xlsx` and the matching JSON file.

---

## 6. Smoke Test

Run the end-to-end verification pipeline:
```bash
python scripts/run_smoke_test.py
```
