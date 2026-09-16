# SparsePOD-Sim2Real

[English](README.md) | [中文说明](README_zh.md)

Few-shot Extreme-Sparse to Full-field Sim2Real PDE Neural Operator Learning Framework with Physics Modal Priors (Targeting ICLR 2027).

---

## 1. Overview & Motivation

Existing Scientific Machine Learning (SciML) and neural operator benchmarks (such as RealPDEBench) predominantly assume full-field dense observations on regular Cartesian grids, supported by abundant real-world training trajectories.

However, in realistic engineering deployments (wind tunnels, aerospace structures, marine vehicles):
* **Extreme Spatial Sparsity**: Full-field optical diagnostics (PIV) are costly and restricted by optical access. In-situ experiments rely on only **16 to 64 discrete surface probes (sensor coverage < 0.1% of total DOFs)**.
* **Few-shot Scarcity**: Real-world experimental runs are expensive, yielding only **1 to 3 trajectories**.
* **Operator Collapse**: Standard 23M parameter 3D-UNet and FNO suffer from severe **spatial gradient isolation** and hallucination when fine-tuned on few-shot sparse point observations.

**SparsePOD-Sim2Real** bridges this gap by decoupling modal geometry from temporal dynamics:
* Extracts global spatial orthonormal coherent structures from dense numerical simulation (CFD).
* Employs **Differentiable Gappy-POD** and **Grassmannian Subspace Warping ($\mathbf{W}_{\text{align}}$)** to lift sparse sensor readings into continuous, physically sound full-field flows.
* Integrates high-capacity 23M 3D-UNet backbones to model high-frequency non-linear residuals, achieving provable physics stability and SOTA accuracy.

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
# 1. Preprocess raw Arrow/HDF5 data into downsampled 64x128 .pt tensors
python scripts/preprocess_to_tensors.py \
    --data-root data/foil \
    --resolution 64 128 \
    --num-workers 8

# 2. Extract and precompute POD spatial basis from simulation trajectories
python scripts/compute_sim_pod_basis.py \
    --tensor-dir data/foil/tensor_cache_64x128/numerical \
    --output-file data/foil/pod_basis_64x128.pt \
    --rank 64
```

---

## 4. Benchmark Models & Configurations

* **`Masked-UNet3D`** (`configs/baselines/01_masked_unet3d.yaml`): Official RealPDEBench 23.0M 3D-UNet adapted with input mask channel.
* **`Masked-FNO3D`** (`configs/baselines/02_masked_fno3d.yaml`): 3D Fourier Neural Operator with masked input.
* **`Classical Gappy-POD`** (`configs/baselines/03_classical_gappy_pod.yaml`): Traditional Tikhonov-regularized modal reconstruction.
* **`POD-ResUNet3D-Sparse`** (`configs/our_models/01_pod_res_unet3d_sparse.yaml`): Flagship SOTA combining Gappy-POD physical lifting, Grassmannian $\mathbf{W}_{\text{align}}$ rotation, and 23M UNet3D residual refinement.
* **`MISF-NO`** (`configs/our_models/02_misf_no_latent.yaml`): Dual-Path Gappy Variational Encoder + Latent Dynamics 1D-FNO.

---

## 5. Training & Evaluation

### Batch Training
```bash
# Linux / Server:
bash scripts/train_all.sh configs/our_models/ 0

# Windows PowerShell:
.\scripts\train_all.ps1 -ConfigDir configs/our_models -Gpu 0
```

### Batch Evaluation & Excel Export
```bash
# Linux / Server:
bash scripts/evaluate_all.sh best_checkpoints data/foil 0

# Windows PowerShell:
.\scripts\evaluate_all.ps1 -CkptDir best_checkpoints -DataRoot data/foil -Gpu 0
```

Generates:
* Formatted comparison summary table in console.
* `evaluation_results_YYYYMMDD_HHMMSS.xlsx` containing Rel-L2, RMSE, MAE, R2, Vorticity Rel-L2, KE Error, and MVPE.
* Structured JSON for publication plots.

---

## 6. Smoke Test

Run the end-to-end verification pipeline:
```bash
python scripts/run_smoke_test.py
```
