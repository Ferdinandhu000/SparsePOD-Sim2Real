# SparsePOD-Sim2Real: 少样本极端稀疏到全场流体 Sim2Real 迁移学习基准与神经算子

[English](README.md) | [中文说明](README_zh.md)

面向 **ICLR 2027**，构建针对真实工程物理场景的 **少样本极端稀疏到全场流体（Few-shot Extreme-Sparse to Full-field）Sim2Real 预测基准与 SOTA 模型库**。

---

## 一、 项目背景与核心痛点重构

现有前沿偏微分方程神经算子基准（如 RealPDEBench）普遍假定：
1. **全场网格密集观测**：真实世界与数值模拟一样具备全空间 $64 \times 128$ 规则网格；
2. **海量真实轨迹**：拥有数十条长序列真实物理轨迹（上万个训练窗口样本）。

但在真实工业现场（如风洞测试、水下航行体、航空机翼）：
* **极度稀疏（Extreme Spatial Sparsity）**：全场 PIV 光学诊断昂贵且易受遮挡，现场仅能布置 **16 ~ 64 个离散表面压力探针或风速探头（测点数 $< 0.1\%$ 全场自由度）**；
* **极度缺乏（Few-shot Scarcity）**：真实物理实验成本极高，通常仅能获取 **1 ~ 3 条有效演化轨迹**；
* **算子失效**：传统 23M 参数量的 3D-UNet 与 FNO 在极端稀疏少样本下，因**空间梯度孤立效应**产生严重的“空间幻觉”，未布设传感器的 99.9% 区域迅速崩塌，长程自回归发散。

**本项目提出基于物理模态先验（POD/SVD）与大容量神经算子正交解耦的 SOTA 解决方案**：
* 借助仿真全场数据离线提取空间正交相干结构；
* 通过**可微 Gappy-POD 投影**与**格拉斯曼流形子空间对齐（$\mathbf{W}_{\text{align}}$）**，瞬时将稀疏点升维为连续物理粗场；
* 接入高参数量 3D-UNet 专门建模非线性高频残差，在保证少样本绝对物理稳定性的同时达到全新 SOTA 精度！

---

## 二、 环境配置

```bash
# 1. 创建并激活 Python 3.10 环境
conda create -n sparse_pod python=3.10 -y
conda activate sparse_pod

# 2. 安装 PyTorch (以 CUDA 11.8 为例)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖并以开发模式注册
pip install -r requirements.txt
pip install -e .
```

---

## 三、 数据准备与高速张量缓存

为避免服务器在训练过程中受限于 Arrow/HDF5 文件的随机 IO 瓶颈，在训练前先执行一键张量缓存：

```bash
# 将原始数据转换为 downsampled 64x128 .pt 缓存张量
python scripts/preprocess_to_tensors.py \
    --data-root data/foil \
    --resolution 64 128 \
    --num-workers 8
```

预处理完成后，从全量数值模拟（CFD）数据中提取并固化前 $K=64$ 阶全场空间正交模态基底：

```bash
# 提取并保存 POD 空间基底 (pod_basis_64x128.pt)
python scripts/compute_sim_pod_basis.py \
    --tensor-dir data/foil/tensor_cache_64x128/numerical \
    --output-file data/foil/pod_basis_64x128.pt \
    --rank 64
```

---

## 四、 模型库与配置文件体系

配置文件统一存放于 `configs/` 目录下：

| 类别 | 配置文件 | 模型架构 | 特点描述 |
| :--- | :--- | :--- | :--- |
| **官方基线** | `configs/baselines/01_masked_unet3d.yaml` | **Masked-UNet3D** | 官方 23.0M 3D-UNet，输入拼接 0/1 二值掩码通道 |
| **官方基线** | `configs/baselines/02_masked_fno3d.yaml` | **Masked-FNO3D** | 3D 谱卷积算子，输入拼接掩码通道 |
| **经典基线** | `configs/baselines/03_classical_gappy_pod.yaml` | **Classical Gappy-POD** | 基于吉洪诺夫正则化的经典纯无监督模态反演 |
| **旗舰 SOTA** | `configs/our_models/01_pod_res_unet3d_sparse.yaml` | **POD-ResUNet3D-Sparse** | **POD 物理升维 + 格拉斯曼旋转 $\mathbf{W}_{\text{align}}$ + 23M UNet3D 时空残差细化** |
| **解耦 SOTA** | `configs/our_models/02_misf_no_latent.yaml` | **MISF-NO** | 双路径变分 Gappy 编码器 + 隐空间 1D-FNO 动力学推进 |

---

## 五、 模型训练与迁移

### 1. 一键批量训练指定目录下的所有模型
```bash
# Linux/服务器：
bash scripts/train_all.sh configs/our_models/ 0

# Windows PowerShell:
.\scripts\train_all.ps1 -ConfigDir configs/our_models -Gpu 0
```

### 2. 单独训练某一个配置
```bash
python -m sparse_pod_sim2real.training.trainer --config configs/our_models/01_pod_res_unet3d_sparse.yaml --gpu 0
```

训练过程分为两阶段自动调度：
* **Stage 1 (Sim Pre-training)**：在全量仿真数据上施加人工稀疏掩码联合预训练，产出 `best_sim.pt`；
* **Stage 2 (Few-shot Real Fine-tuning)**：仅抽取 $n_{\text{real}} \in \{1, 3, 5\}$ 条真实轨迹微调旋转矩阵 $\mathbf{W}_{\text{align}}$ 与适配层，产出 `best.pt`。
* 最佳权重、配置文件与训练日志自动归档于 `best_checkpoints/<run_name>/`。

---

## 六、 批量评估与导出 Excel 汇总表

训练完成后，对 `best_checkpoints/` 下的所有模型执行一键盲测评估：

```bash
# Linux/服务器：
bash scripts/evaluate_all.sh best_checkpoints data/foil 0

# Windows PowerShell:
.\scripts\evaluate_all.ps1 -CkptDir best_checkpoints -DataRoot data/foil -Gpu 0
```

评估完成后自动输出：
1. 终端打印对齐论文格式的基准对比大表；
2. 生成 `evaluation_results_YYYYMMDD_HHMMSS.xlsx`（包含 Overall Rel-L2、RMSE、MAE、R2、Vorticity Rel-L2、KE Error、MVPE 等核心指标）；
3. 生成同名结构化 JSON 文件供论文绘图调用。

---

## 七、 冒烟测试与系统验证

在服务器或本地运行冒烟测试，验证全管线前向后向及评价指标无误：
```bash
python scripts/run_smoke_test.py
```
