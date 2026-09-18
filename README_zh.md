# SparsePOD-Sim2Real: 少样本极端稀疏到全场流体 Sim2Real 迁移学习基准与神经算子

[English](README.md) | [中文说明](README_zh.md)

面向 **ICASSP / ICLR 2027**，构建针对真实工程物理场景的 **少样本极端稀疏到全场流体（Few-shot Extreme-Sparse to Full-field）Sim2Real 预测基准与模型库**。

---

## 一、 项目背景与核心痛点重构

现有前沿偏微分方程神经算子基准（如 RealPDEBench）普遍假定：
1. **全场网格密集观测**：真实世界与数值模拟一样具备全空间 $64 \times 128$ 规则网格；
2. **海量真实轨迹**：拥有数十条长序列真实物理轨迹（上万个训练窗口样本）。

但在真实工业现场（如风洞测试、水下航行体、航空机翼）：
* **极度稀疏（Extreme Spatial Sparsity）**：全场 PIV 光学诊断昂贵且易受遮挡，现场仅能布置 **16 ~ 64 个离散探针（占网格位置的 $0.195\%$ ~ $0.781\%$）**；
* **极度缺乏（Few-shot Scarcity）**：真实物理实验成本极高，通常仅能获取 **1 ~ 3 条有效演化轨迹**；
* **病态稀疏预测**：离散探针使大部分空间自由度不可观测，全场预测需要明确的先验，并通过冻结协议下的实验比较验证。

**本项目研究基于物理模态先验（POD/SVD）与神经算子解耦的解决方案**：
* 借助仿真全场数据离线提取空间正交相干结构；
* 通过**可微 Gappy-POD 投影**与**格拉斯曼流形子空间对齐（$\mathbf{W}_{\text{align}}$）**，瞬时将稀疏点升维为连续物理粗场；
* 接入 3D-UNet 建模非线性高频残差；模型精度与稳定性由冻结协议下的 ID/OOD 实验验证。

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
# 转换完整 99+99 数据、冻结官方数据划分与传感器位置，
# 并分别生成开发阶段 POD 与全 Sim 最终 POD。
bash scripts/prepare_server_data.sh \
    data/foil/hf_dataset/real \
    data/foil/hf_dataset/sim \
    data/foil/tensor_cache_64x128 \
    manifests
```

---

## 四、 模型库与配置文件体系

主模型和基线配置位于 `configs/yaml_main_v1`，受控消融配置位于 `configs/yaml_ablations`。主配置包括 Masked-UNet3D、Masked-FNO3D、Classical Gappy-POD、Gappy-LinearAR、POD-ResUNet3D-Sparse 和 MISF-NO。

---

## 五、 模型训练与迁移

### 1. 一键批量训练指定目录下的所有模型
```bash
# Linux/服务器：
bash scripts/train_all.sh configs/yaml_main_v1 0 data/foil

# Windows PowerShell:
.\scripts\train_all.ps1 -ConfigDir configs/yaml_main_v1 -Gpu 0 -DataRoot data/foil
```

### 2. 按完整协议训练单个配置
```bash
python scripts/run_two_stage.py \
    --config configs/yaml_main_v1/00_pod_res_unet3d_sparse.yaml \
    --dev-basis artifacts/pod_basis_dev_64x128.pt \
    --final-basis artifacts/pod_basis_final_64x128.pt \
    --manifest-dir manifests --data-root data/foil --gpu 0
```

训练过程分为两阶段自动调度：
* **Development Sim**：仅用 Sim train-time block 构建 dev POD，通过 Sim val block 选择最优优化器步数 $S^*$；
* **Final Source**：用覆盖全部 99 条 Sim 完整时间范围的均衡快照构建 final POD；最终源模型使用全部 Sim 帧严格训练 $S^*$ 步，不读取验证集；
* **Few-shot Real**：仅使用 $K\in\{1,3,5\}$ 条真实轨迹的稀疏传感器监督，根据配置保存 `best_dense_val.pt`、`best_sensor_val.pt` 或 `fixed_budget_step_<N>.pt`。
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
2. 生成 `evaluation_results_YYYYMMDD_HHMMSS.xlsx`，分别记录 ID/OOD 的全流体域、传感器、未观测区、分通道、涡量和 RMSE 指标；
3. 生成同名结构化 JSON 文件供论文绘图调用。

---

## 七、 冒烟测试与系统验证

在服务器或本地运行冒烟测试，验证全管线前向后向及评价指标无误：
```bash
python scripts/run_smoke_test.py
```
