# SparsePOD-Sim2Real 研究与实现审查

日期：2026-09-17。审查对象：当前本地 SparsePOD-Sim2Real，辅以 POD-Sim2Real 的 20260912 结果归档及用户提供的 RealPDEBench v2 PDF。

审查方式：本地代码检查、CPU 定向复现、既有 checkpoint 小样本推理、针对性原始文献检索。未修改生产代码，未重训，未访问远程服务器。非独立外部审稿；review_independence: local_single_agent；acceptance_status: provisional。不能据此判定完整实验的 SOTA 或会议录用概率。

## 1. 总体判断

Few-shot、真实稀疏观测、Sim 全场预训练、未来全场预测的交集值得研究，也给 POD 提供了比 dense-to-dense 更合理的应用位置。但现状仍是研究原型：核心协议有不一致，若干路径存在确定错误，本地没有完成的 Real 微调比较。README 中的 SOTA、绝对物理稳定、基线崩塌等表述没有当前证据支撑。

推荐核心问题：在只允许少量真实传感器历史进行适配的情况下，如何用仿真先验构造可观测、对域偏移稳健的低维状态，并避免把传感器不可辨识的自由度过拟合成全场变化？

RealPDEBench 本身已把真实数据昂贵、可观测变量有限及 Sim2Real 作为动机；不能把“真实数据少”本身当作新发现。新增价值应明确为严格 sensor-only adaptation、空间采样算子和未来全场预测的统一协议，以及针对可辨识性的具体方法。

## 2. 当前真正实现的任务

- 输入过去 20 帧的固定位置二维速度传感器数据，输出未来 20 帧的二维速度全场。
- 每个测点观测 u、v 两个通道；不是表面压力传感器。
- real 训练损失在 sparse_sensors 模式下确实只使用 y_sensor_values，并加适配正则。
- 实际数据集仍将 x_full 和 y_full 送入所有 batch；这不是自动的数据泄露，但使协议脆弱，且验证阶段确实消费全场。
- few_shot_k 是轨迹数量，不是窗口或帧数。3990 帧轨迹、20+20 窗口、stride=5 会产生 791 个高度相关窗口。论文应报告轨迹数、独立工况数、采集时间、唯一帧数、传感器数量，不能把它称为只有一个训练样本。
- 从真实 PIV 全场抽取测点是合理的受控稀疏观测 benchmark，但不是实际部署物理探针的验证。PIV 本身的测量和后处理误差也应保留说明。
- 在 64×128 网格中，16/32/64 点占 0.195%/0.391%/0.781%，不是 README 中的 <0.1%。按同样 u、v 通道计数，比例不变。

建议主任务写为：Few-trajectory, sensor-only adaptation for sparse-history-to-full-field forecasting under simulation-to-reality shift。实时重构与持续传感器同化可作附加任务，必须与不接受未来观测的开放环预测分开。

## 3. 实现审查：确定问题与边界

### P0：先处理，再进行论文主实验

1. **Real 全场用于选 checkpoint。** training/trainer.py:205-244 收集 y_full 计算指标，:288-305 按该 rel_l2 选择 best。违反严格真实全场不可用的模型选择预算。它不是已经证明的 test leakage，而是使用了额外 dense validation 标签。修复：主协议采用预算内传感器的验证、Sim 验证锁定超参或预先固定适配步数。额外 dense Real validation 只能标为 oracle。K=1 可在同一轨迹做有间隔的时间划分，并避免滑窗跨边界与预处理滤波支持域重叠。

2. **已有索引无匹配时失效开放。** data/dataset.py:99-109：matched 为空仍保留全文件列表；JSON 异常被吞掉；由于 index 文件存在，随机划分也不执行。合成复现中 train/val/test 都包含同一组全部文件。修复为显式错误；规范 .h5/.pt ID；写出匹配数及未匹配 ID；校验不同集合交集与预期数量。当前本地缓存没有这些索引，实际 fallback 划分是三条不同轨迹，因此不能说当前本地训练已经发生 train/test 重叠。

3. **POD 计算与模型训练不是同一个 Sim split。** compute_sim_pod_basis.py:46-48 使用排序前 80%，dataset.py 默认 seed=42 打乱后前 50%。实际本地 Sim train=10000_0.0，val=10000_10.0，test=10000_15.0，而 basis manifest 包含前两个。因此当前 Sim validation 已进入基底构造。原始 RealPDEBench 允许使用所有 Sim 工况，不能一概称 Sim 覆盖 Real test 工况为泄露；应分别规定“允许全 Sim 工况”和“Sim 也严格留出工况”的协议。现有代码声称 Sim train-only 却不一致，必须纠正。

4. **Grassmann QR 改变预训练坐标。** our_model/subspace_warping.py:64-67 使用 q,_=qr(phi+perp@A) 并直接返回 q。实际 basis 在 A=0 时 34/64 列符号翻转，||q-phi||/||phi||=1.4579。虽然张成空间几乎不变，Gappy 编码/解码可以相互抵消符号，但中间已有 modal_prop 没有相应坐标变换。修复：采用正对角 R 的符号规范，或合适的 polar retraction；非零适配也应保持基底坐标连续，并测试 dense/sparse 一致性。不要只测 principal angles：角度为零不能保证坐标一致。

5. **MISF-NO 的批量坐标形状错误。** dataset 返回 [P,2]，DataLoader 叠成 [B,P,2]；PhysicsCrossAttention.forward 在 :42 无条件执行两次 unsqueeze 后 expand 为 [B,T,P,2]，实际是五维扩四维并报错。已用 B=2 复现。smoke test 手造未批量化坐标，掩盖此错误。支持两种形状，且用真正 DataLoader 验证。

6. **rollout 目标时间不推进。** training/evaluate.py:21-49 每一轮都对同一个 curr_batch['y_full'] 打分，只更新输入；第二轮以后评价无效。主 evaluate_checkpoint 也没有调用 rollout，因此当前汇总不支持长程稳定性结论。修复：携带 trajectory_id/start_time/完整未来目标，逐轮推进标签，处理 Tin≠Tout，单独报告每个预测 horizon。

7. **传感器与 split 的 seed 在训练/评估链路不一致。** trainer.py:78 构造模型用的 temp_ds 没传 config.seed；训练 loader 传了；evaluate_checkpoint 又没传 seed。改变 seed 后 random topology 会出现模型 sensor_indices 与输入测点不一致；test split 也可能与 train split 不再隔离。必须保存并复用实际 indices 和 split manifest，不能只依赖重生成。

### P1：会影响可信度、稳定性和可复现性

- **无训练的 classical baseline 无法进入现有汇总。** 其 sim_epochs=real_epochs=0，不保存 best/best_sim；评估却要求至少一个存在并静默跳过。应独立解析评估，或保存 buffer state。
- **随机基底 fallback 不适用于正式实验。** trainer.py:129-134 找不到 POD 时自动随机 QR，会把路径错误伪装成正常研究运行。只在显式 smoke 模式允许。
- **默认搜索与归档可能覆盖实验来源。** 评估优先加载 config/data_root 的 basis，后加载 checkpoint 附带版本；嵌套 best_checkpoints/test_run/model 不被只扫描一级目录的 main 找到。需记录实际 basis 哈希、checkpoint 阶段、seed、数据 ID；没有 Real best 时不能不加说明地用 best_sim 冒充微调结果。
- **AMP 求解尚未保证 FP32。** gappy_solver.py:83-85 先做 matmul/einsum 再 .float()；CUDA autocast 下前面的乘法可能已低精度。建议把整个 Gram/RHS/QR/solve 放入 autocast disabled FP32 区域。此次只有 CPU 验证，未实测 CUDA 错误。
- **MISF 时域 FFT 的 FP16 风险。** misf_no.py:78 在默认 AMP 中对长度 20 做 FFT，没有显式 float；需 CUDA 单测。其 CPU FP32 forward/backward 已通过，不应误报为普遍失效。
- **Q-DEIM 通道排列错误。** sensor_topology.py 中 basis 存储为 [所有u;所有v]，却 reshape(HW,2,K)，把相邻同通道空间位置误当 u/v。修复可先 reshape(2,HW,K) 再转轴，但逐模态取通道范数的 pivot 不等价于对成组 u/v 观测做真正 block-QR，传感器设计仍需明确定义。
- **wall 几何固定。** 写死 NACA0025 的中心、弦长和零攻角坐标，未读取 aoa/几何掩码；与原数据坐标如何配准尚无证据。uniform/random 无流体有效区过滤；uniform 某些 P 值生成格点不足时直接随机补替整个布局，名称与实现可能不一致。需依据元数据检查各攻角、遮挡和 solid cells。
- **指标不能统称完整官方指标。** 当前有基础误差与自定义 vorticity，缺 FE/fRMSE/Update Ratio。vorticity 用 roll 跨非周期边界且没有 dx/dy；MVPE 先跨样本平均再取绝对误差，会抵消不同工况偏差。应按轨迹分别计算再汇总，输出 fluid-only、未观测区域、尾流、u/v、波动场以及按 horizon 的指标。KE 的时间去均值形式与论文相近，不应把整个指标实现都判错。
- **完整评估把所有 pred/target 放在 CPU 后 cat。** 真实全量重叠窗口可能占数十 GB，应在线累积逐样本/逐轨迹指标。
- **数据源目录被 gitignore 的 data/ 广泛忽略。** 本地 src/.../data/*.py 存在，但 rg --files 默认不列出。需要实际 git 跟踪检查再判断是否已入库；建议显式保留源码路径，避免分发缺少 Dataset。
- **全局随机性未统一固定。** Dataset seed 只控制局部采样，trainer 没有统一设置 torch/numpy/python seed；多 seed 实验要区分初始化 seed、support selection seed、sensor seed。

## 4. 模型设计的有效点与不足

### POD-ResUNet

有效点：正交 Sim 模态提供空间关联的先验；Real 只调 warping 和 modal_prop，避免完全开放 23M residual backbone；真实损失没有调用全场。这些适合低观测预算。

不足：

- Sim 训练走 dense projection，Real 走病态 Gappy inversion。这是可避免的输入分布偏移。Sim 阶段同样应模拟稀疏和噪声输入，同时仍用全场 Sim 标签训练，所有基线同等处理。可保留 dense teacher，并蒸馏到 sparse student。
- UNet residual 没有被限制在 POD 正交补，联合训练时可吞掉低模态预测任务；名称中的“高频”和“正交解耦”没有实现保证。POD 正交补也不等价于空间高频。需要组件能量/误差分解，或尝试 (I-Phi Phi^T) residual 并承认它可能损失性能。
- mean_flow 永久固定在 Sim 均值；warp 只能改变波动基底，而真实域均值、增益、坐标配准都可能变化。优先试少参数的均值修正/输入偏置和校准，不要把所有误差都归因于动力学。
- Phi_perp 只有后 16 个 Sim 模态，适配始终在前 80 个模态的联合空间内，无法覆盖任意真实新结构。缺 perp 时 Cayley 仅改变同一子空间内部坐标，principal angles 理论上不发生实质变化。
- 实测 23,680,530 参数，其中 modal_prop 722,688，Grassmann A 1,024。Real adaptation 并非“仅千级参数”。ModalLinearPropagator 实际是 GELU MLP，不是线性动力学。
- 正交基底限制表示几何，不保证预测动力学的能量有界或 rollout 稳定。不能由 POD/Grassmann 直接推出绝对物理稳定。

### MISF-NO

实测总参数 175,461，值得作为紧凑路线继续研究；但当前 cross-attention 在 dense pretrain 完全跳过，只有少量 Real 数据训练它，浪费了 Sim 全场监督。LayerNorm(delta_a) 会削弱补偿量幅度的可解释性。所谓 LatentDynamicsFNO1D 是 Conv1D 加固定频谱截断，没有标准 FNO 的可学习频域权重；Bayesian MAP 可解释为各向同性 Gaussian ridge prior，但当前未接 singular_values，没有后验方差、变分分布或 KL，不能称为变分不确定性模型。

## 5. 实际证据

复现脚本和原始结果：review_diagnostics_20260917.py/json、checkpoint_probe_20260917.py/json。CPU PyTorch 2.14.0，无 CUDA。

### 既有记录

本地新项目只发现 best_sim，日志记录 Sim 1 epoch：Val Rel-L2=7.93%、vorticity=43.03%；Real 开始后没有完成指标，没有 best.pt。不能将这些 Sim 指标与原论文 Real 测试或旧项目分数直接比较。

基底由本地 3 条 Sim 中前 2 条生成，64 阶捕获 88.286% 波动能量，不能假设覆盖几乎全部结构。sqrt(1-energy)=34.23% 是这组构建快照的波动范数残差比例，不是含均值全场或 Real 的 Rel-L2。

旧 20260912 归档中，POD-AFNO 总 Rel-L2 6.151%，v 通道 54.860%，涡量 79.361%；POD-UNet 对应 7.578%、63.103%、84.367%。这提示总误差可能掩盖横向速度和涡结构误差，但此次未统一旧 baseline 的协议和指标，不能给跨目录的正式排名。

### 原项目演示窗口的复现

固定 Real 10000_15.0.h5.pt 第 50–69 帧，现有 basis，64 测点，主配置 lambda=1e-4；每项均为同帧重构而非预测。

|方法|全场 Rel-L2|采样矩阵条件数|
|---|---:|---:|
|Sim 均值场|25.79%|不适用|
|全场可见的固定基底投影 oracle|23.40%|不适用|
|wall Gappy|66.92%|2253.09|
|uniform Gappy|69.07%|507.17|
|wake_rake Gappy|47.69%|89.62|
|random Gappy|55.66%|345.02|

条件数并不单独决定真实误差；模型失配、正则和观测噪声同时起作用。该 oracle 只约束固定均值+固定基底的重构，不是含自由 UNet 残差模型的误差下限。现有 demo 的 lambda=1e-3，与训练主配置也不同。

进一步使用每条 Real 轨迹固定的 50/1000/2000/3000 起点，各 20 帧，共 80 帧：0°/10°/15° 的固定基底 oracle Rel-L2 分别为 48.40%/23.60%/23.47%，wall Gappy 为 105.42%/66.98%/67.56%。仍是探索性诊断，不能替代独立测试或被继续拿来调参。

### 已有 checkpoint 的小样本预测

同一轨迹过去 50–69 帧预测 70–89 帧，已有 1 epoch Sim checkpoint，无 Real 适配：

|推理路径|全场 Rel-L2|
|---|---:|
|全场历史 oracle 输入|26.87%|
|当前 sparse 路径|36.53%|
|仅诊断中统一 QR 符号|33.86%|

符号修正有影响，但绝非解决所有问题。该样本全场历史 persistence 为 2.35%，属于拥有额外 dense 输入的 privileged reference，不能当成稀疏公平基线来宣布击败模型；它提示这个短时 horizon 可能平滑，必须加合法的 Gappy/传感器 persistence 对照及更长 horizon。

## 6. 新意与基线公平性

已有工作足以否定“POD+少量测点+重构本身就是新范式”的强说法，但本次是针对性检索，不是穷尽查新：

- SHRED：利用传感器时间历史重构并预测全场。[论文](https://arxiv.org/abs/2301.12011)
- SHRED-ROM：POD 压缩、稀疏传感器、跨参数场景，与你们设想非常接近。[Nature Communications 2025](https://www.nature.com/articles/s41467-025-65126-y)
- Voronoi-CNN：把非规则测点映射到网格后用 CNN 做全场重构。[Nature Machine Intelligence](https://www.nature.com/articles/s42256-021-00402-2)
- Gappy AE：非线性低维表示下的稀疏反演。[论文](https://arxiv.org/abs/2312.07902)
- Transfer Learning for Reduced-Order Modeling of Transonic Flows Using Multifidelity Data：多保真迁移与部分表面数据/POD 反演相关，需要逐项比较数据权限和任务。[ICAS 2024 原文](https://www.icas.org/icas_archive/icas2024/data/papers/icas2024_1110_paper.pdf)

当前经典 baseline 实际是 Gappy 重构最后一帧然后 persistence，不是具有预测能力的全部 classical ROM。应至少加入：Gappy+线性 AR/DMD、sensor-history MLP/GRU+POD decoder、SHRED 类模型；在可比训练预算下加入 Gappy+同款 UNet 无适配版，识别提升来自 lifting 还是新模块。

Masked UNet/FNO 主配置已启用 Voronoi，不应被描述成只用零填充的弱基线。但它们同样需要在 Sim 阶段接触稀疏输入。比较 full fine-tune 与预算匹配 adapter/head-only；同样的 sensor masks、support 轨迹、预训练数据、噪声、早停规则和调参次数。

实测参数数目（实标量）：POD-ResUNet 23.68M、Masked UNet 22.98M、Masked FNO 16.79M、MISF 0.175M。当前 FNO 不是原论文 Table 1 Foil 的 50.4M 配置；复数参数统计约定也须统一。称“基于官方实现的稀疏适配基线”比“完全官方配置”准确。

## 7. 最值得研究的方案：可观测性约束的模态适配

建议保持一个主方法，先建立强的小模型，不同时叠加变分、FNO、Grassmann、大 UNet 等多个尚无证据的故事。

定义 y_t=H u_t+epsilon_t，u_t=mu_eta+Phi_eta a_t+r_t。H 必须说明测量的物理量、坐标与可见性；压力不能直接等同速度场采样。采用下面的时序先验作为起点：

    a_t^- = f_theta(a_{t-1}, context)
    a_t^+ = argmin_a ||y_t-H(mu_eta+Phi_eta a)||^2_{R^-1}
                       + ||a-a_t^-||^2_{P_t^-1}

这本身是经典 MAP/滤波形式，不是新贡献。可研究的新增机制：在传感器可辨识的方向上更新低维域偏移参数，未观测方向保留仿真先验并输出不确定性；使用一段传感器历史提升可辨识性，而不是逐帧无先验反演后再接 MLP。

静态可观测性从 G=H Phi 的奇异值判断。16 个双通道点最多 32 个标量方程，K=64 必然秩不足；ridge 使求解唯一不意味着真实状态可辨识。32 点即使方程数等于 K 也可能病态。

对局部线性动力学 a_{t+1}=F a_t，使用观测矩阵 O_L=[G;GF;...;GF^(L-1)] 或有限时域 observability Gramian；对非线性网络用沿轨迹 Jacobian 局部近似，并限定解释范围。评估时间窗口 L、有效秩、sigma_min 与预测误差/噪声放大的关系。

建议先试三个可独立证伪的机制：

1. **时间先验的 Gappy/MAP 推断。** 对比 framewise ridge 与历史窗口推断；P、R 从 Sim train 或预算内传感器估计，不能用真实 dense 测试统计。
2. **受限域适配。** 先校准 mean/gain，再比较 dynamics-only、basis-only、joint；basis 更新受先验、步长和可观测性约束。若完全不可见的方向也被自由更新，稀疏训练误差不保证全场误差。
3. **受约束残差。** 首先无残差；确有 projection bottleneck 时再试小 decoder、正交补残差、Sim replay/teacher consistency。不能宣称 nullspace residual 从稀疏标签可被直接监督到。

开放环预测只在历史观测阶段执行同化，未来完全 rollout。持续同化作为不同任务使用新测量，并明确测量预算。稳定性理论需给出 contraction/bounded Jacobian、噪声界和模型偏差等条件；不能由正交性推出。

## 8. 最小实验计划与停止规则

算力环境尚不明确，以下给出相对成本；不能把本地一次 627.8s 的 Sim epoch 外推为所有实验 GPU 时间。

|优先级|实验|成本|决定什么|
|---|---|---|---|
|P0|修复 split/seed/QR/coords/rollout，真 DataLoader smoke|CPU 到少量 GPU 分钟|流程能否可信运行|
|P0|锁定协议、实际 support 与 sensor manifest、model selection 权限|CPU|是否真的 sensor-only few-shot|
|P1|Sim train 与诊断集的均值、配准、oracle POD rank 扫描、H Phi 频谱|CPU；无需训练|问题是表示还是观测|
|P1|matched sparse Sim pretraining，对所有方法使用相同 masks/noise|每架构一次预训练|去掉人为输入域偏移|
|P1|Gappy+AR/DMD、GRU-POD、SHRED、Voronoi-UNet、当前主模型|先 3 seeds 小矩阵|复杂结构是否有必要|
|P2|K_real={1,3,5}, P={16,32,64}；锁定 test|多次廉价适配|预算优势与置信区间|
|P2|关 warp、dynamics-only、mean-only、小 residual、Gappy+同 UNet|复用预训练|定位真正贡献|
|P2|噪声、dropout、未见布局、参数外推、长 horizon|主要是推理|鲁棒性及域泛化|
|P3|至少第二物理场景，随后第三个|新预训练|泛化论据，尤其 ICLR|

支持集用同一随机排列的前 K 条以便嵌套比较；按轨迹/独立工况重采样求置信区间，不把滑窗当独立样本。报告 Real 适配参数量、训练与推理时间、显存。对 Foil 三维切片不应硬套严格二维不可压 Navier–Stokes 约束：未测量的跨平面导数可能重要。

停止/转向规则：如果 GRU-POD 或 SHRED 达到同等精度，删除无收益的大 residual；如果 oracle 投影很差，先处理均值/几何/字典；如果仅 dense Real validation 才能选出好模型，主设定尚未解决；若仅墙面特定布局获益，限定结论并解释观测机制。

## 9. Claim 矩阵

|证据结果|允许主张|不允许主张|
|---|---|---|
|只跑通前后向/一轮 Sim|可运行原型|SOTA、真实泛化|
|严格 sensor-only，多 seeds，同预算优于强基线|该预算与数据集上的有效提升|所有 PDE 或工业部署通用|
|关适配后退化且输入/参数公平|该适配模块有贡献|单凭主表称 Grassmann 是原因|
|sigma/历史窗口变化预测误差与理论一致|有条件的可观测性机制解释|无条件可恢复任意全场|
|多场景、噪声、布局与参数外推稳定获益|较强跨场景鲁棒性证据|绝对物理稳定|
|oracle dense validation 优于 sensor selection|oracle 模型选择上界|严格低预算部署结果|

建议论文结构：严格任务及数据权限 → 观测病态和域偏移的分解 → 一个可观测性约束方法 → 公平主表 → 机制与失效区间 → 限制。主图优先展示实际预算、sensor geometry、sigma/误差趋势和长程预测，而不是堆叠模块图。

## 10. 投稿定位与日期

ICASSP 的技术重心可放在稀疏逆问题、时间先验和可观测子空间适配；ICLR 则更需要一般学习原则、多场景与强机制证据。这是本次审查判断，不是录用预测。当前版本两者都不应以 SOTA 完成稿定位。

2026-09-17 查询官方信息：ICLR 2027 摘要截止 2026-09-18 23:59 AoE，全文 2026-09-25 23:59 AoE，对应北京时间 9 月 19 日/26 日 19:59。ICASSP 2027 官方 Paper Kit 公布常规/Special Session/OJSP 截止 2026-09-16；具体当前可提交状态及时区以投稿系统为准。不能因为会议年份为 2027 就假定还有数月准备。

[ICLR 2027 CFP](https://www.iclr.cc/Conferences/2027/CallForPapers)
[ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
[RealPDEBench 原文](https://arxiv.org/abs/2601.01829)

## 11. 审查轮次与结论边界

第一轮：逐项核对问题定义、代码、配置和产物，提出协议/基底/观测/模型选择疑点。
第二轮：合成复现 split 与 batched coords；使用真实基底验证 QR；按固定演示窗口测量条件数和投影误差。
第三轮：已有 checkpoint 验证符号变换影响，扩展固定窗口诊断，并用原始文献约束新意与下一步方案。

仍未解决：完整训练后的表现、全部官方索引格式/远程数据划分、不同几何的真实坐标映射、CUDA AMP 行为、完整基线排名与全面查新。上述建议为可验证路线，不能保证 SOTA。新增文件仅为独立审查报告和诊断脚本/JSON；生产代码修复和正式实验尚未执行。
