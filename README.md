# 量子噪声研究系统（Python）

面向**重复实验**的小规模量子噪声研究工具：先建立线路、噪声设定和实验批次，
完成几种明确定义的噪声演化与测量统计，理想情形与有噪情形可并排检查。
系统刻意只做建模与分析，**不含线路编辑页面**；噪声强度缩放接口为后续
误差缓解（如零噪声外推）预留。

## 设计要点

- **噪声位置与顺序真实参与计算**：NoiseModel 的每个步骤锚定在某个酉层
  之前（before=i，0 基）或线路末端（before=-1，默认）；同一锚点上的
  多个通道按声明顺序依次作用。模拟器在相应时刻对当前密度矩阵施加 Kraus
  映射，因此改位置/改顺序会得到不同结果（见
  tests/test_quantum_noise.py 中的两个专用测试）。
- **基本概率条件始终成立**：所有噪声均为 Kraus 通道，演化保持迹 1、
  半正定、厄米；采样概率在数值误差内非负且归一。测试对每条通道和
  完整演化都做校验。
- **种子与批次可复现**：Batch 记录线路、噪声、shots、seed 和噪声
  强度列表，用 numpy.random.default_rng(seed).spawn(...) 派生理想/
  有噪独立随机流。保存为 JSON 后重新加载、重跑，计数逐比特串一致；
  Batch.rerun_verifies() 直接核对。
- **理想 vs 有噪并排**：每个强度同时给出精确解析概率、有限采样计数、
  Wilson 置信区间半宽和卡方拟合优度检验。
- **样本不足不下确定结论**：只有当 Wilson 区间最大半宽小于设定的最小
  效应尺度 min_effect，且卡方检验显著时，才报告“检测到噪声效应”；
  否则明确提示“样本不足，统计波动可能与真实效应同量级”。

## 模块

- quantum_noise/circuit.py：Circuit（I/X/Y/Z/H/S/T/Rx/Ry/Rz/CX/CZ、
  屏障）与 NoiseModel（带锚点的噪声步骤，可序列化）。
- quantum_noise/channels.py：比特翻转、相位翻转、去极化、振幅阻尼、
  双比特去极化的 Kraus 算子与通道注册表。
- quantum_noise/simulator.py：密度矩阵模拟器，局部酉/Kraus 算子嵌入、
  按锚点演化、精确概率与采样。
- quantum_noise/statistics.py：边际概率、Wilson 区间、自实现的卡方
  生存函数（仅依赖标准库，无需 scipy）、分布比较与样本量估算。
- quantum_noise/experiment.py：Batch、噪声强度缩放（scale_noise，
  供后续零噪声外推使用）、JSON 保存/加载/复现校验。
- examples/demo_experiment.py：单体例子（Ry + 振幅阻尼）、Bell 短
  关联线路、样本量警示、批次保存与复现。

## 快速开始

```bash
pip install numpy pytest
python -m pytest tests/ -q
PYTHONPATH=. python examples/demo_experiment.py
```

```python
import numpy as np
from quantum_noise import Circuit, NoiseModel, Batch

# Bell 线路，末端施加双比特去极化
circuit = Circuit(2).gate("H", 0).gate("CX", 0, 1)
noise = NoiseModel().add("two_qubit_depolarizing", (0, 1), p=0.2)

# scale=0 为理想对照；同一批次多种噪声强度并排
batch = Batch(
    "bell", circuit, noise,
    shots=4000, seed=20260929, scales=[0.0, 0.5, 1.0],
    min_effect=0.02,
).run()

for r in batch.results:
    print(r.scale, r.noisy_probs, r.comparison_result().summary())

batch.save("data/bell.json")
assert Batch.load("data/bell.json").rerun_verifies()
```

## 门与位序约定

- 比特编号从 0 开始，比特 0 为计算基高位：|q0 q1>，比特串按
  int(x, 2) 解释。
- 双比特门按 (控制, 目标) 传入，例如 gate("CX", 0, 1)；非相邻或
  反向的比特对同样支持（内部做子空间置换）。
- barrier() 不做酉作用，只用于强制噪声插入位置。

## 统计判定口径

- 卡方拟合优度比较“理想分布 vs 有噪采样计数”，期望频数 <5 的桶合并；
- 显著性要求 p < alpha（默认 0.05）**且** 样本量足够；
- 效应尺度用 L1 距离 sum |p_emp - p_ideal| 报告，但不作为显著性的
  唯一依据，防止小样本把波动当效果；
- shots_for_halfwidth(0.02) 给出达到区间半宽 0.02 所需的保守样本量。

## 后续误差缓解的接入点

- Batch(scales=[0, 0.5, 1, ...]) 已能一次产出多个噪声强度下的同一
  可观测量估计；scale_noise 线性缩放 p/gamma。
- 每个结果同时保留精确概率与计数，后续可在不改动线路层的前提下加入
  零噪声外推、概率错误抵消等模块。

