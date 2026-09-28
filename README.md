# 量子噪声研究系统

这是一个面向重复实验的小规模 Python 量子噪声研究工具包。系统使用密度矩阵和 Kraus 通道计算，不包含线路编辑页面；线路由代码中的 `Circuit` 明确定义。

## 能力

- 小规模线路：单量子比特门、`RX/RY/RZ`、`CNOT`、`CZ`、`SWAP`。
- 物理噪声：`bit_flip`、`phase_flip`、`depolarizing`、`amplitude_damping`、通用 Pauli 通道。
- 顺序有效：`NoiseRule(after_gate=...)` 按线路门索引插入；同一位置的多条规则按列表顺序执行。
- 状态校验：每次模拟检查迹、Hermitian 性、正半定性和测量概率归一。
- 并排检查：同一线路同时输出理想与有噪精确概率、采样计数和差异解释。
- 批次复现：保存线路、噪声设定、基础种子、派生采样种子、重复次数与样本计数；归档可重新执行并核对。
- 保守统计：使用精确概率计算期望计数；样本不足或稀有事件标记为 `inconclusive`。若精确概率无差异，采样波动不会被报告成确定噪声效果。

## 直接运行

```bash
python examples/single_qubit_ordering.py
python examples/bell_short_correlation.py
```

命令行示例：

```bash
python -m quantum_noise.cli --preset single --shots 2000 --seed 20260929 --ordering ad-bf
python -m quantum_noise.cli --preset bell --shots 2000 --seed 20260929 --ordering late --output batch.json
python -m quantum_noise.cli --verify batch.json
```

## 最小代码示例

```python
from quantum_noise import Circuit, ExperimentBatch, NoiseModel, NoiseRule, run_batch

circuit = Circuit(1).h(0).h(0).h(0)
noise = NoiseModel(
    "damping-then-flip",
    [
        NoiseRule("amplitude_damping", 0.15, (0,), after_gate=0),
        NoiseRule("bit_flip", 0.15, (0,), after_gate=0),
    ],
)
batch = ExperimentBatch(circuit, noise, shots=2000, base_seed=20260929, name="single")
run = run_batch(batch)[0]
print(run.ideal_probabilities, run.noisy_probabilities)
```

两条规则插入在第一个 `H` 与后两个 `H` 之间。交换两条规则的顺序会改变 Kraus 映射组合，并改变最终测量概率。

## 后续误差缓解

当前架构把理想线路、噪声模型、采样结果和统计判断分离。后续可在不改线路描述和批次归档格式核心语义的情况下加入：

- ZNE：同一逻辑线路生成多个噪声强度批次并外推。
- 概率错误抵消：新增校准批次和后处理权重。
- 缓解前后对照：在 `BatchRun` 旁边新增 mitigated counts，而不是覆盖原始测量数据。

## 测试

```bash
python -m unittest discover -s tests -v
```
