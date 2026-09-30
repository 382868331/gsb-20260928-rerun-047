# 量子噪声研究系统

这是一个面向重复实验的小规模 Python 量子噪声研究工具包。系统使用密度矩阵和 Kraus 通道计算，不包含线路编辑页面；线路由代码中的 `Circuit` 明确定义。

## 能力

- 小规模线路：单量子比特门、`RX/RY/RZ`、`CNOT`、`CZ`、`SWAP`。
- 中途测量与条件操作：`circuit.measure(qubit, bit)` 与 `circuit.cx_conditional(bit, qubit)`，精确计算枚举所有经典分支。
- 时间相关噪声：`CorrelatedNoiseEvent` 用每量子比特的平稳二元 Markov 链描述跨时刻 Pauli 扰动；`correlation` 为相邻误差指示量的相关系数，`0` 为独立、`1` 为两次扰动完全相同，不允许用独立重复抽样代替。
- 随机流分离：相关噪声轨迹由噪声专用种子驱动，中途测量与最终采样由测量专用种子驱动，逐 shot 追溯 `noise_errors`、`branch_path`、中途测量位与最终结果。
- 相关强度扫描：`sweep_correlation` 固定同一线路，仅改变相关强度，并排比较精确概率、有限采样计数与保守统计结论。
- 物理噪声：`bit_flip`、`phase_flip`、`depolarizing`、`amplitude_damping`、通用 Pauli 通道。
- 顺序有效：`NoiseRule(after_gate=...)` 按线路门索引插入；同一位置的多条规则按列表顺序执行。
- 状态校验：每次模拟检查迹、Hermitian 性、正半定性和测量概率归一。
- 并排检查：同一线路同时输出理想与有噪精确概率、采样计数和差异解释。
- 短线路核对：`exact_probabilities` 对短线路直接枚举噪声路径与测量分支，可直接核对概率数值、密度矩阵合法性和条件分支走向。
- 批次复现：保存线路、噪声设定、基础种子、噪声/测量派生种子、重复次数与样本计数；`replay_archive` 同时支持新的相关噪声归档（schema 2）和旧的独立噪声实验归档（schema 1）。
- 保守统计：使用精确概率计算期望计数；样本不足或稀有事件标记为 `inconclusive`。若精确概率无差异，采样波动不会被报告成确定噪声效果。

## 直接运行

```bash
python examples/single_qubit_ordering.py
python examples/bell_short_correlation.py
python examples/correlated_trajectory.py
```

命令行示例：

```bash
python -m quantum_noise.cli --preset single --shots 2000 --seed 20260929 --ordering ad-bf
python -m quantum_noise.cli --preset bell --shots 2000 --seed 20260929 --ordering late --output batch.json
python -m quantum_noise.cli --verify batch.json
python -m quantum_noise.cli --preset corr --shots 2000 --noise 0.2 --sweep 0,0.5,1
python -m quantum_noise.cli --preset corr --shots 2000 --noise 0.2 --rho 0.8 --output correlated.json
python -m quantum_noise.cli --verify correlated.json
```

相关噪声线路中，门槽从 0 开始计数（条件门也占一个槽位），`after_gate` 指向扰动插入的槽位。两次 X 门之间各插入一次概率 `p=0.2` 的比特翻转时，独立情形 `P(1)=0.32`，完全相关时两次翻转成对抵消 `P(1)=0`。

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
