# 量子噪声研究系统

这是一个面向重复实验的小规模 Python 量子噪声研究工具包。系统使用密度矩阵和 Kraus 通道计算，不包含线路编辑页面；线路由代码中的 `Circuit` 明确定义。

## 能力

- 小规模线路：单量子比特门、`RX/RY/RZ`、`CNOT`、`CZ`、`SWAP`。
- 物理噪声：`bit_flip`、`phase_flip`、`depolarizing`、`amplitude_damping`、通用 Pauli 通道。
 - 顺序有效：`NoiseRule(after_gate=...)` 按线路门索引插入；同一位置的多条规则按列表顺序执行。
- 时间相关噪声：`CorrelatedNoiseGroup` 用两态马尔可夫链耦合多个线路位置的错误事件，相关强度 `correlation` 从 0（独立）到 1（完全相关）；联合噪声轨迹被精确枚举并加权混合，绝不用独立重复抽样代替相关过程。
- 中途测量与条件操作：`Circuit.measure(qubit, cbit)` 在经典寄存器上分裂分支，`Circuit.apply(..., condition=(cbit, value))` 只在匹配分支上作用。
- 随机性可追溯：相关噪声批次中，每个 shot 先按 `noise_seed` 抽噪声轨迹、再按 `measure_seed` 抽测量结果，两种随机性分开追溯、分别可复现；运行结果记录噪声轨迹分布。
- 状态校验：每次模拟检查迹、Hermitian 性、正半定性和测量概率归一。
- 并排检查：同一线路同时输出理想与有噪精确概率、采样计数和差异解释。
 - 批次复现：保存线路（含测量与条件）、噪声设定（含相关组）、时间安排、基础种子、派生采样种子、重复次数与样本计数；归档可重新执行并核对。归档为 schema 3，旧的 schema 1/2 实验仍可读取并重放。
- 持续时间环境扰动：`Schedule` 为每个操作指定持续时间与其后的空闲等待；`CorrelatedNoiseGroup(time_correlation=...)` 让错误事件的 Pearson 相关按经过时间 `time_correlation ** delta_t` 衰减。同一操作序列在不同时间安排（含中途空闲区间）下是不同实验，联合噪声轨迹不同。
- 分支独立的环境延续：中途测量分裂出的每个经典分支各自延续自己的两态环境；两个分支不会因为后续操作数量相同就被认为经历了相同扰动。
- 环境、测量条件与统计分开保存：每个 run 同时归档精确环境路径（标签、所属测量分支、联合概率、条件测量分布）、测量分支计数（`measurement_conditions`）与采样环境路径计数（`noise_paths`）。
- 暂停/恢复重放：`ResumableBatchRun` 按段推进 shot 并把噪声/测量两条 RNG 流的状态写入 `BatchCheckpoint`；在中途测量或空闲区间后暂停，恢复后继续执行，结果与一次性运行逐 shot 一致。交错执行多个批次不会改变各自结果。
- 保守统计：使用精确概率计算期望计数；样本不足或稀有事件标记为 `inconclusive`。若精确概率无差异，采样波动不会被报告成确定噪声效果。

## 直接运行

```bash
python examples/single_qubit_ordering.py
python examples/bell_short_correlation.py
python examples/correlated_strength_sweep.py
python examples/timing_schedules.py
```

命令行示例：

```bash
python -m quantum_noise.cli --preset single --shots 2000 --seed 20260929 --ordering ad-bf
python -m quantum_noise.cli --preset bell --shots 2000 --seed 20260929 --ordering late --output batch.json
python -m quantum_noise.cli --preset corr --noise 0.2 --correlation 0.8 --shots 2000
python -m quantum_noise.cli --preset timing --noise 0.2 --time-correlation 0.5 --shots 2000
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

## 时间安排与暂停恢复示例

```python
from quantum_noise import Circuit, CorrelatedNoiseGroup, NoiseModel, ExperimentBatch, ResumableBatchRun
from quantum_noise.schedule import Schedule

circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
schedule = Schedule((1.0, 1.0), (8.0, 0.0), name="spaced")
group = CorrelatedNoiseGroup(
    channel="bit_flip",
    sites=((0, 0), (1, 0)),
    probability=0.2,
    time_correlation=0.5,
)
batch = ExperimentBatch(
    circuit=circuit,
    noise_model=NoiseModel("spaced", (), (group,)),
    shots=2000,
    base_seed=20261001,
    name="spaced",
    schedule=schedule,
)

runner = ResumableBatchRun(batch)
runner.advance(500)          # 在任意 shot 边界暂停（含中途测量/空闲区间之后）
checkpoint = runner.checkpoint()
# ...之后用 ResumableBatchRun.restore(batch, checkpoint) 恢复并完成剩余 shot
```
