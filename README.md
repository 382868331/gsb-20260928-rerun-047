# 量子噪声研究系统

这是一个面向重复实验的小规模 Python 量子噪声研究工具包。系统使用密度矩阵和 Kraus 通道计算，不包含线路编辑页面；线路由代码中的 `Circuit` 明确定义。

## 能力

- 小规模线路：单量子比特门、`RX/RY/RZ`、`CNOT`、`CZ`、`SWAP`。
- 物理噪声：`bit_flip`、`phase_flip`、`depolarizing`、`amplitude_damping`、通用 Pauli 通道。
 - 顺序有效：`NoiseRule(after_gate=...)` 按线路门索引插入；同一位置的多条规则按列表顺序执行。
- 时间相关噪声：`CorrelatedNoiseGroup` 用两态马尔可夫链耦合多个线路位置的错误事件，相关强度 `correlation` 从 0（独立）到 1（完全相关）；联合噪声轨迹被精确枚举并加权混合，绝不用独立重复抽样代替相关过程。
- 共同环境事件：`JointEnvironmentEvent` 在半开持续时间窗 `[start, end)` 内一次性抽取共同 Bernoulli 条件，并把 `bit_flip`/`phase_flip` 同时作用于指定量子位；另可叠加每量子位独立条件。共同项、独立项和混合项分别保留为完整联合测量分布。
- 中途测量与条件操作：`Circuit.measure(qubit, cbit)` 在经典寄存器上分裂分支，`Circuit.apply(..., condition=(cbit, value))` 只在匹配分支上作用。
- 中途测量续接：`save_mid_measurement_continuation` 保存已观察经典分支、锚点时间和已发生联合环境条件；`MidMeasurementContinuation.continue_batch("observed")` 固定这些条件，`continue_batch("resample")` 只让后续事件重新抽样，新增事件不得早于锚点。
- 随机性可追溯：相关噪声批次中，每个 shot 先按 `noise_seed` 抽噪声轨迹、再按 `measure_seed` 抽测量结果，两种随机性分开追溯、分别可复现；运行结果记录噪声轨迹分布。
- 时间安排与持续扰动：`TimeSchedule` 为每个操作和随后的空闲区间指定时间；相关组使用有限 `correlation_time`，相关随两次环境事件的实际间隔指数衰减。
- 分支独立环境：中途测量后的每个经典分支单独延续环境状态；条件门被跳过时，不会只因为分支操作数量相同就共享扰动历史。
- 可恢复批次：时间表对照分别保存环境变化、测量条件和统计结果；单批次与多批次检查点可在中途测量或空闲暂停后恢复，改变套件执行顺序不改变各批次结果。
- 状态校验：每次模拟检查迹、Hermitian 性、正半定性和测量概率归一。
- 并排检查：同一线路同时输出理想与有噪精确概率、采样计数和差异解释。
 - 批次复现：保存线路（含测量与条件）、噪声设定（含相关组）、基础种子、派生采样种子、重复次数与样本计数；归档可重新执行并核对。归档为 schema 2，旧的 schema 1 独立噪声实验仍可读取并重放。
- 保守统计：使用精确概率计算期望计数；样本不足或稀有事件标记为 `inconclusive`。若精确概率无差异，采样波动不会被报告成确定噪声效果。

## 直接运行

```bash
python examples/single_qubit_ordering.py
python examples/bell_short_correlation.py
python examples/correlated_strength_sweep.py
python examples/time_schedule_comparison.py
python examples/perturbation_timing_comparison.py
python examples/joint_environment_comparison.py
python examples/perturbation_identification.py
```

命令行示例：

```bash
python -m quantum_noise.cli --preset single --shots 2000 --seed 20260929 --ordering ad-bf
python -m quantum_noise.cli --preset bell --shots 2000 --seed 20260929 --ordering late --output batch.json
python -m quantum_noise.cli --preset corr --noise 0.2 --correlation 0.8 --shots 2000
python -m quantum_noise.cli --verify batch.json
```

### 时间安排对照

```python
schedule = TimeSchedule(
    operation_durations=(0.0, 0.0),
    idle_after=(0.0, 20.0),
    name="long-idle",
)
group = CorrelatedNoiseGroup(
    channel="bit_flip",
    sites=((0, 0), (1, 0)),
    probability=0.2,
    correlation=1.0,
    correlation_time=1.0,
)
batch = ExperimentBatch(circuit, NoiseModel("time", (), (group,)), 2000, 20261001,
                        name="long-idle", schedule=schedule)
```

`compare_schedules` 用同一操作序列生成多个独立命名批次；`save_schedule_comparison` 为每个时间安排分别写入环境变化、测量条件与统计结果。`start_resumable_batch`/`resume_batch` 和 `BatchCheckpoint.resume(..., order=[...])` 用于暂停恢复和检查执行顺序无关性。

### 外部扰动时间窗

`Perturbation(start, end, probability=...)` 表示一段 `[start, end)` 的外部扰动；事件按 `TimeSchedule` 的实际时间取环境状态，落在空闲区间或操作中途都不会被推迟到下一次操作。`end` 边界恢复为基础环境，因此 `start == end` 的瞬时扰动是无效果的边界对照。多个扰动可重叠：高 `priority` 决定当前环境，同优先级以后加入者为准；扰动结束后自动恢复到基础条件或仍在生效的其他扰动。

`ExperimentPrefix.fork(name, perturbations)` 从同一前缀派生不同扰动时刻的独立批次，并保留锚点之前已发生的扰动；尝试把新扰动写到锚点之前会报错。`ExperimentBatch.with_perturbations(...)` 用于没有共同历史约束的直接派生。暂停恢复可使用 `start_resumable_batch(..., pause_at_perturbation=(name, phase))`，阶段为 `start`、`overlap` 或 `end`；恢复时重放原批次，采样结果与直接执行及改变检查点执行顺序一致。`direct_perturbed_probabilities` 提供无扰动、持续扰动和半开区间瞬时边界的短线路直接对照。

### 共同环境与独立环境对照

```python
event = JointEnvironmentEvent(
    name="pulse",
    start=1.0,
    end=3.0,
    after_gate=1,
    qubits=(0, 1),
    common_probability=0.2,
    independent_probability=0.1,
)
model = NoiseModel("mixed-environment", (), (), (event,))
```

二量子位 `|00>` 在事件点后的精确分布可手算：

- 完全共同翻转 `c=0.2, q=0`：`P(00)=0.8`，`P(11)=0.2`。
- 完全独立翻转 `c=0, q=0.1`：`P(00)=0.81`，`P(01)=P(10)=0.09`，`P(11)=0.01`。
- 混合 `c=0.2, q=0.1`：`P(00)=0.648`，`P(01)=P(10)=0.072`，`P(11)=0.208`。

模拟器枚举的是联合模式 `00/01/10/11`；`11` 的共同分量不会拆成两个量子位分别抽样。`start == end` 或 `event.cancel()` 后事件无效果；共同/独立概率必须位于 `[0, 1]`。

### 共同扰动与独立扰动辨识

`quantum_noise.identification` 根据联合测量计数区分共同扰动与独立扰动。研究者提供二至三量子位线路、`TimeSchedule`、计数数据和若干 `CandidateWindow` 候选事件时窗；`fit_event_models` 在合法概率范围 `[0, 1]` 内网格扫描每个时窗的 `(common_probability, independent_probability)`，用精确多项对数似然评分完全共同、完全独立与混合三类模型：

- 混合类嵌套另外两类，因此类间比较使用 AIC（与最优类差距小于 2 的类保留）；类内参数组合按似然比阈值保留为可选解释。
- 样本不足（期望计数过低）、多个时窗同样拟合（参数不可辨识）或多类模型同时保留时，结果为 `inconclusive`，所有存留解释及其依据都记录在 `reasons` 中。
- 各比特边际相同不能宣称模型等价：当最优共同拟合与最优独立拟合的边际在采样误差内一致、但联合分布的总变差距离超出采样误差时，结果附带 `marginals_warning`。

`predict_joint_distribution` 把选中的候选组合放到新线路上预测联合分布，`test_prediction` 用追加的独立样本检验该预测，样本不足时报告 `inconclusive`。`PerturbationStudy` 把研究绑定到原始线路、时窗和计数；后续证据必须声明 `"observed"`（沿已观察结果继续）或 `"resample"`（重新抽样）模式，`pooled_counts` 只合并同一模式，两种模式的证据不能混用，`identify()` 始终只使用原始计数。

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
