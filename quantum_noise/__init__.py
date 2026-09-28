"""量子噪声研究系统：小规模噪声演化、测量统计与可复现实验批次。

模块划分：
- circuit: 线路与噪声设定（只建模，不做线路编辑器）；
- channels: Kraus 噪声通道；
- simulator: 密度矩阵演化与采样；
- statistics: 边际统计、置信区间与拟合优度检验；
- experiment: 带种子的实验批次、保存与复现。
"""
from .circuit import Circuit, NoiseModel, NoiseStep, Operation
from .channels import (
    bit_flip,
    phase_flip,
    depolarizing,
    amplitude_damping,
    two_qubit_depolarizing,
    kraus_operators,
)
from .simulator import DensityMatrixSimulator, run_circuit
from .statistics import (
    marginal_probs,
    wilson_interval,
    compare_distributions,
    shots_for_halfwidth,
    int_to_bits,
)
from .experiment import Batch, ExperimentResult, scale_noise

__all__ = [
    "Circuit",
    "NoiseModel",
    "NoiseStep",
    "Operation",
    "bit_flip",
    "phase_flip",
    "depolarizing",
    "amplitude_damping",
    "two_qubit_depolarizing",
    "kraus_operators",
    "DensityMatrixSimulator",
    "run_circuit",
    "marginal_probs",
    "wilson_interval",
    "compare_distributions",
    "shots_for_halfwidth",
    "int_to_bits",
    "Batch",
    "ExperimentResult",
    "scale_noise",
]
