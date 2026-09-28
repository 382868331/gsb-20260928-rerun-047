"""噪声通道的 Kraus 表示。

每个通道工厂返回作用在 k 个比特上的 Kraus 算子列表，满足
sum_k K_k^dag K_k = I。模拟器负责把这些局部算子嵌入到全系统，
因此所有演化天然保持迹与半正定性（概率条件）。
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _check_prob(name: str, p: float, allow_one: bool = False) -> None:
    hi = 1.0 if allow_one else 1.0
    if not 0.0 <= p <= hi:
        raise ValueError(f"{name} 的参数必须在 [0, 1]，收到 {p}")


def bit_flip(p: float) -> list[np.ndarray]:
    """比特翻转通道：以概率 p 施加 X。"""
    _check_prob("bit_flip", p)
    return [np.sqrt(1.0 - p) * I2, np.sqrt(p) * X]


def phase_flip(p: float) -> list[np.ndarray]:
    """相位翻转通道：以概率 p 施加 Z。"""
    _check_prob("phase_flip", p)
    return [np.sqrt(1.0 - p) * I2, np.sqrt(p) * Z]


def depolarizing(p: float) -> list[np.ndarray]:
    """单比特去极化：p 概率等权替换为 I/2，Kraus 权重 p/4。"""
    _check_prob("depolarizing", p)
    return [np.sqrt(1.0 - 3.0 * p / 4.0) * I2] + [
        np.sqrt(p / 4.0) * P for P in (X, Y, Z)
    ]


def amplitude_damping(gamma: float) -> list[np.ndarray]:
    """振幅阻尼（向 |0> 弛豫），gamma 为弛豫概率。"""
    _check_prob("amplitude_damping", gamma)
    e0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    e1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return [e0, e1]


def two_qubit_depolarizing(p: float) -> list[np.ndarray]:
    """双比特去极化：p 概率等权替换为 I/4，共 16 个 Pauli 对。"""
    _check_prob("two_qubit_depolarizing", p)
    paulis = [I2, X, Y, Z]
    root = np.sqrt(p / 16.0)
    ops = [np.sqrt(1.0 - 15.0 * p / 16.0) * np.kron(I2, I2)]
    for a in paulis:
        for b in paulis:
            if a is I2 and b is I2:
                continue
            ops.append(root * np.kron(a, b))
    return ops


# 通道名 -> (工厂, 期望参数名, 作用比特数)
REGISTRY: dict[str, tuple] = {
    "bit_flip": (bit_flip, ("p",), 1),
    "phase_flip": (phase_flip, ("p",), 1),
    "depolarizing": (depolarizing, ("p",), 1),
    "amplitude_damping": (amplitude_damping, ("gamma",), 1),
    "two_qubit_depolarizing": (two_qubit_depolarizing, ("p",), 2),
}


def kraus_operators(channel: str, qubits: int, params: dict) -> list[np.ndarray]:
    """按名称构造局部 Kraus 算子，并校验参数与比特数。"""
    if channel not in REGISTRY:
        raise ValueError(f"未知噪声通道: {channel}，可选 {sorted(REGISTRY)}")
    factory, param_names, arity = REGISTRY[channel]
    if len(qubits) != arity:
        raise ValueError(f"通道 {channel} 需要 {arity} 个比特，收到 {len(qubits)}")
    missing = [name for name in param_names if name not in params]
    if missing:
        raise ValueError(f"通道 {channel} 缺少参数: {missing}")
    return factory(*(params[name] for name in param_names))
