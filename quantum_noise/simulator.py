"""密度矩阵小规模模拟器。

- 酉门按层作用，噪声在指定层之后按 NoiseModel 中的声明顺序作用；
- 噪声通过局部 Kraus 算子嵌入全系统，演化后状态保持半正定、迹为 1；
- 测量分两种：精确边际概率（解析）与有限采样（实验统计）。

噪声位置/顺序不同，输入给 Kraus 演化的密度矩阵不同，因此
计算结果会真实改变；屏障层只用于控制噪声插入点。
"""
from __future__ import annotations

import numpy as np

from .channels import kraus_operators
from .circuit import Circuit, NoiseModel, SINGLE_QUBIT_GATES, PARAM_GATES, TWO_QUBIT_GATES


def _unitary(name: str, param: float | None) -> np.ndarray:
    if name == "I":
        return np.eye(2, dtype=complex)
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "Z":
        return np.diag([1.0, -1.0]).astype(complex)
    if name == "H":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if name == "S":
        return np.diag([1.0, 1j]).astype(complex)
    if name == "SDG":
        return np.diag([1.0, -1j]).astype(complex)
    if name == "T":
        return np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(complex)
    if name == "TDG":
        return np.diag([1.0, np.exp(-1j * np.pi / 4)]).astype(complex)
    if name in ("RX", "RY", "RZ"):
        theta = float(param)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        if name == "RX":
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        if name == "RY":
            return np.array([[c, -s], [s, c]], dtype=complex)
        return np.diag(np.exp([-1j * theta / 2, 1j * theta / 2])).astype(complex)
    raise ValueError(f"未知单体门: {name}")


def _two_qubit_unitary(name: str) -> np.ndarray:
    if name == "CX":
        # 位序约定：qubit 0 为高位（|q0 q1>），CX 以 (控制, 目标) 存储
        # 即 00,01,10,11 中翻转目标位：10 <-> 11
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]][::1], dtype=complex
        )
    if name == "CZ":
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    raise ValueError(f"未知双比特门: {name}")


class DensityMatrixSimulator:
    """n 个比特的密度矩阵模拟器（比特 0 为高位序）。"""

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        dim = 2**n_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        self.rho = rho

    # ---- 基本算符嵌入 -------------------------------------------------
    def _embed(self, local: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
        """把局部算符（2^k 维）嵌入到 qubits 指定的全希尔伯特空间。"""
        k = len(qubits)
        if k == 0:
            return np.eye(2**self.n, dtype=complex)
        # local 的轴按传入 qubits 的顺序编号（如 CX 为 控制,目标），
        # 张量轴变换时需要升序排列，先做局部子置换。
        order = sorted(range(k), key=lambda i: qubits[i])
        if order != list(range(k)):
            # local 的基矢按传入顺序编号；重排为编号升序：
            # 新下标 j 的第 i 位等于旧下标的第 order[i] 位。
            basis = 2**k
            row = np.arange(basis)
            new_index = np.zeros(basis, dtype=int)
            for i in range(k):
                new_index |= ((row >> (k - 1 - order[i])) & 1) << (k - 1 - i)
            local = local[np.ix_(new_index, new_index)]
        ordered = tuple(sorted(qubits))
        rest = [q for q in range(self.n) if q not in ordered]
        eye_rest = np.eye(2 ** len(rest), dtype=complex)
        # 局部比特按编号升序排列后与其余比特做克罗内克积，再轴转位。
        # grouped 的张量轴顺序记录在 source_order 中。
        if ordered == tuple(range(k)):
            grouped = np.kron(local, eye_rest)
            source_order = list(ordered) + rest
        else:
            grouped = np.kron(eye_rest, local)
            source_order = rest + list(ordered)
        grouped = grouped.reshape([2] * (2 * self.n))
        perm_out = [source_order.index(q) for q in range(self.n)]
        perm_in = [self.n + source_order.index(q) for q in range(self.n)]
        grouped = np.transpose(grouped, perm_out + perm_in)
        return grouped.reshape(2**self.n, 2**self.n)

    def apply_unitary(self, local_u: np.ndarray, qubits: tuple[int, ...]) -> None:
        u = self._embed(local_u, qubits)
        self.rho = u @ self.rho @ u.conj().T

    def apply_kraus(self, ops: list[np.ndarray], qubits: tuple[int, ...]) -> None:
        embedded = [self._embed(k, qubits) for k in ops]
        new_rho = np.zeros_like(self.rho)
        for k in embedded:
            new_rho += k @ self.rho @ k.conj().T
        self.rho = new_rho

    # ---- 线路执行 ------------------------------------------------------
    def run(self, circuit: Circuit, noise: NoiseModel | None = None) -> np.ndarray:
        if circuit.n_qubits != self.n:
            raise ValueError("线路比特数与模拟器不一致")
        if noise is not None:
            noise.validate(circuit)
        for layer_idx, op in enumerate(circuit.operations):
            if noise is not None:
                for step in noise.at(layer_idx):
                    local_ops = kraus_operators(step.channel, step.qubits, step.params)
                    self.apply_kraus(local_ops, step.qubits)
            if op.name == "BARRIER":
                pass
            elif op.name in SINGLE_QUBIT_GATES or op.name in PARAM_GATES:
                self.apply_unitary(_unitary(op.name, op.param), op.qubits)
            elif op.name in TWO_QUBIT_GATES:
                self.apply_unitary(_two_qubit_unitary(op.name), op.qubits)
            else:
                raise ValueError(f"未知操作: {op.name}")
        if noise is not None:
            for step in noise.at(-1):
                local_ops = kraus_operators(step.channel, step.qubits, step.params)
                self.apply_kraus(local_ops, step.qubits)
        return self.rho

    # ---- 读取 ----------------------------------------------------------
    def probabilities(self) -> np.ndarray:
        """计算基下的精确测量概率（密度矩阵对角元）。"""
        probs = np.real(np.diag(self.rho))
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / probs.sum()
        return probs

    def sample(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """按精确概率抽取 shots 个计算基比特串，返回整数索引数组。"""
        if shots < 0:
            raise ValueError("shots 不能为负")
        return rng.choice(2**self.n, size=shots, p=self.probabilities())


def run_circuit(circuit: Circuit, noise: NoiseModel | None = None) -> np.ndarray:
    """便捷接口：从 |0...0> 执行线路，返回最终密度矩阵。"""
    sim = DensityMatrixSimulator(circuit.n_qubits)
    return sim.run(circuit, noise)
