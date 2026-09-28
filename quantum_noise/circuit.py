"""小规模量子线路与噪声设定（仅建模，不含编辑器）。

线路是一个有序的层列表，每层是一个 (门, 作用比特) 元组。
噪声在每个线路层之后按设定逐比特/逐对插入，因此噪声的
位置与顺序会实际参与计算，而不只是记录参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# 支持的单体门：I, X, Y, Z, H, S, SDG, T, TDG 以及 Rx/Ry/Rz(参数)
SINGLE_QUBIT_GATES = {"I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG"}
PARAM_GATES = {"RX", "RY", "RZ"}
TWO_QUBIT_GATES = {"CX", "CZ"}


@dataclass(frozen=True)
class Operation:
    """线路中的一层：门名、作用比特、可选转角参数。"""

    name: str
    qubits: tuple[int, ...]
    param: float | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "qubits": list(self.qubits), "param": self.param}

    @staticmethod
    def from_dict(data: dict) -> "Operation":
        return Operation(data["name"], tuple(data["qubits"]), data.get("param"))

    def __str__(self) -> str:
        q = ",".join(map(str, self.qubits))
        if self.param is not None:
            return f"{self.name.lower()}({self.param:.4g})[{q}]"
        return f"{self.name}[{q}]"


@dataclass
class Circuit:
    """固定比特数的顺序线路；支持追加单体门、受控门和屏障。"""

    n_qubits: int
    operations: list[Operation] = field(default_factory=list)

    def _check(self, name: str, qubits: Iterable[int]) -> tuple[int, ...]:
        qs = tuple(qubits)
        for q in qs:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"比特 {q} 超出范围（线路共 {self.n_qubits} 比特）")
        if len(set(qs)) != len(qs):
            raise ValueError(f"门 {name} 不能在同一比特上重复: {qs}")
        return qs

    def gate(self, name: str, *qubits: int, param: float | None = None) -> "Circuit":
        name = name.upper()
        qs = self._check(name, qubits)
        if name in SINGLE_QUBIT_GATES:
            if param is not None:
                raise ValueError(f"{name} 不接受参数")
            if len(qs) != 1:
                raise ValueError(f"{name} 是单体门")
        elif name in PARAM_GATES:
            if param is None:
                raise ValueError(f"{name} 需要转角参数 param")
            if len(qs) != 1:
                raise ValueError(f"{name} 是单体门")
        elif name in TWO_QUBIT_GATES:
            if param is not None:
                raise ValueError(f"{name} 不接受参数")
            if len(qs) != 2:
                raise ValueError(f"{name} 是双比特门，按 (控制, 目标) 给出")
        else:
            raise ValueError(f"未知门: {name}")
        self.operations.append(Operation(name, qs, param))
        return self

    def barrier(self) -> "Circuit":
        """屏障：强制在该处插入噪声，自身无酉作用。"""
        self.operations.append(Operation("BARRIER", ()))
        return self

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "operations": [op.to_dict() for op in self.operations],
        }

    @staticmethod
    def from_dict(data: dict) -> "Circuit":
        c = Circuit(data["n_qubits"])
        c.operations = [Operation.from_dict(op) for op in data["operations"]]
        return c


@dataclass(frozen=True)
class NoiseStep:
    """一次噪声插入：通道名、作用比特、通道参数。

    噪声作用点用锚点描述：
    - before: 在第 before 个酉层之前（0 基索引）；
    - -1    : 线路末端（所有酉层之后）。
    同一锚点上的多个步骤按列表顺序依次作用。
    """

    channel: str
    qubits: tuple[int, ...]
    params: dict = field(default_factory=dict)
    before: int = -1

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "qubits": list(self.qubits),
            "params": self.params,
            "before": self.before,
        }

    @staticmethod
    def from_dict(data: dict) -> "NoiseStep":
        anchor = data.get("before", data.get("after", -1))
        return NoiseStep(
            data["channel"],
            tuple(data["qubits"]),
            dict(data.get("params", {})),
            anchor,
        )


@dataclass
class NoiseModel:
    """噪声设定：一组带位置锚点的噪声步骤。"""

    steps: list[NoiseStep] = field(default_factory=list)

    def add(self, channel: str, qubits: Iterable[int], before: int = -1, **params) -> "NoiseModel":
        self.steps.append(NoiseStep(channel, tuple(qubits), dict(params), before))
        return self

    def at(self, before: int) -> list[NoiseStep]:
        """按声明顺序返回锚定在第 before 层之前的步骤。"""
        return [s for s in self.steps if s.before == before]

    def validate(self, circuit: Circuit) -> None:
        n_layers = len(circuit.operations)
        for step in self.steps:
            for q in step.qubits:
                if not 0 <= q < circuit.n_qubits:
                    raise ValueError(f"噪声 {step.channel} 的比特 {q} 超出范围")
            if step.before != -1 and not 0 <= step.before < n_layers:
                raise ValueError(f"噪声锚点 before={step.before} 超出层数 {n_layers}")

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps]}

    @staticmethod
    def from_dict(data: dict) -> "NoiseModel":
        return NoiseModel([NoiseStep.from_dict(s) for s in data.get("steps", [])])
