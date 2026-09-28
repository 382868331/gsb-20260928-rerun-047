"""Small-gate library and circuit description."""

from __future__ import annotations

from cmath import exp as cmath_exp
from dataclasses import dataclass, field
from math import cos, sin
from typing import Sequence, Tuple

from .linalg import Matrix

I: Matrix = [[1 + 0j, 0j], [0j, 1 + 0j]]
X: Matrix = [[0j, 1 + 0j], [1 + 0j, 0j]]
Y: Matrix = [[0j, -1j], [1j, 0j]]
Z: Matrix = [[1 + 0j, 0j], [0j, -1 + 0j]]
H: Matrix = [
    [1 / 2**0.5 + 0j, 1 / 2**0.5 + 0j],
    [1 / 2**0.5 + 0j, -1 / 2**0.5 + 0j],
]
S: Matrix = [[1 + 0j, 0j], [0j, 1j]]


def rx(theta: float) -> Matrix:
    return [
        [cos(theta / 2) + 0j, -1j * sin(theta / 2)],
        [-1j * sin(theta / 2), cos(theta / 2) + 0j],
    ]


def ry(theta: float) -> Matrix:
    return [
        [cos(theta / 2) + 0j, -sin(theta / 2) + 0j],
        [sin(theta / 2) + 0j, cos(theta / 2) + 0j],
    ]


def rz(theta: float) -> Matrix:
    return [[cmath_exp(-1j * theta / 2), 0j], [0j, cmath_exp(1j * theta / 2)]]


CNOT: Matrix = [
    [1 + 0j, 0j, 0j, 0j],
    [0j, 1 + 0j, 0j, 0j],
    [0j, 0j, 0j, 1 + 0j],
    [0j, 0j, 1 + 0j, 0j],
]

CZ: Matrix = [
    [1 + 0j, 0j, 0j, 0j],
    [0j, 1 + 0j, 0j, 0j],
    [0j, 0j, 1 + 0j, 0j],
    [0j, 0j, 0j, -1 + 0j],
]
SWAP: Matrix = [
    [1 + 0j, 0j, 0j, 0j],
    [0j, 0j, 1 + 0j, 0j],
    [0j, 1 + 0j, 0j, 0j],
    [0j, 0j, 0j, 1 + 0j],
]


@dataclass(frozen=True)
class Operation:
    name: str
    matrix: Matrix
    qubits: Tuple[int, ...]


@dataclass
class Circuit:
    num_qubits: int
    operations: list[Operation] = field(default_factory=list)

    def apply(self, name: str, matrix: Matrix, qubits: int | Sequence[int]) -> "Circuit":
        if isinstance(qubits, int):
            target = (qubits,)
        else:
            target = tuple(qubits)
        self._validate_qubits(target)
        expected = 2 ** len(target)
        if len(matrix) != expected or any(len(row) != expected for row in matrix):
            raise ValueError(f"matrix for {name!r} must be {expected}x{expected}")
        self.operations.append(Operation(name, matrix, target))
        return self

    def h(self, qubit: int) -> "Circuit":
        return self.apply("h", H, qubit)

    def x(self, qubit: int) -> "Circuit":
        return self.apply("x", X, qubit)

    def y(self, qubit: int) -> "Circuit":
        return self.apply("y", Y, qubit)

    def z(self, qubit: int) -> "Circuit":
        return self.apply("z", Z, qubit)

    def s(self, qubit: int) -> "Circuit":
        return self.apply("s", S, qubit)

    def rx(self, qubit: int, theta: float) -> "Circuit":
        return self.apply(f"rx({theta:.12g})", rx(theta), qubit)

    def ry(self, qubit: int, theta: float) -> "Circuit":
        return self.apply(f"ry({theta:.12g})", ry(theta), qubit)

    def rz(self, qubit: int, theta: float) -> "Circuit":
        return self.apply(f"rz({theta:.12g})", rz(theta), qubit)

    def cnot(self, control: int, target: int) -> "Circuit":
        return self.apply("cnot", CNOT, (control, target))

    def cz(self, left: int, right: int) -> "Circuit":
        return self.apply("cz", CZ, (left, right))

    def swap(self, left: int, right: int) -> "Circuit":
        return self.apply("swap", SWAP, (left, right))

    def _validate_qubits(self, qubits: Tuple[int, ...]) -> None:
        if len(set(qubits)) != len(qubits):
            raise ValueError("repeated qubit in one operation")
        if any(not 0 <= qubit < self.num_qubits for qubit in qubits):
            raise ValueError("qubit index outside circuit")


def bell_circuit() -> Circuit:
    return Circuit(2).h(0).cnot(0, 1)


def ghz_circuit(num_qubits: int) -> Circuit:
    if num_qubits < 2:
        raise ValueError("GHZ circuit needs at least two qubits")
    circuit = Circuit(num_qubits).h(0)
    for qubit in range(1, num_qubits):
        circuit.cnot(0, qubit)
    return circuit
