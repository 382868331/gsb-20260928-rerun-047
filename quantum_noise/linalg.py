"""Small complex-matrix helpers used by the density-matrix simulator.

Matrices are nested Python lists containing ``complex`` numbers.  The module is
deliberately limited to the operations needed for small research circuits.
"""

from __future__ import annotations

from math import sqrt
from typing import List, Sequence

Matrix = List[List[complex]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0j for _ in range(cols)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    result = zeros(size, size)
    for index in range(size):
        result[index][index] = 1.0 + 0.0j
    return result


def dagger(matrix: Matrix) -> Matrix:
    rows = len(matrix)
    cols = len(matrix[0])
    result = zeros(cols, rows)
    for row in range(rows):
        for col in range(cols):
            result[col][row] = matrix[row][col].conjugate()
    return result


def matmul(left: Matrix, right: Matrix) -> Matrix:
    if len(left[0]) != len(right):
        raise ValueError("matrix dimensions do not match")
    shared = len(right)
    result = zeros(len(left), len(right[0]))
    for row in range(len(left)):
        for middle in range(shared):
            scalar = left[row][middle]
            if scalar == 0:
                continue
            for col in range(len(right[0])):
                result[row][col] += scalar * right[middle][col]
    return result


def kron(left: Matrix, right: Matrix) -> Matrix:
    left_rows = len(left)
    left_cols = len(left[0])
    right_rows = len(right)
    right_cols = len(right[0])
    result = zeros(left_rows * right_rows, left_cols * right_cols)
    for lr in range(left_rows):
        for lc in range(left_cols):
            value = left[lr][lc]
            base_row = lr * right_rows
            base_col = lc * right_cols
            for rr in range(right_rows):
                for rc in range(right_cols):
                    result[base_row + rr][base_col + rc] = value * right[rr][rc]
    return result


def kron_all(matrices: Sequence[Matrix]) -> Matrix:
    if not matrices:
        return identity(1)
    result = matrices[0]
    for matrix in matrices[1:]:
        result = kron(result, matrix)
    return result


def trace(matrix: Matrix) -> complex:
    return sum(matrix[index][index] for index in range(min(len(matrix), len(matrix[0]))))


def embedding(operator: Matrix, qubits: Sequence[int], num_qubits: int) -> Matrix:
    """Embed an operator on selected qubits into the full computational space.

    Tensor-factor ordering follows bit labels: qubit 0 is the most significant
    bit in a basis string such as ``"010"``.
    """

    selected = tuple(qubits)
    if len(set(selected)) != len(selected):
        raise ValueError("a gate cannot act repeatedly on one qubit")
    if any(not 0 <= qubit < num_qubits for qubit in selected):
        raise ValueError("qubit index outside circuit")
    expected_size = 2 ** len(selected)
    if len(operator) != expected_size or len(operator[0]) != expected_size:
        raise ValueError("operator dimension does not match target qubits")

    return multi_qubit_embedding(operator, selected, num_qubits)


def multi_qubit_embedding(operator: Matrix, qubits: Sequence[int], num_qubits: int) -> Matrix:
    """Embed an arbitrary operator using explicit computational-basis permutation."""

    qubits = tuple(qubits)
    size = 2**num_qubits
    selected_size = 2 ** len(qubits)
    if len(operator) != selected_size:
        raise ValueError("operator dimension does not match target qubits")
    result = zeros(size, size)
    for input_index in range(size):
        input_bits = _bits(input_index, num_qubits)
        selected_input = _collect_bits(input_bits, qubits)
        for output_index in range(size):
            output_bits = _bits(output_index, num_qubits)
            if [output_bits[q] for q in range(num_qubits) if q not in qubits] != [
                input_bits[q] for q in range(num_qubits) if q not in qubits
            ]:
                continue
            selected_output = _collect_bits(output_bits, qubits)
            result[output_index][input_index] = operator[selected_output][selected_input]
    return result


def _bits(index: int, width: int) -> list[int]:
    return [(index >> (width - 1 - position)) & 1 for position in range(width)]


def _collect_bits(bits: Sequence[int], qubits: Sequence[int]) -> int:
    value = 0
    for bit in (bits[qubit] for qubit in qubits):
        value = (value << 1) | bit
    return value



def normalized(vector: Sequence[complex]) -> List[complex]:
    norm = sqrt(sum(abs(value) ** 2 for value in vector))
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return [complex(value) / norm for value in vector]
