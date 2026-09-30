"""Circuit JSON serialization shared by archive versions."""

from __future__ import annotations

from typing import Any, Optional

from .gates import CNOT, CZ, H, S, SWAP, Circuit, ConditionalOperation, Measurement, X, Y, Z, rx, ry, rz


def circuit_to_dict(circuit: Circuit) -> dict[str, Any]:
    operations = []
    for instruction in circuit.operations:
        if isinstance(instruction, Measurement):
            operations.append(
                {"type": "measure", "qubit": instruction.qubit, "bit": instruction.bit}
            )
        elif isinstance(instruction, ConditionalOperation):
            kind, angle = _operation_kind_and_angle(instruction.operation.name)
            operations.append(
                {
                    "type": "conditional",
                    "bit": instruction.bit,
                    "name": kind,
                    "qubit": instruction.operation.qubits[0],
                    "angle": angle,
                }
            )
        else:
            kind, angle = _operation_kind_and_angle(instruction.name)
            operations.append(
                {
                    "type": "gate",
                    "name": kind,
                    "qubits": list(instruction.qubits),
                    "angle": angle,
                }
            )
    return {"num_qubits": circuit.num_qubits, "operations": operations}


def circuit_from_dict(data: dict[str, Any]) -> Circuit:
    circuit = Circuit(int(data["num_qubits"]))
    for item in data["operations"]:
        kind_value = item.get("type", "gate")
        if kind_value == "measure":
            circuit.measure(int(item["qubit"]), int(item["bit"]))
            continue
        name = item["name"]
        angle = item.get("angle")
        matrix = _gate_matrix(name, angle)
        if kind_value == "conditional":
            circuit.conditional(int(item["bit"]), name, matrix, int(item["qubit"]))
            continue
        operation_name = f"{name}({angle:.12g})" if angle is not None else name
        circuit.apply(operation_name, matrix, tuple(item["qubits"]))
    return circuit


def _operation_kind_and_angle(name: str) -> tuple[str, Optional[float]]:
    if name.endswith(")") and "(" in name:
        kind, value = name.split("(", 1)
        return kind, float(value[:-1])
    return name, None


def _gate_matrix(name: str, angle: Optional[float]) -> Any:
    table = {
        "h": H,
        "x": X,
        "y": Y,
        "z": Z,
        "s": S,
        "cnot": CNOT,
        "cz": CZ,
        "swap": SWAP,
    }
    if name in table:
        return table[name]
    if angle is None:
        raise ValueError(f"unknown gate {name!r}")
    if name == "rx":
        return rx(angle)
    if name == "ry":
        return ry(angle)
    if name == "rz":
        return rz(angle)
    raise ValueError(f"unknown gate {name!r}")
