"""端到端演示：单体例子与短关联线路上的噪声演化、测量统计与复现。

运行：
    python examples/demo_experiment.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from quantum_noise import (
    Batch,
    Circuit,
    NoiseModel,
    shots_for_halfwidth,
)


LABELS_1Q = ["|0>", "|1>"]
LABELS_2Q = ["|00>", "|01>", "|10>", "|11>"]


def show(title: str, labels: list[str], batch: Batch) -> None:
    print("=" * 68)
    print(title)
    print(f"种子={batch.seed}  shots={batch.shots}")
    for result in batch.results:
        print(f"  噪声强度 scale={result.scale}")
        ideal = np.array(result.ideal_probs)
        noisy = np.array(result.noisy_probs)
        print("    理想精确概率 :", " ".join(f"{p:.3f}" for p in ideal))
        print("    有噪精确概率 :", " ".join(f"{p:.3f}" for p in noisy))
        print("    理想采样计数 :", result.ideal_counts)
        print("    有噪采样计数 :", result.noisy_counts)
        print("    统计判定     :", result.comparison_result().summary())


def demo_single_qubit() -> None:
    # Ry(pi/2)|0> = |+y>，末端振幅阻尼把 |1> 布居向 |0> 弛豫
    circuit = Circuit(1).gate("RY", 0, param=np.pi / 2)
    noise = NoiseModel().add("amplitude_damping", (0,), gamma=0.3)
    batch = Batch(
        "single-qubit-ad",
        circuit,
        noise,
        shots=2000,
        seed=20260929,
        scales=[0.0, 0.5, 1.0],
        min_effect=0.03,
    ).run()
    show("单体例子：Ry(pi/2) + 振幅阻尼", LABELS_1Q, batch)


def demo_bell() -> None:
    # Bell 短关联线路 + 双比特去极化
    circuit = Circuit(2).gate("H", 0).gate("CX", 0, 1)
    noise = NoiseModel().add("two_qubit_depolarizing", (0, 1), p=0.2)
    batch = Batch(
        "bell-2q-depol",
        circuit,
        noise,
        shots=4000,
        seed=4242,
        scales=[0.0, 1.0],
        min_effect=0.02,
    ).run()
    show("短关联线路：Bell + 双比特去极化", LABELS_2Q, batch)


def demo_sample_size_warning() -> None:
    print("=" * 68)
    print("样本不足警示：同一理想线路，10 个样本 vs 4000 个样本")
    circuit = Circuit(1).gate("H", 0)
    noise = NoiseModel().add("amplitude_damping", (0,), gamma=0.05)
    tiny = Batch("tiny", circuit, noise, shots=10, seed=1, min_effect=0.02).run()
    big = Batch("big", circuit, noise, shots=4000, seed=1, min_effect=0.02).run()
    print("  10 shots  :", tiny.results[-1].comparison_result().summary())
    print("  4000 shots:", big.results[-1].comparison_result().summary())
    needed = shots_for_halfwidth(0.02)
    print(f"  要让单比特比例的 95% 区间半宽 < 0.02，建议至少 {needed} 个样本")


def main() -> None:
    out_dir = Path("data")
    demo_single_qubit()
    demo_bell()
    demo_sample_size_warning()

    circuit = Circuit(2).gate("H", 0).gate("CX", 0, 1)
    noise = NoiseModel().add("depolarizing", (0,), p=0.08).add("depolarizing", (1,), p=0.08)
    batch = Batch(
        "saved-bell", circuit, noise, shots=1000, seed=20260929, scales=[0.0, 0.5, 1.0]
    ).run()
    path = batch.save(out_dir / "batch_demo.json")
    print("=" * 68)
    print(f"批次已保存：{path}")
    print("重新加载并用保存种子复现：", "通过" if Batch.load(path).rerun_verifies() else "失败")


if __name__ == "__main__":
    main()
