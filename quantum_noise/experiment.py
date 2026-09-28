"""实验批次：固定线路、噪声与种子，重复运行并可保存复现。

设计要点：
- 每次批次同时给出理想线路与有噪线路的精确概率和有限采样结果，
  便于并排检查；
- 种子（seed）与完整配置一并保存到 JSON，重新加载后逐比特串
  复现完全相同的样本；
- 噪声强度 scale 会乘到 p/gamma 上，为后续误差缓解
  （如零噪声外推）预留接口，但本模块不做线路编辑。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .circuit import Circuit, NoiseModel, NoiseStep
from .simulator import DensityMatrixSimulator
from .statistics import compare_distributions, ComparisonResult


def scale_noise(noise: NoiseModel | None, scale: float) -> NoiseModel | None:
    """把噪声强度参数（p/gamma）整体乘以 scale（scale=0 表示无噪）。"""
    if noise is None or scale == 1.0:
        return noise
    scaled = NoiseModel()
    for step in noise.steps:
        params = dict(step.params)
        for key in ("p", "gamma"):
            if key in params:
                value = float(params[key]) * scale
                params[key] = min(max(value, 0.0), 1.0)
        scaled.steps.append(NoiseStep(step.channel, step.qubits, params, step.before))
    return scaled


@dataclass
class ExperimentResult:
    """单次实验（一个噪声强度）的结果。"""

    scale: float
    ideal_probs: list[float]
    noisy_probs: list[float]
    ideal_counts: list[int]
    noisy_counts: list[int]
    comparison: dict

    def comparison_result(self) -> ComparisonResult:
        c = self.comparison
        return ComparisonResult(**c)


@dataclass
class Batch:
    """一个可复现实验批次：同一线路在若干噪声强度下的结果。"""

    name: str
    circuit: Circuit
    noise: NoiseModel | None
    shots: int
    seed: int
    scales: list[float] = field(default_factory=lambda: [1.0])
    confidence: float = 0.95
    min_effect: float = 0.02
    results: list[ExperimentResult] = field(default_factory=list)

    def run(self) -> "Batch":
        """执行批次；重复调用同配置同种子会得到逐样本一致的结果。"""
        if self.shots < 0:
            raise ValueError("shots 不能为负")
        # 父生成器派生子流：理想/有噪互不干扰，整体仍由单一种子决定
        root = np.random.default_rng(self.seed)
        self.results = []
        # 理想线路只与线路有关，在所有 scale 间共享
        ideal_sim = DensityMatrixSimulator(self.circuit.n_qubits)
        ideal_rho = ideal_sim.run(self.circuit, None)
        ideal_probs = ideal_sim.probabilities()
        for scale in self.scales:
            streams = root.spawn(2)
            ideal_samples = ideal_sim.sample(self.shots, streams[0])
            scaled_noise = scale_noise(self.noise, scale)
            noisy_sim = DensityMatrixSimulator(self.circuit.n_qubits)
            noisy_sim.run(self.circuit, scaled_noise)
            noisy_probs = noisy_sim.probabilities()
            noisy_samples = noisy_sim.sample(self.shots, streams[1])

            dim = 2**self.circuit.n_qubits
            ideal_counts = np.bincount(ideal_samples, minlength=dim).tolist()
            noisy_counts = np.bincount(noisy_samples, minlength=dim).tolist()
            cmp_result = compare_distributions(
                ideal_probs,
                np.asarray(noisy_counts),
                confidence=self.confidence,
                min_effect=self.min_effect,
            )
            self.results.append(
                ExperimentResult(
                    scale=scale,
                    ideal_probs=[float(v) for v in ideal_probs],
                    noisy_probs=[float(v) for v in noisy_probs],
                    ideal_counts=ideal_counts,
                    noisy_counts=noisy_counts,
                    comparison=asdict(cmp_result),
                )
            )
        return self

    # ---- 持久化 -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "circuit": self.circuit.to_dict(),
            "noise": self.noise.to_dict() if self.noise is not None else None,
            "shots": self.shots,
            "seed": self.seed,
            "scales": self.scales,
            "confidence": self.confidence,
            "min_effect": self.min_effect,
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def from_dict(data: dict) -> "Batch":
        batch = Batch(
            name=data["name"],
            circuit=Circuit.from_dict(data["circuit"]),
            noise=NoiseModel.from_dict(data["noise"]) if data.get("noise") else None,
            shots=data["shots"],
            seed=data["seed"],
            scales=list(data.get("scales", [1.0])),
            confidence=data.get("confidence", 0.95),
            min_effect=data.get("min_effect", 0.02),
        )
        batch.results = [ExperimentResult(**r) for r in data.get("results", [])]
        return batch

    @staticmethod
    def load(path: str | Path) -> "Batch":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Batch.from_dict(data)

    def rerun_verifies(self) -> bool:
        """用保存的种子重新执行，核对计数是否逐 scale 一致。"""
        fresh = Batch(
            self.name, self.circuit, self.noise, self.shots, self.seed,
            list(self.scales), self.confidence, self.min_effect,
        ).run()
        for old, new in zip(self.results, fresh.results):
            if old.noisy_counts != new.noisy_counts or old.ideal_counts != new.ideal_counts:
                return False
        return True
