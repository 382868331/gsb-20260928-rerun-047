"""测量统计：从有限样本估计概率，并避免把统计波动当成确定效果。

提供：
- 比特串计数 / 单比特边际概率；
- Wilson 二项比例置信区间；
- 理想分布与实测分布的卡方拟合优度检验（不依赖 scipy）；
- 样本量充分性判断（最大区间半宽是否小于效应尺度）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GLOBAL_Z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def int_to_bits(value: int, n_qubits: int) -> tuple[int, ...]:
    """整数索引 -> 比特元组，比特 0 为高位。"""
    return tuple((value >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits))


def counts_to_array(counts: dict[str | int, int], n_qubits: int) -> np.ndarray:
    """把 {比特串/整数: 次数} 归一为 2^n 的计数数组。"""
    arr = np.zeros(2**n_qubits, dtype=np.int64)
    for key, value in counts.items():
        if isinstance(key, str):
            idx = int(key, 2)
        else:
            idx = int(key)
        arr[idx] += value
    return arr


def marginal_probs(
    samples: np.ndarray, n_qubits: int, qubits: tuple[int, ...] | None = None
) -> np.ndarray:
    """样本数组 -> 指定比特上的经验分布（默认全部比特）。"""
    shots = len(samples)
    if qubits is None:
        qubits = tuple(range(n_qubits))
    if shots == 0:
        return np.full(2 ** len(qubits), np.nan)
    index = np.zeros(shots, dtype=np.int64)
    for q in qubits:
        index = (index << 1) | ((samples >> (n_qubits - 1 - q)) & 1)
    probs = np.bincount(index, minlength=2 ** len(qubits)) / shots
    return probs


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson 得分区间；n=0 时返回 (nan, nan)。"""
    if n <= 0:
        return float("nan"), float("nan")
    z = GLOBAL_Z.get(round(confidence, 2), 1.96)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return center - half, center + half


@dataclass
class ComparisonResult:
    """理想 vs 实测分布比较结果。"""

    chi_square: float
    dof: int
    p_value: float
    significant: bool
    max_interval_halfwidth: float
    effect_size: float
    enough_shots: bool

    def summary(self) -> str:
        if not self.enough_shots:
            verdict = "样本不足：统计波动可能与真实效应同量级，暂不下结论"
        elif self.significant:
            verdict = "存在统计显著偏离：噪声效应大于抽样波动"
        else:
            verdict = "无统计显著偏离：暂未检测到噪声效应"
        return (
            f"{verdict} (chi2={self.chi_square:.3f}, dof={self.dof}, "
            f"p={self.p_value:.4f}, 效应L1={self.effect_size:.4f}, "
            f"区间半宽={self.max_interval_halfwidth:.4f})"
        )


def _chi2_sf(x: float, dof: int) -> float:
    """卡方分布生存函数的正则下不完全伽马表示。"""
    if x <= 0:
        return 1.0
    return _gammaincc(dof / 2.0, x / 2.0)


def _gammaincc(a: float, x: float) -> float:
    """正则上不完全伽马函数 Q(a,x)（级数 + 连分式，数值稳定）。"""
    if x < a + 1.0:
        # 下不完全伽马级数 P(a,x)
        term = 1.0 / a
        total = term
        for n in range(1, 200):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # 上不完全伽马连分式（Lentz，Numerical Recipes gcf）
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    f = d
    for i in range(1, 200):
        an = i * (a - i)
        b += 2.0
        d = an * d + b
        if abs(c) < tiny:
            c = tiny
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        delta_f = d * c
        f *= delta_f
        if abs(delta_f - 1.0) < 1e-14:
            break
    return f * math.exp(-x + a * math.log(x) - math.lgamma(a))


def compare_distributions(
    ideal_probs: np.ndarray,
    observed_counts: np.ndarray,
    confidence: float = 0.95,
    alpha: float = 0.05,
    min_effect: float = 0.02,
) -> ComparisonResult:
    """比较理想概率分布与实测计数。

    - 卡方检验使用 Bonferroni 式保守性由调用方在多输出时控制；
      这里对单个分布做标准检验。
    - enough_shots: 95% Wilson 区间的最大半宽必须小于 min_effect，
      避免样本太少时把噪声误判为“效果”。
    """
    observed_counts = np.asarray(observed_counts, dtype=np.int64)
    shots = int(observed_counts.sum())
    ideal_probs = np.asarray(ideal_probs, dtype=float)
    expected = ideal_probs * shots

    # 把期望频数过小（<5）的桶并入“其他”，避免零概率桶造成退化；
    # 理想分布中严格为零的桶单独累计，出现观测即计入该合并桶。
    big = expected >= 5.0
    if not np.any(big):
        big = expected == expected.max()
    other_expected = float(expected[~big].sum())
    other_observed = int(observed_counts[~big].sum())
    exp_list = list(expected[big])
    obs_list = list(observed_counts[big].astype(float))
    if other_expected > 0 or other_observed > 0:
        exp_list.append(max(other_expected, 1e-12))
        obs_list.append(float(other_observed))
    exp_arr = np.asarray(exp_list)
    obs_arr = np.asarray(obs_list)
    dof = max(len(exp_arr) - 1, 1)
    if shots == 0:
        chi2 = float("nan")
        p_value = float("nan")
        significant = False
        max_half = float("inf")
        effect = float("inf")
        enough = False
        return ComparisonResult(chi2, dof, p_value, significant, max_half, effect, enough)

    diff = observed_counts.astype(float) - expected
    chi2 = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
    p_value = 0.0 if math.isinf(chi2) else _chi2_sf(chi2, dof)

    empirical = observed_counts / shots
    effect = float(np.sum(np.abs(empirical - ideal_probs)))
    halfwidths = [
        wilson_interval(int(observed_counts[i]), shots, confidence) for i in range(len(ideal_probs))
    ]
    max_half = max(hi - (lo + hi) / 2 for lo, hi in halfwidths)
    enough = max_half < min_effect
    significant = bool(p_value < alpha) and enough
    return ComparisonResult(chi2, dof, p_value, significant, max_half, effect, enough)


def shots_for_halfwidth(target_halfwidth: float, p: float = 0.5, confidence: float = 0.95) -> int:
    """达到给定 Wilson 区间半宽所需的近似样本量（p=0.5 为最保守情形）。"""
    z = GLOBAL_Z.get(round(confidence, 2), 1.96)
    n = (z**2 * p * (1 - p)) / target_halfwidth**2
    return math.ceil(n)
