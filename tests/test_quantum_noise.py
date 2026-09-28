"""核心性质测试。"""
import numpy as np
import pytest

from quantum_noise import (
    Circuit,
    NoiseModel,
    run_circuit,
    DensityMatrixSimulator,
    bit_flip,
    phase_flip,
    amplitude_damping,
    two_qubit_depolarizing,
    marginal_probs,
    wilson_interval,
    compare_distributions,
    shots_for_halfwidth,
    Batch,
    scale_noise,
)


def assert_valid_density_matrix(rho):
    assert abs(np.trace(rho) - 1.0) < 1e-9
    assert np.abs(rho - rho.conj().T).max() < 1e-9
    eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    assert eigvals.min() > -1e-9


KRAUS_FAMILIES = [
    (bit_flip(0.37), 1),
    (phase_flip(0.21), 1),
    (amplitude_damping(0.63), 1),
    (two_qubit_depolarizing(0.42), 2),
]


@pytest.mark.parametrize("ops,arity", KRAUS_FAMILIES)
def test_kraus_completeness(ops, arity):
    total = sum(k.conj().T @ k for k in ops)
    assert np.allclose(total, np.eye(2**arity))


@pytest.mark.parametrize("channel", ["bit_flip", "depolarizing", "phase_flip", "amplitude_damping"])
def test_valid_probabilities_under_noise(channel):
    circuit = Circuit(2).gate("H", 0).gate("CX", 0, 1)
    param = {"p": 0.25} if channel != "amplitude_damping" else {"gamma": 0.25}
    noise = NoiseModel().add(channel, (0,), **param).add(channel, (1,), **param)
    rho = run_circuit(circuit, noise)
    assert_valid_density_matrix(rho)
    probs = np.real(np.diag(rho))
    assert np.all(probs >= -1e-12)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_known_gate_results():
    # Bell 态：|00> 与 |11> 各 1/2
    bell = Circuit(2).gate("H", 0).gate("CX", 0, 1)
    probs = np.real(np.diag(run_circuit(bell)))
    assert np.allclose(probs, [0.5, 0, 0, 0.5])

    # Rx(pi)|0> = -i|1>
    rx = Circuit(1).gate("RX", 0, param=np.pi)
    assert np.allclose(np.real(np.diag(run_circuit(rx))), [0, 1], atol=1e-10)


def test_noise_position_changes_computation():
    # X 之后的比特翻转会把 |1> 拉回 |0>；X 之前的翻转也作用在 |0> 上，
    # 二者解析结果相同但与“放在两层中间、先翻再 X”的另一种位置不同。
    # 这里用 H 前后的相位翻转：Z 在 H 之前不改变计算基概率，
    # Z 在 H 之后会把 |+> 变成 |->，再经末端无噪测量仍为 50/50；
    # 真正区分位置的是 H -> Z -> H 结构。
    circuit = Circuit(1).gate("H", 0).gate("H", 0)
    # 相位翻转夹在两个 H 之间：等价 X，最终测量几乎全为 |1>
    middle = NoiseModel().add("phase_flip", (0,), before=1, p=1.0)
    # 相位翻转放在末端：最终状态 |0>
    terminal = NoiseModel().add("phase_flip", (0,), p=1.0)
    rho_middle = run_circuit(circuit, middle)
    rho_terminal = run_circuit(circuit, terminal)
    assert not np.allclose(rho_middle, rho_terminal)
    assert np.allclose(np.real(np.diag(rho_middle)), [0.0, 1.0], atol=1e-10)
    assert np.allclose(np.real(np.diag(rho_terminal)), [1.0, 0.0], atol=1e-10)


def test_noise_order_changes_computation():
    # 振幅阻尼与去极化一般不交换；同一锚点上声明顺序决定作用次序。
    circuit = Circuit(1).gate("RY", 0, param=np.pi / 2)
    ad_first = NoiseModel()
    ad_first.add("amplitude_damping", (0,), gamma=0.4)
    ad_first.add("depolarizing", (0,), p=0.4)
    dep_first = NoiseModel()
    dep_first.add("depolarizing", (0,), p=0.4)
    dep_first.add("amplitude_damping", (0,), gamma=0.4)
    rho_ad = run_circuit(circuit, ad_first)
    rho_dep = run_circuit(circuit, dep_first)
    assert not np.allclose(rho_ad, rho_dep)
    assert_valid_density_matrix(rho_ad)
    assert_valid_density_matrix(rho_dep)


def test_short_entangled_circuit_noise_reduces_correlation():
    # Bell 后双比特去极化应降低 |00>/|11> 之外对角概率的集中程度
    bell = Circuit(2).gate("H", 0).gate("CX", 0, 1)
    ideal = np.real(np.diag(run_circuit(bell)))
    noise = NoiseModel().add("two_qubit_depolarizing", (0, 1), p=0.8)
    noisy = np.real(np.diag(run_circuit(bell, noise)))
    assert noisy[0] < ideal[0]
    assert noisy[1] > 0.0 and noisy[2] > 0.0


def test_marginal_and_wilson():
    samples = np.array([0, 0, 1, 3], dtype=np.int64)
    probs = marginal_probs(samples, 2)
    assert np.allclose(probs, [0.5, 0.25, 0.0, 0.25])
    lo, hi = wilson_interval(3, 10)
    assert 0.0 < lo < 0.3 < hi < 1.0


def test_low_shots_not_treated_as_effect():
    rng = np.random.default_rng(7)
    ideal = np.array([0.5, 0.5])
    # 10 个样本从理想分布抽取，即使卡方偶尔偏离也不得判为确定效应
    flags = []
    for seed in range(200):
        counts = np.bincount(rng.choice(2, size=10, p=ideal), minlength=2)
        result = compare_distributions(ideal, counts, min_effect=0.02)
        flags.append(result.significant)
    assert not any(flags)
    # 样本充分时真实偏移可被检出
    noisy = np.array([0.3, 0.7])
    counts = np.bincount(rng.choice(2, size=4000, p=noisy), minlength=2)
    result = compare_distributions(ideal, counts, min_effect=0.02)
    assert result.significant and result.enough_shots


def test_shots_guidance_reasonable():
    assert shots_for_halfwidth(0.05) > 300
    assert shots_for_halfwidth(0.2) < 50


def test_batch_seed_reproducibility(tmp_path):
    circuit = Circuit(2).gate("H", 0).gate("CX", 0, 1)
    noise = NoiseModel().add("depolarizing", (0,), p=0.1)
    batch = Batch("bell-depol", circuit, noise, shots=500, seed=20260929, scales=[0.0, 1.0])
    batch.run()
    assert batch.rerun_verifies()

    path = batch.save(tmp_path / "batch.json")
    loaded = Batch.load(path)
    assert loaded.seed == 20260929
    assert loaded.rerun_verifies()
    before = [r.noisy_counts for r in loaded.results]
    loaded.run()
    after = [r.noisy_counts for r in loaded.results]
    assert before == after
    # scale=0 时退化为理想分布
    assert np.allclose(batch.results[0].noisy_probs, batch.results[0].ideal_probs)


def test_scaled_noise_interpolation():
    noise = NoiseModel().add("bit_flip", (0,), p=0.4)
    half = scale_noise(noise, 0.5)
    assert half.steps[0].params["p"] == pytest.approx(0.2)
    assert scale_noise(noise, 3.0).steps[0].params["p"] == 1.0


def test_sampling_matches_distribution():
    circuit = Circuit(1).gate("H", 0)
    sim = DensityMatrixSimulator(1)
    sim.run(circuit)
    rng = np.random.default_rng(0)
    samples = sim.sample(20000, rng)
    freq = np.bincount(samples, minlength=2) / 20000
    assert abs(freq[0] - 0.5) < 0.02
