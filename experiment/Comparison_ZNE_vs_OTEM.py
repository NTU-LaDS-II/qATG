"""
Comparison_ZNE_vs_OTEM.py
==========================
Reviewer response: compare Single-shot OTEM, Zero-Noise Extrapolation (ZNE),
and their combination (ZNE + OTEM) for error mitigation.

Fault model  : Transient_Qubit_Faulty_Backend  (per-qubit SX + ECR faults)
               matching OTEM_simulation-new.ipynb.
               The middle qubit (index 2) has ft_to_ff_rate_magnification=1,
               so it spends ~50 % of time in the faulty state → P_raw ≈ 0.3.
               Other qubits have magnification=1000 (quickly recover).

Target circuit: 5-qubit QFT ∘ QFT⁻¹  (ideal output |0…0⟩, P(all-zeros) = 1)
Metric        : P(all-zeros) — consistent with the rest of the paper.
ZNE method    : gate-level unitary folding at scale factors [1, 3, 5] with
                Richardson extrapolation to λ → 0.

Three scenarios
---------------
  (A) Raw              — no mitigation, scale=1
  (B) ZNE only         — Richardson extrapolation, no OTEM filtering
  (C) ZNE + OTEM       — OTEM post-selection at each scale, then extrapolation

Usage
-----
  python Comparison_ZNE_vs_OTEM.py
"""

import sys, os
sys.path.insert(0, os.path.abspath('..'))   # qatg package root
sys.path.insert(0, os.path.abspath('.'))    # fault_simulators.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import QFT

from fault_simulators import (
    Transient_Qubit_Faulty_Backend,
    get_counts_from_mem,
    get_em_counts_from_mem,
)

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

N_QUBITS      = 5
COUPLING      = [[i, i + 1] for i in range(N_QUBITS - 1)]   # linear topology
BASIS_GATES   = ['sx', 'rz', 'ecr']
FAULT_ID      = 6
SHOTS         = 10000
N_REPS        = 5
SCALE_FACTORS = [1, 3, 5]

# Per-qubit fault toggle rate
SWAP_RATE = [0.005] * N_QUBITS

# ft_to_ff_rate_magnification:
#   >1  → qubit recovers quickly from faulty state (rarely faulty overall)
#   =1  → faulty↔good equally likely  → ~50 % time faulty  → P_raw ≈ 0.3
MIDDLE = N_QUBITS // 2                                  # qubit index 2
FT_TO_FF = [1000.0 if i != MIDDLE else 1.0
            for i in range(N_QUBITS)]

print(f"Per-qubit ft_to_ff_rate_magnification: {FT_TO_FF}")
print(f"  → qubit {MIDDLE} stays ~50 % faulty; others recover quickly")

ALL_ZEROS = '0' * N_QUBITS   # ideal measurement outcome

# ---------------------------------------------------------------------------
# 1. Fault data and backends
# ---------------------------------------------------------------------------

sx_fault   = np.load('data/faults.npy', allow_pickle=True)[FAULT_ID]
ecr_faults = np.load('data/ecr_fault_list.npy', allow_pickle=True)
ecr_fault1 = ecr_faults[1][FAULT_ID]   # target qubit faulty, control good
ecr_fault2 = ecr_faults[2][FAULT_ID]   # control qubit faulty, target good
ecr_fault3 = ecr_faults[3][FAULT_ID]   # both qubits faulty
print(f"Loaded fault_id={FAULT_ID}: sx_fault shape={sx_fault.shape}, "
      f"ecr shapes={ecr_fault1.shape}")

def get_depolarizing_backend(e1, e2=None):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ['sx', 'u'])
    if e2:
        nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2),
                                       ['ecr', 'unitary'])
    return AerSimulator(noise_model=nm)

ff_backend = get_depolarizing_backend(0.001, 0.005)

tqfb = Transient_Qubit_Faulty_Backend(
    sx_fault, ecr_fault1, ecr_fault2, ecr_fault3,
    good_backend                = ff_backend,
    basis_gates                 = BASIS_GATES,
    num_qubits                  = N_QUBITS,
    coupling_map                = COUPLING,
    swap_rate                   = SWAP_RATE,
    ft_to_ff_rate_magnification = FT_TO_FF,
)

# ---------------------------------------------------------------------------
# 2. Online-test (OT) circuit
#    Tensor the single-qubit test template across all N_QUBITS qubits,
#    then transpile to the hardware basis — matching OTEM_simulation-new.ipynb.
# ---------------------------------------------------------------------------

OT_CIRCUIT_ID = 5    # best test circuit for fault_id=6 (from the paper)
test_datas = [
    QuantumCircuit.from_qasm_file(f'test_circuits/test_circuit_{i}.qasm')
    for i in range(29)
]

ot_qc = test_datas[OT_CIRCUIT_ID]
for i in range(1, N_QUBITS):
    ot_qc = ot_qc.tensor(test_datas[OT_CIRCUIT_ID])
ot_qc = transpile(ot_qc, basis_gates=BASIS_GATES, coupling_map=COUPLING)
print(f"OT circuit: {N_QUBITS} qubits, depth={ot_qc.depth()}")

# ---------------------------------------------------------------------------
# 3. Benchmark circuit: QFT ∘ QFT⁻¹  → ideal |0…0⟩
# ---------------------------------------------------------------------------

qft = QFT(N_QUBITS)
qc_bench_raw = QuantumCircuit(N_QUBITS)
qc_bench_raw.append(qft,           range(N_QUBITS)); qc_bench_raw.barrier()
qc_bench_raw.append(qft.inverse(), range(N_QUBITS))
qc_bench_raw.measure_all()
benchmark_qc = transpile(qc_bench_raw, basis_gates=BASIS_GATES,
                          coupling_map=COUPLING)
print(f"Benchmark: depth={benchmark_qc.depth()}  gates={benchmark_qc.size()}")

# ---------------------------------------------------------------------------
# 4. Unitary folding  (gate-level ZNE noise amplification)
#
#   Scale λ = 2k+1:  G → G · (G⁻¹ · G)^k
#   After folding, re-transpile to BASIS_GATES so Permanent_Faulty_Backend_Multi
#   sees no non-basis gates (prevents SXdg → extra SX decomposition that would
#   inject the fault more times than the scale factor intends).
# ---------------------------------------------------------------------------

_NON_FOLDABLE = {'barrier', 'measure', 'reset', 'delay', 'snapshot'}

def fold_gates(qc, scale_factor):
    assert scale_factor % 2 == 1 and scale_factor >= 1
    k = (scale_factor - 1) // 2
    if k == 0:
        return qc.copy()
    folded = qc.copy_empty_like()
    for inst in qc.data:
        op, qargs, cargs = inst.operation, inst.qubits, inst.clbits
        folded.append(op, qargs, cargs)
        if op.name in _NON_FOLDABLE:
            continue
        try:
            op_inv = op.inverse()
        except Exception:
            continue
        for _ in range(k):
            folded.append(op_inv, qargs, cargs)
            folded.append(op,     qargs, cargs)
    return folded


folded_qcs = {}
for lam in SCALE_FACTORS:
    qc_f = fold_gates(benchmark_qc, lam)
    # Re-transpile to basis: converts any SXdg → {sx, rz} so the fault backend
    # does not accidentally inject extra SX-fault instances during its internal
    # transpile step.
    qc_f = transpile(qc_f, basis_gates=BASIS_GATES,
                     coupling_map=COUPLING, optimization_level=0)
    folded_qcs[lam] = qc_f
    print(f"  scale={lam}: depth={qc_f.depth()}  gates={qc_f.size()}")

# ---------------------------------------------------------------------------
# 5. Extrapolation methods
# ---------------------------------------------------------------------------

def extrapolate_richardson(scales, values):
    """Lagrange interpolation through all (scales[i], values[i]) at x=0.
    Equivalent to Richardson extrapolation of order len(scales)-1.
    For scales=[1,3,5]: coefficients = [15/8, -5/4, 3/8]."""
    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)
    result = 0.0
    for i in range(len(scales)):
        c = 1.0
        for j in range(len(scales)):
            if i != j:
                c *= (0.0 - scales[j]) / (scales[i] - scales[j])
        result += c * values[i]
    return float(result)

def extrapolate_linear(scales, values):
    coeffs = np.polyfit(scales, values, 1)
    return float(np.polyval(coeffs, 0))

# ---------------------------------------------------------------------------
# 6. Simulation — run all three scenarios
# ---------------------------------------------------------------------------

def correct_rate(counts):
    total = sum(counts.values())
    return counts.get(ALL_ZEROS, 0) / total if total else 0.0


raw_rates      = []
zne_rates      = []    # list of dicts {scale: P}
otem_zne_rates = []    # list of dicts {scale: P}

print(f"\nRunning {N_REPS} reps × {len(SCALE_FACTORS)} scale factors "
      f"({SHOTS} shots each) …\n")

for rep in range(N_REPS):
    zne_rep  = {}
    otem_rep = {}

    for lam in SCALE_FACTORS:
        qc_f = folded_qcs[lam]

        # (B) ZNE only: transient backend, no OT filter
        mem_noem = tqfb.run(qc_f, shots=SHOTS, reset=(lam == SCALE_FACTORS[0]))[0]
        p_zne    = correct_rate(get_counts_from_mem(mem_noem))
        zne_rep[lam] = p_zne

        # (C) ZNE + OTEM: interleave OT shot then folded-alg shot, filter by OT
        mem_otem     = tqfb.run_single_em(ot_qc, qc_f,
                                           ot_shots=1, alg_shots=1,
                                           repeat=SHOTS)
        counts_otem  = get_em_counts_from_mem(mem_otem,
                                              shots_ot=1, ot_pass_th=1,
                                              shots_alg=1)
        p_otem       = correct_rate(counts_otem)
        otem_rep[lam] = p_otem

        print(f"  rep={rep+1}  λ={lam}:  ZNE={p_zne:.4f}  ZNE+OTEM={p_otem:.4f}")

    raw_rates.append(zne_rep[1])
    zne_rates.append(zne_rep)
    otem_zne_rates.append(otem_rep)

# ---------------------------------------------------------------------------
# 7. Extrapolate and summarise
# ---------------------------------------------------------------------------

zne_extrap      = [extrapolate_richardson(SCALE_FACTORS,
                   [zne_rates[r][l]      for l in SCALE_FACTORS]) for r in range(N_REPS)]
otem_zne_extrap = [extrapolate_richardson(SCALE_FACTORS,
                   [otem_zne_rates[r][l] for l in SCALE_FACTORS]) for r in range(N_REPS)]

scenarios = {
    '(A) Raw':        raw_rates,
    '(B) ZNE only':   zne_extrap,
    '(C) ZNE + OTEM': otem_zne_extrap,
}

print(f"\n{'Scenario':<20} {'Mean':>8} {'Std':>8}  (ideal = 1.000)")
print('-' * 42)
for label, vals in scenarios.items():
    print(f"{label:<20} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")

# ---------------------------------------------------------------------------
# 8. Plot — IEEE TCAD Paper Style
# ---------------------------------------------------------------------------
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2))

# 匹配原論文的配色 (Raw=紅, ZNE=藍, OTEM=綠)
color_raw  = 'tab:red'
color_zne  = 'tab:blue'
color_otem = 'tab:green'

# ── Left: ZNE curves ────────────────────────────────────────────────────────
mean_zne  = [np.mean([zne_rates[r][l]      for r in range(N_REPS)]) for l in SCALE_FACTORS]
mean_otem = [np.mean([otem_zne_rates[r][l] for r in range(N_REPS)]) for l in SCALE_FACTORS]
std_zne   = [np.std ([zne_rates[r][l]      for r in range(N_REPS)]) for l in SCALE_FACTORS]
std_otem  = [np.std ([otem_zne_rates[r][l] for r in range(N_REPS)]) for l in SCALE_FACTORS]

ext_zne   = np.mean(zne_extrap)
ext_otem  = np.mean(otem_zne_extrap)
raw_mean  = np.mean(raw_rates)

# 繪製 ZNE 點與線
ax_left.errorbar(SCALE_FACTORS, mean_zne, yerr=[s*2 for s in std_zne],
                 fmt='s--', color=color_zne, capsize=4, linewidth=1.5,
                 markersize=6, label='ZNE only')
ax_left.scatter([0], [ext_zne], marker='*', color=color_zne, s=180, zorder=5,
                label=f'ZNE Extrap. = {ext_zne:.3f}')

# 繪製 ZNE+OTEM 點與線
ax_left.errorbar(SCALE_FACTORS, mean_otem, yerr=[s*2 for s in std_otem],
                 fmt='o--', color=color_otem, capsize=4, linewidth=1.5,
                 markersize=6, label='ZNE + OTEM')
ax_left.scatter([0], [ext_otem], marker='*', color=color_otem, s=180, zorder=5,
                label=f'ZNE+OTEM Extrap. = {ext_otem:.3f}')

# Baseline
ax_left.axhline(raw_mean, color=color_raw, linestyle='-.', linewidth=1.5,
                label=f'Raw ($\\lambda=1$) = {raw_mean:.3f}')
ax_left.axhline(1.0, color='black', linestyle=':', linewidth=1.2)

ax_left.set_xlabel('Noise Scale Factor $\\lambda$', fontsize=12)
ax_left.set_ylabel('Success Rate $P(|0...0\\rangle)$', fontsize=12)
ax_left.set_title('(a) ZNE Curves', fontsize=12)
ax_left.set_xlim(-0.5, max(SCALE_FACTORS) + 0.5)
ax_left.set_ylim(0, 1.05)
ax_left.set_xticks([0] + SCALE_FACTORS)
ax_left.legend(fontsize=9, edgecolor='black', loc='upper right')
ax_left.grid(True, linestyle='--', alpha=0.5)
ax_left.set_axisbelow(True)

# ── Right: bar chart comparison ──────────────────────────────────────────────
labels = ['Raw\n(Unmitigated)', 'ZNE only', 'ZNE + OTEM']
means  = [raw_mean, ext_zne, ext_otem]
# Raw 的 std 取 lambda=1 的標準差，Extrapolation 的 std 取多次實驗的標準差
stds   = [np.std(raw_rates), np.std(zne_extrap), np.std(otem_zne_extrap)]
colors = [color_raw, color_zne, color_otem]

x = np.arange(len(labels))
bar_width = 0.55
bars = ax_right.bar(x, means, yerr=[s * 2 for s in stds],
                    color=colors, capsize=5, alpha=0.9, width=bar_width)

ax_right.axhline(1.0, color='black', linestyle=':', linewidth=1.2, label='Ideal = 1.000')

# 在柱狀圖上方標註數值
for bar, mean, std in zip(bars, means, stds):
    ax_right.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 2 * std + 0.02,
                  f'{mean:.3f}', ha='center', va='bottom', fontsize=11)

ax_right.set_xticks(x)
ax_right.set_xticklabels(labels, fontsize=11)
ax_right.set_ylim(0, 1.15)
ax_right.set_ylabel('Extrapolated Success Rate', fontsize=12)
ax_right.set_title('(b) Error Mitigation Performance', fontsize=12)
ax_right.grid(True, axis='y', linestyle='--', alpha=0.5)
ax_right.set_axisbelow(True)

plt.tight_layout()
out_file = 'comparison_zne_vs_otem_style.pdf'
plt.savefig(out_file, bbox_inches='tight')
print(f"\nFigure saved to {out_file}")