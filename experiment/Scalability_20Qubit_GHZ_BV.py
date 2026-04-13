"""
Scalability_20Qubit_GHZ_BV.py
==============================
Section VII (Scalability): Single-shot OTEM on 20-qubit GHZ and
Bernstein-Vazirani (BV) circuits on IBM Quantum hardware.

If preprocessing finds two disjoint sets of faulty qubits that each support a
20-qubit connected layout, the two 20-qubit circuits are composed into a single
40-qubit circuit (one per physical half) so both algorithm instances run
simultaneously on the hardware.  Results are marginalised per layout after
measurement.

Execution protocol
------------------
  1. OTEM preprocessing — select the best online-test circuit per qubit.
  2. Find up to 2 non-overlapping connected 20-qubit physical layouts centred
     on the most-faulty qubits (BFS over the device coupling map).
  3. For each benchmark (GHZ, BV), build:
       • "without OTEM" circuit  — algorithm(s) + measure
       • "with OTEM" circuit     — OT(s) + measure(testresult)
                                   + algorithm(s) + measure(measureresult)
     When 2 layouts are found both are composed into one N_TOTAL-qubit circuit
     (N_TOTAL = 20 or 40).
  4. Submit one batch job; post-process per layout.
  5. Plot a grouped bar chart per layout.

Classical registers (per layout i)
-----------------------------------
  test_l{i}  — online-test result  (only present in "with OTEM" circuits)
  meas_l{i}  — algorithm result

Usage
-----
  python Scalability_20Qubit_GHZ_BV.py \\
      --token YOUR_IBMQ_TOKEN \\
      --instance YOUR_IBMQ_INSTANCE \\
      --backend BACKEND_NAME \\
      [--preprocess-job-id JOB_ID] \\
      [--main-job-id JOB_ID] \\
      [--shots 10000] \\
      [--reps 1]
"""

import sys, os, argparse
from collections import deque
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from otem import OTEM
from result_analyze import srate   # reuse: srate(counts, nqb) = P(all-zeros)

# ---------------------------------------------------------------------------
# 0. Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--token',              required=True)
parser.add_argument('--instance',           required=True)
parser.add_argument('--backend',            required=True)
parser.add_argument('--preprocess-job-id',  default=None, dest='preprocess_job_id')
parser.add_argument('--main-job-id',        default=None, dest='main_job_id')
parser.add_argument('--shots',  type=int,   default=10000)
parser.add_argument('--reps',   type=int,   default=1)
args = parser.parse_args()

N_QUBITS   = 20
SHOTS      = args.shots
N_REPS     = args.reps
MAX_LAYOUTS = 2

TEST_CIRCUIT_IDX = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# ---------------------------------------------------------------------------
# 1. Connect to IBM Quantum
# ---------------------------------------------------------------------------

service  = QiskitRuntimeService(channel='ibm_cloud', instance=args.instance, token=args.token)
backend  = service.backend(args.backend)
sampler  = Sampler(backend)
print(f"Backend: {backend.name}  ({backend.num_qubits} qubits)")

# ---------------------------------------------------------------------------
# 2. Load test circuits and run OTEM preprocessing
# ---------------------------------------------------------------------------

test_datas = [
    QuantumCircuit.from_qasm_file(f'test_circuits/test_circuit_{i}.qasm')
    for i in TEST_CIRCUIT_IDX
]
print(f"Loaded {len(test_datas)} test circuits")

my_em          = OTEM(backend, test_datas)
preprocess_qcs = my_em.build_preprocess_test()

if args.preprocess_job_id:
    job_pre    = service.job(args.preprocess_job_id)
    pre_result = job_pre.result()
    print(f"Preprocessing result retrieved (job {args.preprocess_job_id})")
else:
    job_pre    = sampler.run(preprocess_qcs, shots=SHOTS)
    print(f"Preprocessing job submitted: {job_pre.job_id()}")
    pre_result = job_pre.result()

# reuse existing OTEM method
max_ac_qb, _, qb_f_ac, test_id = my_em.qubit_and_online_test_selection_from_result(pre_result)
print(f"Most faulty qubit: {max_ac_qb}")

# ---------------------------------------------------------------------------
# 3. Find up to MAX_LAYOUTS non-overlapping connected 20-qubit physical layouts
#    centred on the highest-scoring faulty qubits (BFS over coupling map).
# ---------------------------------------------------------------------------

def find_connected_layouts(backend, test_id, qb_f_ac, n_qubits, max_layouts):
    adj = {q: set() for q in range(backend.num_qubits)}
    for a, b in backend.coupling_map.get_edges():
        adj[a].add(b); adj[b].add(a)

    faulty = sorted(
        [(qb, qb_f_ac[qb]) for qb in range(len(test_id)) if test_id[qb] != -1],
        key=lambda x: -x[1],
    )
    print(f"Faulty qubits: {[qb for qb, _ in faulty]}")

    used, layouts = set(), []
    for center, score in faulty:
        if center in used:
            continue
        visited = {center}
        queue   = deque([center])
        layout  = [center]
        while queue and len(layout) < n_qubits:
            node = queue.popleft()
            for nb in sorted(adj[node]):
                if nb not in visited and nb not in used:
                    visited.add(nb); queue.append(nb); layout.append(nb)
                    if len(layout) >= n_qubits:
                        break
        if len(layout) < n_qubits:
            print(f"  Skipping qubit {center}: only {len(layout)} free connected qubits")
            continue
        layout = sorted(layout)
        used.update(layout)
        layouts.append(layout)
        print(f"  Layout {len(layouts)}: {layout}  (center={center}, score={score:.4f})")
        if len(layouts) >= max_layouts:
            break

    if not layouts:
        raise RuntimeError("No valid layout — no faulty qubits detected during preprocessing.")
    return layouts


ALL_LAYOUTS     = find_connected_layouts(backend, test_id, qb_f_ac, N_QUBITS, MAX_LAYOUTS)
N_LAY           = len(ALL_LAYOUTS)
N_TOTAL         = N_QUBITS * N_LAY
COMBINED_LAYOUT = [q for layout in ALL_LAYOUTS for q in layout]
print(f"\nFound {N_LAY} layout(s) → {N_TOTAL}-qubit combined circuit.")

# ---------------------------------------------------------------------------
# 4. Build per-layout OT circuits (N_QUBITS wide, identity on non-faulty qubits)
# ---------------------------------------------------------------------------

def build_ot_circuit(test_datas, test_id, physical_layout):
    """N_QUBITS-wide OT: local index i ↔ physical_layout[i]."""
    qc, active = QuantumCircuit(len(physical_layout)), []
    for local_idx, qb_phys in enumerate(physical_layout):
        tid = test_id[qb_phys] if qb_phys < len(test_id) else -1
        if tid != -1:
            q = test_datas[tid].copy()
            q.remove_final_measurements()
            for inst in q.data:
                qc.append(inst.operation, [local_idx])
            active.append(qb_phys)
    return qc, active


ot_circuits = []
for li, layout in enumerate(ALL_LAYOUTS):
    qc_ot, active = build_ot_circuit(test_datas, test_id, layout)
    ot_circuits.append(qc_ot)
    print(f"Layout {li+1} OT — active physical qubits: {active}")

# ---------------------------------------------------------------------------
# 5. Build benchmark circuits (forward ∘ inverse → ideal output |0…0⟩)
# ---------------------------------------------------------------------------

def build_ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1): qc.cx(i, i + 1)
    qc.barrier()
    for i in reversed(range(n - 1)): qc.cx(i, i + 1)
    qc.h(0)
    return qc


def build_bv_circuit(n, secret=None):
    if secret is None:
        secret = [i % 2 for i in range(n)]

    def bv_core(n, secret):
        qc = QuantumCircuit(n)
        qc.h(range(n)); qc.barrier()
        for i in range(n):
            if secret[i]: qc.z(i)
        qc.barrier(); qc.h(range(n))
        return qc

    fwd = bv_core(n, secret)
    qc  = QuantumCircuit(n)
    qc.compose(fwd, inplace=True); qc.barrier()
    qc.compose(fwd.inverse(), inplace=True)
    return qc


# Logical benchmark circuits — transpiled per-layout inside the build helpers
benchmarks = {'GHZ': build_ghz_circuit(N_QUBITS), 'BV': build_bv_circuit(N_QUBITS)}
print("Benchmark circuits built.")

# ---------------------------------------------------------------------------
# 6. Circuit builders
#
#    Both functions produce an N_TOTAL-qubit circuit.
#    For N_LAY=1: identical to the original 20-qubit design.
#    For N_LAY=2: layout 0 occupies qubits 0–19, layout 1 occupies qubits 20–39.
#
#    Classical registers per layout i:
#      "without OTEM":  meas_l{i}
#      "with OTEM":     test_l{i}  +  meas_l{i}
#
#    optimization_level=1 lets the transpiler route within each layout's
#    connected subgraph (required because BFS layouts are not guaranteed chains).
# ---------------------------------------------------------------------------

def build_without_otem(qc_alg, layouts, backend):
    qc = QuantumCircuit(N_TOTAL)
    for i in range(N_LAY):
        qc.compose(qc_alg, qubits=range(i * N_QUBITS, (i + 1) * N_QUBITS), inplace=True)
    for i in range(N_LAY):
        cr = ClassicalRegister(N_QUBITS, name=f'meas_l{i}')
        qc.add_register(cr)
        qc.measure(range(i * N_QUBITS, (i + 1) * N_QUBITS), cr)
    return transpile(qc, backend, initial_layout=COMBINED_LAYOUT, optimization_level=1)


def build_with_otem(qc_alg, ot_circuits, layouts, backend):
    qc = QuantumCircuit(N_TOTAL)
    # — OT layer —
    for i, qc_ot in enumerate(ot_circuits):
        qc.compose(qc_ot, qubits=range(i * N_QUBITS, (i + 1) * N_QUBITS), inplace=True)
    for i in range(N_LAY):
        cr = ClassicalRegister(N_QUBITS, name=f'test_l{i}')
        qc.add_register(cr)
        qc.measure(range(i * N_QUBITS, (i + 1) * N_QUBITS), cr)
    # — Algorithm layer —
    for i in range(N_LAY):
        qc.compose(qc_alg, qubits=range(i * N_QUBITS, (i + 1) * N_QUBITS), inplace=True)
    for i in range(N_LAY):
        cr = ClassicalRegister(N_QUBITS, name=f'meas_l{i}')
        qc.add_register(cr)
        qc.measure(range(i * N_QUBITS, (i + 1) * N_QUBITS), cr)
    return transpile(qc, backend, initial_layout=COMBINED_LAYOUT, optimization_level=1)

# ---------------------------------------------------------------------------
# 7. Assemble all circuits
#
#    Order: for each rep → for each benchmark → [no_otem, with_otem]
#    Total: N_REPS × n_benchmarks × 2
# ---------------------------------------------------------------------------

all_qcs, circuit_index = [], {}
n_bench = len(benchmarks)

for rep in range(N_REPS):
    for bi, (name, qc_alg) in enumerate(benchmarks.items()):
        flat_base = rep * n_bench * 2 + bi * 2
        circuit_index[(rep, name, 'no')]  = flat_base
        circuit_index[(rep, name, 'yes')] = flat_base + 1
        all_qcs.append(build_without_otem(qc_alg, ALL_LAYOUTS, backend))
        all_qcs.append(build_with_otem(qc_alg, ot_circuits, ALL_LAYOUTS, backend))

print(f"\nTotal circuits: {len(all_qcs)}  "
      f"({N_REPS} reps × {n_bench} benchmarks × 2 variants, each {N_TOTAL} qubits)")

# ---------------------------------------------------------------------------
# 8. Submit single batch job (or retrieve from cache)
# ---------------------------------------------------------------------------

if args.main_job_id:
    job_main = service.job(args.main_job_id)
    results  = list(job_main.result())
    print(f"Main job result retrieved (job {args.main_job_id})")
else:
    job_main = sampler.run(all_qcs, shots=SHOTS)
    print(f"Main job submitted: {job_main.job_id()}")
    results  = list(job_main.result())

# ---------------------------------------------------------------------------
# 9. Post-processing: success rates per layout
#
#    no_otem:   reuse srate() from result_analyze — P(all-zeros in meas_l{i})
#    with_otem: keep only shots where ALL test registers are all-zeros
#               (every OT on every layout passed), then compute P(all-zeros
#               in meas_l{i}) for the layout of interest.
# ---------------------------------------------------------------------------

ALL_ZEROS = '0' * N_QUBITS


def get_no_otem_rate(result_data, layout_idx):
    counts = getattr(result_data, f'meas_l{layout_idx}').get_counts()
    return srate(counts, N_QUBITS)   # from result_analyze


def get_with_otem_rate(result_data, layout_idx):
    test_bss = [getattr(result_data, f'test_l{i}').get_bitstrings() for i in range(N_LAY)]
    meas_bs  = getattr(result_data, f'meas_l{layout_idx}').get_bitstrings()
    filtered = {}
    for shot, m in enumerate(meas_bs):
        if all(test_bss[li][shot] == ALL_ZEROS for li in range(N_LAY)):
            filtered[m] = filtered.get(m, 0) + 1
    total = sum(filtered.values())
    return filtered.get(ALL_ZEROS, 0) / total if total else 0.0


# rates[layout_idx][bench_name]['no_otem' | 'with_otem'] = list over reps
rates_all = [
    {name: {'no_otem': [], 'with_otem': []} for name in benchmarks}
    for _ in ALL_LAYOUTS
]

for rep in range(N_REPS):
    for name in benchmarks:
        rd_no  = results[circuit_index[(rep, name, 'no')]].data
        rd_yes = results[circuit_index[(rep, name, 'yes')]].data
        for li in range(N_LAY):
            rates_all[li][name]['no_otem'].append(get_no_otem_rate(rd_no,  li))
            rates_all[li][name]['with_otem'].append(get_with_otem_rate(rd_yes, li))

# Print summary
bench_names = list(benchmarks.keys())
for li, (layout, rates) in enumerate(zip(ALL_LAYOUTS, rates_all)):
    print(f"\nLayout {li+1}: physical qubits {layout}")
    print(f"  {'Benchmark':<8} {'Without OTEM':>22} {'With OTEM':>22}")
    print(f"  {'-'*54}")
    for name in bench_names:
        nm = np.mean(rates[name]['no_otem']);  ns = np.std(rates[name]['no_otem'])
        ym = np.mean(rates[name]['with_otem']); ys = np.std(rates[name]['with_otem'])
        print(f"  {name:<8} {nm:.4f} ± {ns:.4f}          {ym:.4f} ± {ys:.4f}")

# ---------------------------------------------------------------------------
# 10. Plot — one subplot per layout
# ---------------------------------------------------------------------------

x, bar_w = np.arange(len(bench_names)), 0.30
fig, axes = plt.subplots(1, N_LAY, figsize=(7 * N_LAY, 5), squeeze=False)

for li, (layout, rates) in enumerate(zip(ALL_LAYOUTS, rates_all)):
    ax = axes[0][li]
    no_m  = [np.mean(rates[n]['no_otem'])   for n in bench_names]
    no_s  = [np.std(rates[n]['no_otem'])    for n in bench_names]
    yes_m = [np.mean(rates[n]['with_otem']) for n in bench_names]
    yes_s = [np.std(rates[n]['with_otem'])  for n in bench_names]

    b_no  = ax.bar(x - bar_w/2, no_m,  bar_w, yerr=[s*2 for s in no_s],
                   label='Without OTEM', color='tab:red',   capsize=5, alpha=0.85)
    b_yes = ax.bar(x + bar_w/2, yes_m, bar_w, yerr=[s*2 for s in yes_s],
                   label='With OTEM',    color='tab:green', capsize=5, alpha=0.85)

    for bar in list(b_no) + list(b_yes):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(bench_names, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Success Rate  P(|0…0⟩)', fontsize=12)
    ax.set_title(
        f'Layout {li+1}: qubits {layout[:3]}…\n'
        f'{backend.name}  ({N_REPS} reps × {SHOTS} shots)',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

plt.suptitle(
    f'Scalability: {N_TOTAL}-Qubit ({N_LAY}×{N_QUBITS}Q) GHZ & BV — {backend.name}',
    fontsize=13, y=1.02,
)
plt.tight_layout()
out_file = f'scalability_{N_TOTAL}qubit_ghz_bv_{backend.name}.pdf'
plt.savefig(out_file, bbox_inches='tight')
plt.show()
print(f"Figure saved to {out_file}")
