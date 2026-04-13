"""
Hardware_Overhead_Trend_Test.py
================================
Section VI-C (Table I extension): SOTEM vs DOTEM execution-time overhead
on IBM Quantum hardware, across device generations.

Motivation
----------
Dynamic-circuit OTEM (DOTEM) uses mid-circuit measurement + classical
feed-forward to conditionally skip the algorithm when the online test
fails.  On older hardware the classical feed-forward latency dominates;
on newer hardware that latency has shrunk significantly.  Running this
script on successive device generations quantifies the trend for Table I.

Execution protocol  (mirrors Section VI-C of the paper)
--------------------------------------------------------
For each round r = 1 … N_ROUNDS:
  ┌─── Qiskit Runtime Session ────────────────────────────────────────┐
  │  1. Original   — target circuit, no mitigation                    │
  │  2. SOTEM      — OT prepended, results post-filtered              │
  │  3. DOTEM      — OT prepended, algorithm gated on classical bit   │
  └───────────────────────────────────────────────────────────────────┘
Both wall-clock time (time.time) and IBM-reported QPU usage seconds
(job.metrics()['usage']['seconds']) are recorded for each job.

Output
------
• Console: summary table (mean ± std of execution time, overhead ratio)
• File: hardware_overhead_trend_<backend>.json  — raw per-round data
• Plot: hardware_overhead_trend_<backend>.pdf   — grouped bar chart

Usage
-----
  python Hardware_Overhead_Trend_Test.py \\
      --token YOUR_IBMQ_TOKEN \\
      --instance YOUR_IBMQ_INSTANCE \\
      --backend BACKEND_NAME \\
      [--preprocess-job-id JOB_ID | --no-reuse-preprocess] \\
      [--rounds 1] \\
      [--shots 10000]
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.abspath('..'))   # qatg package root
sys.path.insert(0, os.path.abspath('.'))    # otem.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler

from otem import OTEM

# ---------------------------------------------------------------------------
# 0. Parse command-line arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Hardware overhead trend test: SOTEM vs DOTEM across device generations.'
)
parser.add_argument('--token',    required=True, help='IBM Quantum API token')
parser.add_argument('--instance', required=True, help='IBM Quantum instance (hub/group/project)')
parser.add_argument('--backend',  required=True,
                    help='Target backend name')
parser.add_argument('--rounds',   type=int, default=1,
                    help='Number of Original→SOTEM→DOTEM rounds (default: 1)')
parser.add_argument('--shots',    type=int, default=10000,
                    help='Shots per circuit per round (default: 10000)')
parser.add_argument('--preprocess-job-id', default=None,
                    dest='preprocess_job_id',
                    help='Job ID of a previously submitted preprocessing job to reuse')
parser.add_argument('--no-reuse-preprocess', action='store_true',
                    dest='no_reuse_preprocess',
                    help='Force live preprocessing even if --preprocess-job-id is given')
args = parser.parse_args()

TOKEN        = args.token
INSTANCE     = args.instance
BACKEND_NAME = args.backend
N_ROUNDS     = args.rounds
SHOTS        = args.shots

REUSE_PREPROCESS_JOB = (args.preprocess_job_id is not None) and (not args.no_reuse_preprocess)
PREPROCESS_JOB_ID    = args.preprocess_job_id

TEST_CIRCUIT_IDX = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24]

# ---------------------------------------------------------------------------
# 1. Connect to IBM Quantum
# ---------------------------------------------------------------------------

service = QiskitRuntimeService(
    channel='ibm_cloud',
    instance=INSTANCE,
    token=TOKEN,
)
backend = service.backend(BACKEND_NAME)
print(f"Backend : {backend.name}  ({backend.num_qubits} qubits)")
print(f"Version : {getattr(backend, 'backend_version', 'n/a')}")
_cfg = backend.configuration()
_dyn = getattr(_cfg, 'dynamic_reprate_enabled', 'check manually')
print(f"Dynamic circuits supported: {_dyn}")

# ---------------------------------------------------------------------------
# 2. Load test circuits and run OTEM preprocessing
# ---------------------------------------------------------------------------

test_datas = [
    QuantumCircuit.from_qasm_file(f'test_circuits/test_circuit_{i}.qasm')
    for i in TEST_CIRCUIT_IDX
]
print(f"\nLoaded {len(test_datas)} test circuits")

my_em          = OTEM(backend, test_datas)
preprocess_qcs = my_em.build_preprocess_test()

if REUSE_PREPROCESS_JOB:
    job_pre    = service.job(PREPROCESS_JOB_ID)
    pre_result = job_pre.result()
    print(f"Preprocessing result retrieved (job {PREPROCESS_JOB_ID})")
else:
    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        job_pre = sampler.run(preprocess_qcs, shots=SHOTS)
        print(f"Preprocessing job submitted: {job_pre.job_id()}")
        pre_result = job_pre.result()

_, _, _, test_id = my_em.qubit_and_online_test_selection_from_result(pre_result)
print(f"Per-qubit test-circuit selection: {test_id}")

# ---------------------------------------------------------------------------
# 3. Build OT1 (mitigation test) and OT2 (target algorithm)
#
#    OT1 — the online-test circuit whose result gates the algorithm.
#    OT2 — a second test circuit that acts as a stand-in for the quantum
#           algorithm.  Using another test circuit as the algorithm means
#           the overhead comparison is algorithm-agnostic; any multi-qubit
#           application circuit could be substituted here.
#
#    Both are assembled in parallel: one single-qubit OT template per
#    physical qubit selected during preprocessing (qubits with test_id==-1
#    get identity padding).
# ---------------------------------------------------------------------------

nqb = backend.num_qubits   # full device width

def build_ot_parallel(test_datas, test_id, n_qubits):
    """Assemble n_qubits-wide OT circuit by tensoring single-qubit OT templates.
    Returns (circuit_without_measurements, list_of_active_qubit_indices).
    """
    qc     = QuantumCircuit(0)
    active = []
    for qb in range(n_qubits):
        tid = test_id[qb] if qb < len(test_id) else -1
        if tid != -1:
            q2 = test_datas[tid].copy()
            q2.remove_final_measurements()
            qc = q2.tensor(qc)
            active.append(qb)
    return qc, active

qc_ot1, layouts = build_ot_parallel(test_datas, test_id, nqb)
n_lay = len(layouts)
print(f"\nSelected {n_lay} active qubits: {layouts}")

# OT2: use the same per-qubit selection → another test circuit as algorithm
qc_ot2, _ = build_ot_parallel(test_datas, test_id, nqb)

# ---------------------------------------------------------------------------
# 4. Build the three circuit variants
#    All three are transpiled to the physical backend at optimization_level=0
#    so that gate counts are stable and timing comparisons are fair.
# ---------------------------------------------------------------------------

def build_original(qc_alg, layouts, backend):
    """Plain algorithm circuit with measurement — no online test."""
    qc  = qc_alg.copy()
    crm = ClassicalRegister(n_lay, name='measureresult')
    qc.add_register(crm)
    qc.measure(range(n_lay), crm)
    return transpile(qc, backend, initial_layout=layouts, optimization_level=0)


def build_sotem(qc_alg, qc_ot, layouts, backend):
    """Single-shot OTEM (SOTEM):
      1. OT circuit + measure into 'testresult'
      2. Algorithm + measure into 'measureresult'
    Post-processing (offline) keeps only shots where testresult == 0…0.
    No hardware feed-forward; classical filtering is entirely off-device.
    """
    qc  = qc_ot.copy()
    crt = ClassicalRegister(n_lay, name='testresult')
    qc.add_register(crt)
    qc.measure(range(n_lay), crt)

    for inst in qc_alg.data:
        qc.append(inst.operation, qargs=inst.qubits)

    crm = ClassicalRegister(n_lay, name='measureresult')
    qc.add_register(crm)
    qc.measure(range(n_lay), crm)
    return transpile(qc, backend, initial_layout=layouts, optimization_level=0)


def build_dotem(qc_alg, qc_ot, test_datas, test_id, layouts, backend):
    """Dynamic-circuit OTEM (DOTEM):
      1. OT circuit + measure into 'testresult'
      2. For each qubit j: IF testresult[j] == 0  (OT passed)
                              THEN replay OT2 gate on that qubit
         (the conditional algorithm run)
      3. Measure into 'measureresult'
    The IF block uses Qiskit's if_test() construct, which compiles to
    real-time classical feed-forward on the hardware.  The algorithm runs
    inside the branch only when every OT qubit passes — making DOTEM a
    true conditional execution rather than post-selection.
    """
    qc  = qc_ot.copy()
    crt = ClassicalRegister(n_lay, name='testresult')
    qc.add_register(crt)
    qc.measure(range(n_lay), crt)

    # Conditional algorithm: per-qubit if_test on the OT classical bit
    for j, qb_phys in enumerate(layouts):
        tid = test_id[qb_phys] if qb_phys < len(test_id) else -1
        if tid == -1:
            continue
        q_template = test_datas[tid].copy()
        q_template.remove_final_measurements()
        with qc.if_test((crt[j], 0)):
            for inst in q_template.data:
                qc.append(inst.operation, qargs=[qc.qregs[0][j]])

    crm = ClassicalRegister(n_lay, name='measureresult')
    qc.add_register(crm)
    qc.measure(range(n_lay), crm)
    return transpile(qc, backend, initial_layout=layouts, optimization_level=0)


print("\nBuilding circuit variants …")
qc_ori   = build_original(qc_ot2, layouts, backend)
qc_sotem = build_sotem(qc_ot2, qc_ot1, layouts, backend)
qc_dotem = build_dotem(qc_ot2, qc_ot1, test_datas, test_id, layouts, backend)

print(f"  Original : depth={qc_ori.depth():4d}   size={qc_ori.size():5d}")
print(f"  SOTEM    : depth={qc_sotem.depth():4d}   size={qc_sotem.size():5d}")
print(f"  DOTEM    : depth={qc_dotem.depth():4d}   size={qc_dotem.size():5d}")

# ---------------------------------------------------------------------------
# 5. Execution loop — N_ROUNDS × (Original → SOTEM → DOTEM) inside a Session
#
#    Using a Session ensures the three jobs within each round are executed
#    on the same backend reservation, minimising between-job drift.
#    Jobs are submitted one at a time (sequential) so that wall-clock times
#    are directly comparable and do not overlap.
# ---------------------------------------------------------------------------

# Storage for timing data
timing = {
    'original': {'wall': [], 'qpu': []},
    'sotem':    {'wall': [], 'qpu': []},
    'dotem':    {'wall': [], 'qpu': []},
}

def _qpu_seconds(job):
    """Extract IBM-reported QPU usage seconds from job metrics.
    Returns None if the backend does not provide usage data.
    """
    try:
        return job.metrics().get('usage', {}).get('seconds', None)
    except Exception:
        return None

def _submit_and_time(sampler, qc, label, shots):
    """Submit a single circuit, wait for result, record timings."""
    t0  = time.time()
    job = sampler.run([qc], shots=shots)
    print(f"    [{label}] job_id={job.job_id()} submitted — waiting …", flush=True)
    _   = job.result()            # blocks until the job finishes
    t1  = time.time()
    wall  = t1 - t0
    qpu   = _qpu_seconds(job)
    print(f"    [{label}] wall={wall:.1f}s  qpu={qpu if qpu is not None else 'N/A'}s")
    return wall, qpu, job

print(f"\nStarting {N_ROUNDS} rounds inside a Qiskit Runtime Session …\n")

all_job_ids = []

with Session(backend=backend) as session:
    sampler = Sampler(mode=session)

    for rnd in range(1, N_ROUNDS + 1):
        print(f"  Round {rnd}/{N_ROUNDS}")

        w_ori,  q_ori,  job_ori   = _submit_and_time(sampler, qc_ori,   'Original', SHOTS)
        w_sot,  q_sot,  job_sotem = _submit_and_time(sampler, qc_sotem, 'SOTEM',    SHOTS)
        w_dot,  q_dot,  job_dotem = _submit_and_time(sampler, qc_dotem, 'DOTEM',    SHOTS)

        timing['original']['wall'].append(w_ori);  timing['original']['qpu'].append(q_ori)
        timing['sotem']['wall'].append(w_sot);      timing['sotem']['qpu'].append(q_sot)
        timing['dotem']['wall'].append(w_dot);      timing['dotem']['qpu'].append(q_dot)

        all_job_ids.extend([job_ori.job_id(), job_sotem.job_id(), job_dotem.job_id()])

print("\nAll rounds complete.")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------

# Choose QPU seconds if available (preferred), else fall back to wall-clock
def _pick_metric(entry):
    """Return QPU seconds if all values are populated, else wall-clock seconds."""
    if all(v is not None for v in entry['qpu']):
        return 'QPU usage (s)', entry['qpu']
    return 'Wall-clock (s)', entry['wall']

ori_label, ori_vals = _pick_metric(timing['original'])
sot_label, sot_vals = _pick_metric(timing['sotem'])
dot_label, dot_vals = _pick_metric(timing['dotem'])

# Use consistent metric across all three
if ori_label == 'QPU usage (s)':
    metric_label = 'QPU usage (s)'
    vals = {
        'Original': timing['original']['qpu'],
        'SOTEM':    timing['sotem']['qpu'],
        'DOTEM':    timing['dotem']['qpu'],
    }
else:
    metric_label = 'Wall-clock (s)'
    vals = {
        'Original': timing['original']['wall'],
        'SOTEM':    timing['sotem']['wall'],
        'DOTEM':    timing['dotem']['wall'],
    }

ori_mean = np.mean(vals['Original'])

print(f"\n{'='*65}")
print(f"  Hardware Overhead Trend — {backend.name}  ({N_ROUNDS} rounds, {SHOTS} shots)")
print(f"  Metric: {metric_label}")
print(f"{'='*65}")
print(f"  {'Circuit':<12} {'Mean':>10} {'Std':>8} {'Overhead vs Ori':>18} {'Ratio':>7}")
print(f"  {'-'*61}")
for name, v in vals.items():
    mean  = np.mean(v)
    std   = np.std(v)
    delta = mean - ori_mean
    ratio = mean / ori_mean if ori_mean > 0 else float('nan')
    print(f"  {name:<12} {mean:>10.2f} {std:>8.2f} {delta:>+18.2f} {ratio:>7.2f}×")
print(f"{'='*65}")

# Raw timing data for cross-device table construction
raw_data = {
    'backend':     backend.name,
    'n_rounds':    N_ROUNDS,
    'shots':       SHOTS,
    'metric':      metric_label,
    'job_ids':     all_job_ids,
    'original':    vals['Original'],
    'sotem':       vals['SOTEM'],
    'dotem':       vals['DOTEM'],
    'original_wall': timing['original']['wall'],
    'sotem_wall':    timing['sotem']['wall'],
    'dotem_wall':    timing['dotem']['wall'],
    'original_qpu':  timing['original']['qpu'],
    'sotem_qpu':     timing['sotem']['qpu'],
    'dotem_qpu':     timing['dotem']['qpu'],
}
out_json = f"hardware_overhead_trend_{backend.name}.json"
with open(out_json, 'w') as f:
    json.dump(raw_data, f, indent=2)
print(f"\nRaw timing data saved to {out_json}")

# ---------------------------------------------------------------------------
# 7. Bar chart — execution time comparison
# ---------------------------------------------------------------------------

labels_bar = list(vals.keys())
means_bar  = [np.mean(v) for v in vals.values()]
stds_bar   = [np.std(v)  for v in vals.values()]
colors_bar = ['tab:blue', 'tab:orange', 'tab:green']

fig, ax = plt.subplots(figsize=(7, 5))
x    = np.arange(len(labels_bar))
bars = ax.bar(x, means_bar, yerr=[s * 2 for s in stds_bar],
              color=colors_bar, capsize=6, alpha=0.85, width=0.45)

for bar, mean, std in zip(bars, means_bar, stds_bar):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2 * std + 0.01 * means_bar[-1],
            f'{mean:.2f}s', ha='center', va='bottom', fontsize=10)

# Annotate overhead ratios above SOTEM and DOTEM bars
for i, (name, v) in enumerate(vals.items()):
    if name == 'Original':
        continue
    ratio = np.mean(v) / ori_mean
    ax.text(x[i], 0.02 * means_bar[-1],
            f'{ratio:.2f}×', ha='center', va='bottom',
            fontsize=9, color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels_bar, fontsize=12)
ax.set_ylabel(metric_label, fontsize=11)
ax.set_title(
    f'Execution-Time Overhead: SOTEM vs DOTEM\n'
    f'{backend.name}  ({N_ROUNDS} rounds × {SHOTS} shots)',
    fontsize=11,
)
ax.yaxis.grid(True, linestyle='--', alpha=0.45)
ax.set_axisbelow(True)
plt.tight_layout()

out_pdf = f"hardware_overhead_trend_{backend.name}.pdf"
plt.savefig(out_pdf, bbox_inches='tight')
plt.show()
print(f"Figure saved to {out_pdf}")

# ---------------------------------------------------------------------------
# 8. Multi-device trend table helper
#    After running on several devices, merge the JSON files produced above
#    into a single Markdown / LaTeX table comparable to Table I.
# ---------------------------------------------------------------------------

def merge_device_results(json_files):
    """Print a combined overhead table from per-device JSON result files.

    Parameters
    ----------
    json_files : list[str]  — paths to JSON files produced by this script.

    Example
    -------
    >>> merge_device_results([
    ...     'hardware_overhead_trend_device_gen1.json',
    ...     'hardware_overhead_trend_device_gen2.json',
    ...     'hardware_overhead_trend_device_gen3.json',
    ... ])
    """
    print(f"\n{'='*78}")
    print(f"  Table I — Execution-Time Overhead Across Device Generations")
    print(f"{'='*78}")
    header = f"  {'Device':<22} {'Metric':<16} {'w/o EM (s)':>12} {'SOTEM (s)':>12} {'DOTEM (s)':>12} {'DOTEM/Ori':>10}"
    print(header)
    print(f"  {'-'*74}")
    for path in json_files:
        if not os.path.exists(path):
            print(f"  {'<not found: ' + path + '>':^74}")
            continue
        with open(path) as f:
            d = json.load(f)
        metric = d.get('metric', 'Wall-clock (s)')
        key    = 'qpu' if 'QPU' in metric else 'wall'
        ori  = np.mean(d[f'original_{key}']) if d.get(f'original_{key}') and all(v is not None for v in d[f'original_{key}']) else np.mean(d['original'])
        sot  = np.mean(d[f'sotem_{key}'])    if d.get(f'sotem_{key}')    and all(v is not None for v in d[f'sotem_{key}'])    else np.mean(d['sotem'])
        dot  = np.mean(d[f'dotem_{key}'])    if d.get(f'dotem_{key}')    and all(v is not None for v in d[f'dotem_{key}'])    else np.mean(d['dotem'])
        print(f"  {d['backend']:<22} {metric:<16} {ori:>12.2f} {sot:>12.2f} {dot:>12.2f} {dot/ori:>10.2f}×")
    print(f"{'='*78}")

# Run immediately with the file just produced (single device)
merge_device_results([out_json])
