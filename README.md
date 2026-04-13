# qATG + OTEM

**qATG** (quantum Automatic Test Generator) v0.8.3 — compatible with **Qiskit 1.0 – 2.0**

**OTEM** (Online Test and Error Mitigation) — adaptive online testing framework built on top of qATG, described in the paper:

> *Online Testing Error Mitigation for Quantum Computers*

---

## Overview

### qATG

qATG generates test configurations for user-defined quantum gate faults. Each configuration contains a test template (a sequence of test elements) and the statistics needed to bound overkill and test-escape rates. One test element consists of one original gate and one activation gate; the activation gate is solved to maximise the output probability difference between the fault-free and faulty circuits.

Key assumptions:
- The basis gate set must be **universal** (any U gate can be expressed in terms of the basis gates).
- Different faults are assumed on different qubits; one qubit carries one consistent fault.

### OTEM

OTEM uses qATG-generated test circuits as online tests interleaved with an algorithm circuit. At runtime it:
1. Runs a preprocessing stage to select the most suspicious qubit and the best test circuit via autocorrelation analysis and fail-rate scoring.
2. Interleaves online test (OT) shots with algorithm shots.
3. Discards algorithm results from windows where the OT detects a fault.

The `experiment/` directory contains all simulation and hardware-experiment notebooks that reproduce the paper's figures.

---

## Installation

### Install via pip

```bash
pip install qatg==0.8
```

### Local install from source

```bash
git clone https://github.com/NTU-LaDS-II/qATG.git
cd qATG
pip install -e .
```

### OTEM experiment dependencies

```bash
pip install statsmodels qiskit-ibm-runtime
```

Or install all dependencies at once:

```bash
pip install -e ".[otem]"
```

### Run in Docker

```bash
docker pull ntuladsii/qatg
docker run -it --rm ntuladsii/qatg
```

---

## Quick Start

### Generate test configurations with qATG

```python
import qiskit.circuit.library as qGate
from qatg import QATG, QATGFault

class myBiasFault(QATGFault):
    def __init__(self):
        super().__init__(qGate.SXGate, 0, "SX bias fault on qubit 0")
    def createOriginalGate(self):
        return qGate.SXGate()
    def createFaultyGate(self, faultfreeGate):
        import numpy as np
        m = faultfreeGate.to_matrix()
        m[0, 1] += 0.1  # perturb matrix
        from qiskit.circuit.library import UnitaryGate
        return UnitaryGate(m)

generator = QATG(
    circuitSize=1,
    basisSingleQubitGateSet=[qGate.SXGate, qGate.XGate, qGate.RZGate],
    circuitInitializedStates={1: [1, 0]},
)
configurations = generator.createTestConfigurationCompressed([[myBiasFault()]])
print(configurations[0])
```

### Run OTEM on a real backend

```python
from otem import OTEM

test_circuits = [cfg.faultfreeQCKT for cfg in configurations]
otem = OTEM(backend, test_circuits, is_real_qc=True)
best_qubit, best_test_id, scores, test_ids = otem.qubit_and_online_test_selection(shots=10000)
```

---

## Experiments

All experiment files are in `experiment/`. Run them from that directory so relative paths (`data/`, `test_circuits/`) resolve correctly.

### Simulation notebooks

These notebooks reproduce the paper's simulation results using Qiskit Aer and require no IBM Quantum access.

| Notebook | Section / Fig. | Description |
|---|---|---|
| `Test Generation.ipynb` | Sec. IV-B | Generate the 29 QASM test circuits in `test_circuits/` from `data/faults.npy` using qATG. Run this first before any other notebook. |
| `sim_multiqubit_benchmark.ipynb` | Sec. V-A / Fig. 13 | Correct rate for QFT, GraphState, QuantumVolume on 3–7 qubits. Three conditions: Fault-Free / Faulty (transient, 1% toggle) / Single-shot OTEM. |
| `sim_depolarizing_noise_sweep.ipynb` | Sec. V-B-1 / Fig. 14 | Test pass rate vs depolarizing noise rate with fixed fault infidelity and 1% toggle probability. |
| `sim_fault_infidelity_sweep.ipynb` | Sec. V-B-2 / Fig. 15 | Test pass rate vs fault-induced infidelity, sweeping all 20 SX fault models at fixed noise and toggle rate. |
| `sim_multishot_fault_occurrence.ipynb` | Sec. V-B-3 / Fig. 16 | Multi-shot OTEM pass rate and overhead vs fault occurrence probability for m/n configurations, with and without bidirectional check. |
| `sim_multishot_toggle_prob.ipynb` | Sec. V-B-4 / Fig. 17 | Same m/n configurations as Fig. 16, but x-axis is toggle probability (log scale). |

### Hardware experiment notebooks

These notebooks run on IBM Quantum hardware via Qiskit Runtime. Replace the token, instance, and job ID placeholders before running.

| Notebook | Section / Fig. | Description |
|---|---|---|
| `exp_rb_error_mitigation.ipynb` | Sec. VI-A / Figs. 18–19 | Single-qubit and two-qubit EPC comparison with/without single-shot OTEM using Standard RB. Qubit selection is determined by OTEM preprocessing. |
| `exp_5qubit_benchmark.ipynb` | Sec. VI-B / Fig. 20 | QFT, GraphState, QuantumVolume on up to two sets of 5 physical qubits with/without single-shot OTEM. Qubit layouts are selected automatically via OTEM preprocessing (BFS over the device coupling map). |
| `exp_singleshot_vs_dynamic_otem.ipynb` | Sec. VI-C / Fig. 21 & Table I | Failing rate comparison for Original / SOTEM / DOTEM. Interleaves three circuit variants per repetition inside a Qiskit Runtime Session. |

### Hardware experiment scripts

Standalone Python scripts for long-running hardware experiments. Each accepts `--token`, `--instance`, and `--backend` as required arguments. Previously submitted job IDs can be passed via `--preprocess-job-id` or `--main-job-id` to avoid resubmitting.

| Script | Section / Fig. | Description |
|---|---|---|
| `Hardware_Overhead_Trend_Test.py` | Sec. VI-C / Table I | SOTEM vs DOTEM execution-time overhead. Records both wall-clock and IBM-reported QPU seconds per round. Outputs a JSON and a bar chart PDF per backend. |
| `Scalability_20Qubit_GHZ_BV.py` | Sec. VII | Single-shot OTEM on 20-qubit GHZ and Bernstein-Vazirani circuits. Automatically selects faulty qubits via preprocessing; if two disjoint fault regions are found, composes them into a single 40-qubit circuit for simultaneous execution. |
| `Comparison_ZNE_vs_OTEM.py` | — | Simulation comparing Raw / ZNE / ZNE+OTEM on a 5-qubit QFT circuit using Richardson extrapolation at noise scale factors [1, 3, 5]. |

### Key experiment files

```
experiment/
├── otem.py                              # OTEM class: preprocessing, qubit selection, online test scheduling
├── fault_simulators.py                  # Transient/Permanent faulty backend simulators (Aer)
├── result_analyze.py                    # Success rate (srate) and RB curve-fitting utilities
├── Hardware_Overhead_Trend_Test.py      # Hardware: SOTEM vs DOTEM timing overhead (Sec. VI-C)
├── Scalability_20Qubit_GHZ_BV.py        # Hardware: 20/40-qubit GHZ & BV scalability (Sec. VII)
├── Comparison_ZNE_vs_OTEM.py            # Simulation: ZNE vs OTEM comparison
├── Test Generation.ipynb                # Generate test_circuits/ — run this first
├── sim_multiqubit_benchmark.ipynb       # Sec. V-A  / Fig. 13
├── sim_depolarizing_noise_sweep.ipynb   # Sec. V-B-1 / Fig. 14
├── sim_fault_infidelity_sweep.ipynb     # Sec. V-B-2 / Fig. 15
├── sim_multishot_fault_occurrence.ipynb # Sec. V-B-3 / Fig. 16
├── sim_multishot_toggle_prob.ipynb      # Sec. V-B-4 / Fig. 17
├── exp_rb_error_mitigation.ipynb        # Sec. VI-A  / Figs. 18–19
├── exp_5qubit_benchmark.ipynb           # Sec. VI-B  / Fig. 20
├── exp_singleshot_vs_dynamic_otem.ipynb # Sec. VI-C  / Fig. 21 & Table I
├── data/
│   ├── faults.npy                       # SX gate fault matrices (qubit-frequency fault model)
│   └── ecr_fault_list.npy               # ECR gate fault matrices
└── test_circuits/                       # 29 qATG-generated QASM online test circuits
```

---

## qATG API Reference

### QATG

```python
generator = QATG(
    circuitSize,               # number of qubits in the circuit
    basisSingleQubitGateSet,   # list of qiskit.circuit.library gate classes
    circuitInitializedStates,  # dict: {num_qubits: initial_state_vector}
    # optional:
    minRequiredStateFidelity=0.4,   # stop growing template when max fidelity < this
    maxTestTemplateSize=50,
    gridSlice=11,
    gradientDescentMaxIteration=1000,
    gradientDescentStep=0.2,
    gradientMeasureStep=0.0001,
    gradientDeltaThreshold=1e-8,
    verbose=False,
)
```

**Methods:**

- `createTestConfiguration(faultList)` — one configuration per fault.
- `createTestConfigurationCompressed(faultGroupList)` — one configuration per fault *group*; the template is grown until all faults in the group are detectable.

### QATGConfiguration

Returned by `createTestConfiguration` / `createTestConfigurationCompressed`.

- `configuration.faultfreeQCKT` — the fault-free test `QuantumCircuit`.
- `print(configuration)` — prints fault, circuit length, repetitions, effect size, simulated overkill/escape.

### QATGFault

Subclass this to describe a fault:

```python
class myFault(QATGFault):
    def __init__(self):
        super().__init__(gateType, qubits, description=None)
    def createOriginalGate(self):
        ...  # return fault-free gate instance
    def createFaultyGate(self, faultfreeGate):
        ...  # return faulty gate instance
```

- `gateType`: a `qiskit.circuit.library` gate class (e.g. `qGate.SXGate`).
- `qubits`: qubit index (int) or list of ints. Must match the gate's arity.

---

## Examples

See `examples/` for runnable scripts:

- **example1.py** — faulty UGate with a bias fault; basis set `{UGate}`.
- **example2.py** — faulty RXGate and RZGate with bias faults; basis set `{RXGate, RZGate}`.
- **example3.py** — faulty CXGate; basis set `{UGate}`.
- **Playground.ipynb** — interactive Jupyter playground.

---

## Contact

NTU-LaDS-II — <lads427@gmail.com>
