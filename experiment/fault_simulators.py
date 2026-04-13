import numpy as np
import random
import sys, os

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, UGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Fault model helpers (used by Transient_Qubit_Faulty_Backend)
# ---------------------------------------------------------------------------

def _get_qatg_fault():
    """Lazy import of QATGFault so fault_simulators can be imported without
    the qatg package on sys.path as long as the new classes are not used."""
    try:
        from qatg import QATGFault
        return QATGFault
    except ImportError:
        raise ImportError(
            "qatg package not found. Add the repo root to sys.path before "
            "importing fault_simulators when using Transient_Qubit_Faulty_Backend."
        )

def _make_sx_fault(fault_matrix, qubit_id):
    """Create a QATGFault for a faulty SX gate on `qubit_id`."""
    import qiskit.circuit.library as qGate
    from qiskit.synthesis import OneQubitEulerDecomposer
    QATGFault = _get_qatg_fault()

    class _SXFault(QATGFault):
        def __init__(self, fgate, qb):
            super().__init__(qGate.SXGate, qb,
                             f"gateType: SX, qubits: {qb}")
            theta, phi, lam = OneQubitEulerDecomposer().angles(fgate)
            self._fgate = UGate(theta, phi, lam)
        def createOriginalGate(self, gate): return qGate.SXGate()
        def createFaultyGate(self, gate):   return self._fgate

    return _SXFault(fault_matrix, qubit_id)


def _make_ecr_fault(fault_matrix, qubit_ids):
    """Create a QATGFault for a faulty ECR gate on `qubit_ids`."""
    import qiskit.circuit.library as qGate
    QATGFault = _get_qatg_fault()

    class _ECRFault(QATGFault):
        def __init__(self, fgate, qbs):
            super().__init__(qGate.ECRGate, qbs,
                             f"gateType: ECR, qubits: {qbs}")
            self._fgate = UnitaryGate(fgate)
        def createOriginalGate(self, gate): return qGate.ECRGate()
        def createFaultyGate(self, gate):   return self._fgate

    return _ECRFault(fault_matrix, qubit_ids)


# ---------------------------------------------------------------------------
# Permanent_Faulty_Backend  (original — single fault model, backward compat.)
# ---------------------------------------------------------------------------

class Permanent_Faulty_Backend:
    def __init__(self, fault_model, good_backend,
                 basis_gates=['sx', 'rz', 'cx'], num_qubits=1):
        self.faultObject = fault_model
        self.sim_noise = good_backend
        self.basis_gates = basis_gates
        self.num_qubits = num_qubits

    def generate_faulty_qc(self, qc):
        # optimization_level=0 prevents Qiskit from cancelling G·G† pairs
        # inserted by ZNE unitary folding.
        qc_t = transpile(qc, basis_gates=self.basis_gates, optimization_level=0)
        qc_f = qc_t.copy_empty_like()
        qbIndexes = self.faultObject.getQubits()
        for ci in qc_t:
            gates = ci.operation
            qubits = [qc_t.find_bit(qb).index for qb in ci.qubits]
            if self.faultObject.isSameGateType(gates) and qubits == qbIndexes:
                qc_f.append(self.faultObject.createFaultyGate(gates), qbIndexes)
            else:
                qc_f.append(ci)
        return qc_f

    def run(self, qc, shots=1024, memory=True):
        if not isinstance(qc, QuantumCircuit):
            raise ValueError("qc should be a QuantumCircuit")
        qc_f = self.generate_faulty_qc(qc)
        return self.sim_noise.run(qc_f, shots=shots, memory=memory).result()


# ---------------------------------------------------------------------------
# Permanent_Faulty_Backend_Multi  (new — list of fault models + coupling_map)
# Matches the Permanent_Faulty_Backend defined in OTEM_simulation-new.ipynb.
# ---------------------------------------------------------------------------

class Permanent_Faulty_Backend_Multi:
    """Permanent faulty backend supporting multiple simultaneous fault models
    (e.g. SX faults on several qubits and ECR faults on several edges).

    Parameters
    ----------
    fault_models  : list of QATGFault objects
    good_backend  : AerSimulator (with or without depolarizing noise)
    basis_gates   : list of str — e.g. ['sx', 'rz', 'ecr']
    num_qubits    : int
    coupling_map  : list of [i, j] pairs
    """
    def __init__(self, fault_models, good_backend,
                 basis_gates=['sx', 'rz', 'ecr'],
                 num_qubits=1, coupling_map=[[0, 1]]):
        self.faultObjects = fault_models
        self.sim_noise    = good_backend
        self.basis_gates  = basis_gates
        self.num_qubits   = num_qubits
        self.coupling_map = coupling_map

    def generate_faulty_qc(self, qc):
        # optimization_level=0 is critical: higher levels cancel G·G† pairs
        # that were deliberately inserted by ZNE unitary folding, undoing the
        # noise amplification entirely.
        qc_t = transpile(qc, basis_gates=self.basis_gates,
                         coupling_map=self.coupling_map, optimization_level=0)
        for faultObject in self.faultObjects:
            qc_f      = qc_t.copy_empty_like()
            qbIndexes = faultObject.getQubits()
            for ci in qc_t:
                gates  = ci.operation
                qubits = [qc_t.find_bit(qb).index for qb in ci.qubits]
                if faultObject.isSameGateType(gates) and qubits == qbIndexes:
                    qc_f.append(faultObject.createFaultyGate(gates), qbIndexes)
                else:
                    qc_f.append(ci)
            qc_t = qc_f
        return qc_t

    def run(self, qc, shots=1024, memory=True):
        if not isinstance(qc, QuantumCircuit):
            raise ValueError("qc should be a QuantumCircuit")
        qc_f = self.generate_faulty_qc(qc)
        return self.sim_noise.run(qc_f, shots=shots, memory=memory)


# ---------------------------------------------------------------------------
# Transient_Faulty_Backend  (original — single device state, backward compat.)
# ---------------------------------------------------------------------------

class Transient_Faulty_Backend:
    def __init__(self, faulty_backend, good_backend=AerSimulator(),
                 swap_rate=0.01, ft_to_ff_rate_magnification=1.):
        self.faulty_backend             = faulty_backend
        self.basis_gates                = faulty_backend.basis_gates
        self.num_qubits                 = faulty_backend.num_qubits
        self.good_backend               = good_backend
        self.swap_rate                  = swap_rate
        self.ft_to_ff_rate_magnification = ft_to_ff_rate_magnification
        self.faulty = False

    def random(self):
        if self.faulty:
            return random.random() / self.ft_to_ff_rate_magnification
        return random.random()

    def run_single_qc(self, qc, shots=1024):
        results        = []
        results_faulty = self.faulty_backend.run(qc, shots=shots, memory=True).get_memory()
        results_good   = self.good_backend.run(qc, shots=shots, memory=True).result().get_memory()
        for i in range(shots):
            if self.random() < self.swap_rate:
                self.faulty = not self.faulty
            results.append(results_faulty[i] if self.faulty else results_good[i])
        return results

    def run(self, qcs, shots=1024):
        if type(qcs) is list:
            return [self.run_single_qc(qc, shots) for qc in qcs]
        return [self.run_single_qc(qcs, shots)]

    def run_single_em(self, ot_qc, alg_qc, ot_shots, alg_shots, repeat):
        results        = []
        ot_res_faulty  = self.faulty_backend.run(ot_qc,  shots=ot_shots  * repeat, memory=True).get_memory()
        ot_res_good    = self.good_backend.run(ot_qc,  shots=ot_shots  * repeat, memory=True).result().get_memory()
        alg_res_faulty = self.faulty_backend.run(alg_qc, shots=alg_shots * repeat, memory=True).get_memory()
        alg_res_good   = self.good_backend.run(alg_qc, shots=alg_shots * repeat, memory=True).result().get_memory()
        for i in range(repeat):
            for j in range(ot_shots):
                if self.random() < self.swap_rate:
                    self.faulty = not self.faulty
                results.append(ot_res_faulty[i * ot_shots + j] if self.faulty
                                else ot_res_good[i * ot_shots + j])
            for j in range(alg_shots):
                if self.random() < self.swap_rate:
                    self.faulty = not self.faulty
                results.append(alg_res_faulty[i * alg_shots + j] if self.faulty
                                else alg_res_good[i * alg_shots + j])
        return results

    def run_em(self, ot_qc, alg_qcs, ot_shots, alg_shots, repeat=1000):
        if type(alg_qcs) is list:
            return [self.run_single_em(ot_qc, qc, ot_shots, alg_shots, repeat)
                    for qc in alg_qcs]
        return [self.run_single_em(ot_qc, alg_qcs, ot_shots, alg_shots, repeat)]


# ---------------------------------------------------------------------------
# Transient_Qubit_Faulty_Backend  (new — per-qubit fault states)
# Matches OTEM_simulation-new.ipynb.
#
# Key design points
# -----------------
# • Each qubit has its own is_faulty flag and swap_rate / ft_to_ff_rate_magnification.
# • Setting ft_to_ff_rate_magnification[i] >> 1 makes qubit i QUICKLY recover
#   from a faulty state (almost always good).
# • Setting ft_to_ff_rate_magnification[i] = 1 makes faulty↔good transitions
#   equally likely, so qubit i spends ~50 % of time faulty — the scenario that
#   produces P_raw ≈ 0.3 for mid-circuit qubits.
# • fault states: 3-type ECR model
#     ecr_fault1 — applied when the TARGET qubit (j) is faulty, CONTROL (i) is good
#     ecr_fault2 — applied when the CONTROL qubit (i) is faulty, TARGET (j) is good
#     ecr_fault3 — applied when BOTH qubits are faulty
# ---------------------------------------------------------------------------

class Transient_Qubit_Faulty_Backend:
    def __init__(self, SXfault, ECRfault1, ECRfault2, ECRfault3,
                 good_backend=AerSimulator(),
                 basis_gates=['sx', 'rz', 'ecr'],
                 num_qubits=5, coupling_map=[],
                 swap_rate=None,
                 ft_to_ff_rate_magnification=None):
        """
        Parameters
        ----------
        SXfault   : 2×2 numpy array — unitary of faulty SX gate
        ECRfault1 : 4×4 numpy array — ECR fault when target qubit j is faulty
        ECRfault2 : 4×4 numpy array — ECR fault when control qubit i is faulty
        ECRfault3 : 4×4 numpy array — ECR fault when both qubits are faulty
        good_backend               : AerSimulator
        basis_gates                : list[str]
        num_qubits                 : int
        coupling_map               : list[[int,int]]
        swap_rate                  : list[float], one per qubit (default 0.01 each)
        ft_to_ff_rate_magnification: list[float], one per qubit.
            >1 → qubit recovers quickly from faulty (rarely stays faulty).
            =1 → faulty↔good equally likely (~50 % faulty in steady state).
        """
        if swap_rate is None:
            swap_rate = [0.01] * num_qubits
        if ft_to_ff_rate_magnification is None:
            ft_to_ff_rate_magnification = [1.] * num_qubits

        assert len(swap_rate) == num_qubits
        assert len(ft_to_ff_rate_magnification) == num_qubits

        self.good_backend               = good_backend
        self.basis_gates                = basis_gates
        self.num_qubits                 = num_qubits
        self.coupling_map               = coupling_map
        self.swap_rate                  = swap_rate
        self.ft_to_ff_rate_magnification = ft_to_ff_rate_magnification
        # fault matrices: [SX, ECR_target_only, ECR_control_only, ECR_both]
        self.faulty_matrix = [SXfault, ECRfault1, ECRfault2, ECRfault3]
        self.is_faulty     = [False] * num_qubits

    # ── state helpers ────────────────────────────────────────────────────────

    def _rand(self, i):
        if self.is_faulty[i]:
            return random.random() / self.ft_to_ff_rate_magnification[i]
        return random.random()

    def get_faulty_idx(self):
        """Encode the per-qubit is_faulty list as a single integer bitmask."""
        idx = 0
        for f in self.is_faulty:
            idx = (idx << 1) | (1 if f else 0)
        return idx

    def set_faulty_idx(self, fid):
        for i in range(len(self.is_faulty)):
            self.is_faulty[i] = bool(fid & (1 << i))

    def next_shot(self):
        """Advance the per-qubit Markov chain by one time step."""
        for j in range(self.num_qubits):
            if self._rand(j) < self.swap_rate[j]:
                self.is_faulty[j] = not self.is_faulty[j]

    # ── fault model construction ─────────────────────────────────────────────

    def _get_fault_list(self):
        """Return QATGFault objects matching the current is_faulty state."""
        flist = []
        # SX faults on faulty qubits
        for i in range(self.num_qubits):
            if self.is_faulty[i]:
                flist.append(_make_sx_fault(self.faulty_matrix[0], i))
        # ECR faults on edges where at least one qubit is faulty
        for (i, j) in self.coupling_map:
            if self.is_faulty[i] and self.is_faulty[j]:
                flist.append(_make_ecr_fault(self.faulty_matrix[3], [i, j]))
            elif self.is_faulty[i]:
                flist.append(_make_ecr_fault(self.faulty_matrix[2], [i, j]))
            elif self.is_faulty[j]:
                flist.append(_make_ecr_fault(self.faulty_matrix[1], [i, j]))
        return flist

    def _get_faulty_backend(self, fid):
        self.set_faulty_idx(fid)
        return Permanent_Faulty_Backend_Multi(
            self._get_fault_list(), self.good_backend,
            basis_gates=self.basis_gates, num_qubits=self.num_qubits,
            coupling_map=self.coupling_map,
        )

    # ── circuit execution ─────────────────────────────────────────────────────

    def _run_qc_with_fault_idx_list(self, qc, fault_idx_list):
        """Run `qc` once for each unique fault configuration, then reassemble."""
        unique_fids = list(set(fault_idx_list))
        pool = {}
        for fid in unique_fids:
            count  = fault_idx_list.count(fid)
            fb     = self._get_faulty_backend(fid)
            pool[fid] = list(fb.run(qc, shots=count, memory=True).result().get_memory())
        results = []
        for fid in fault_idx_list:
            results.append(pool[fid].pop())
        return results

    def run_single_qc(self, qc, shots=1024, reset=True):
        if reset:
            self.is_faulty = [False] * self.num_qubits
        fault_idx_list = []
        for _ in range(shots):
            self.next_shot()
            fault_idx_list.append(self.get_faulty_idx())
        return self._run_qc_with_fault_idx_list(qc, fault_idx_list)

    def run(self, qcs, shots=1024, reset=True):
        if not isinstance(qcs, list):
            qcs = [qcs]
        if reset:
            self.is_faulty = [False] * self.num_qubits
        return [self.run_single_qc(qc, shots, reset=False) for qc in qcs]

    def run_single_em(self, ot_qc, alg_qc, ot_shots, alg_shots, repeat):
        """Interleave OT and algorithm shots with shared per-qubit fault state.

        For each of `repeat` trials:
          • `ot_shots`  OT circuit shots  (fault state advances each shot)
          • `alg_shots` alg circuit shots (fault state advances each shot)

        Returns a flat list: [ot_0, alg_0, ot_1, alg_1, …]
        Compatible with get_em_counts_from_mem().
        """
        ot_fault_idx_list  = []
        alg_fault_idx_list = []
        for _ in range(repeat):
            for _ in range(ot_shots):
                self.next_shot()
                ot_fault_idx_list.append(self.get_faulty_idx())
            for _ in range(alg_shots):
                self.next_shot()
                alg_fault_idx_list.append(self.get_faulty_idx())

        ot_results  = self._run_qc_with_fault_idx_list(ot_qc,  ot_fault_idx_list)
        alg_results = self._run_qc_with_fault_idx_list(alg_qc, alg_fault_idx_list)

        results = []
        for i in range(repeat):
            results += ot_results [ot_shots  * i: ot_shots  * (i + 1)]
            results += alg_results[alg_shots * i: alg_shots * (i + 1)]
        return results

    def run_em(self, ot_qc, alg_qcs, ot_shots, alg_shots, repeat=1000, reset=True):
        if not isinstance(alg_qcs, list):
            alg_qcs = [alg_qcs]
        if reset:
            self.is_faulty = [False] * self.num_qubits
        return [self.run_single_em(ot_qc, qc, ot_shots, alg_shots, repeat)
                for qc in alg_qcs]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_counts_from_mem(mem):
    counts = {}
    for r in mem:
        counts[r] = counts.get(r, 0) + 1
    return counts


def get_em_counts_from_mem(mem, shots_ot=1, ot_pass_th=1, shots_alg=1,
                            bidirectional_check=False):
    """Post-select algorithm shots where the OT result is all-zeros.

    Works for both single-qubit OT ('0') and multi-qubit OT ('0000…').
    The pass criterion is: every OT bitstring equals '0' * len(bitstring).
    """
    shots    = shots_ot + shots_alg
    mem_pass = []
    pre_test_pass = False
    for i in range(len(mem) // shots):
        cnt_pass = 0
        for j in range(shots_ot):
            s = mem[i * shots + j]
            if s == '0' * len(s):       # all-zeros → OT passed
                cnt_pass += 1
        test_pass = (cnt_pass >= ot_pass_th)
        if test_pass:
            for j in range(shots_alg):
                mem_pass.append(mem[i * shots + shots_ot + j])
        elif bidirectional_check and pre_test_pass:
            for j in range(shots_alg):
                mem_pass.pop()
        pre_test_pass = test_pass
    return get_counts_from_mem(mem_pass)
