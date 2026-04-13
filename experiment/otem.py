from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer.backends.statevector_simulator import StatevectorSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

class OTEM:
    def __init__(self, backend, qATG_test_templets, is_real_qc = True):
        self.backend = backend
        self.is_real_qc = is_real_qc
        self.qATG_test_templets = qATG_test_templets
        
    def check_test_templets(self):
        test_data_rm = [data.copy() for data in self.qATG_test_templets]
        backend_sim = StatevectorSimulator()
        for i in range(len(test_data_rm)):
            test_data_rm[i].remove_final_measurements()
        sim_results = backend_sim.run(test_data_rm)
        for i in range(len(test_data_rm)):
            print(sim_results.result().get_statevector(i))
        
    def build_preprocess_test(self):
        qATG_test_templets = self.qATG_test_templets
        backend = self.backend
        num_qubits = backend.num_qubits
        qcs = []
        for data in qATG_test_templets:
            qc = data.copy()
            qc.remove_final_measurements()
            for i in range(num_qubits - 1):
                qc = qc.tensor(data.copy())
                qc.remove_final_measurements()
            qc = transpile(qc, backend, scheduling_method='asap', optimization_level = 0) if self.is_real_qc else transpile(qc, basis_gates = backend.basis_gates, optimization_level = 0)
            qc.measure_all()
            qcs.append(qc)
        qc = QuantumCircuit(num_qubits)
        qc.measure_all()
        qcs.append(qc)
        return qcs

    def get_acf_values_and_fail_rates(self, memory, lag = 1, binsize = 1):
        nqb = self.backend.num_qubits
        acf_values = []
        cnts = []
        for qb in range(nqb):
            cnt = 0
            qb_res = []
            for shot in range(len(memory)):
                res = memory[shot]
                if (shot % binsize == 0):
                    qb_res.append(0)
                qb_res[-1] += int(res[qb])
                if (res[qb] == '1'):
                    cnt += 1
            x = list(range(len(qb_res)))
            y = list(qb_res)
            # Autocorrelation Function (ACF)
            acf_values.append(acf(y, nlags=lag)[lag])
            cnts.append(cnt)
        return acf_values, cnts

    def qubit_and_online_test_selection_from_result(self, job_result, binsize = 10):
        nqb = self.backend.num_qubits
        qb_f_id = [-1] * nqb
        qb_f_ac = [-1] * nqb
        max_ac_qb = -1
        max_ac = -1
        def score(acf, failing_rate):
            return acf * failing_rate

        mem = job_result[-1].data.meas.get_bitstrings()
        acf_values, cnts = self.get_acf_values_and_fail_rates(mem, lag=1, binsize=binsize)
        for qb in range(nqb):
            qb_f_ac[qb] = score(acf_values[qb], cnts[qb] / len(mem))
        for exp_id in range(len(self.qATG_test_templets)):
            mem = job_result[exp_id].data.meas.get_bitstrings()
            acf_values, cnts = self.get_acf_values_and_fail_rates(mem, lag=1, binsize=binsize)
            for qb in range(nqb):
                significance_level = 2.576 / np.sqrt(len(mem) // binsize) # 99% confidence interval
                f_rate = cnts[qb] / len(mem)
                if (acf_values[qb] > significance_level and f_rate > 0.1):
                    sc = score(acf_values[qb], f_rate) 
                    if (qb_f_ac[qb] < sc): 
                        qb_f_id[qb] = exp_id
                        qb_f_ac[qb] = sc
                        if (max_ac < qb_f_ac[qb]):
                            max_ac = qb_f_ac[qb]
                            max_ac_qb = qb
        qb_f_id.reverse()
        qb_f_ac.reverse()
        max_ac_qb = nqb - max_ac_qb - 1
        assert max_ac_qb < nqb, "Sorry, no qubit is selected!!!"
        return max_ac_qb, qb_f_id[max_ac_qb], qb_f_ac, qb_f_id
    
    def qubit_and_online_test_selection(self, shots = 10000):
        preprocess_test_qcs = self.build_preprocess_test()
        sampler = Sampler(self.backend)
        job_test = sampler.run(preprocess_test_qcs, shots = shots)
        print('Preprocess test job id:', job_test.job_id())
        job_result = job_test.result()
        return self.qubit_and_online_test_selection_from_result(job_result)
    