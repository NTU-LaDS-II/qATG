from ATPG import *
import numpy as np
import qiskit.circuit.library as Qgate
from warnings import filterwarnings
filterwarnings('ignore')

gate_set = [Qgate.U3Gate]
# gate_set = [Qgate.U3Gate, Qgate.U2Gate, Qgate.U1Gate]
# gate_set = [Qgate.RZGate, Qgate.RXGate]

GRID_SLICE = 11
SEARCH_TIME = 800
SAMPLE_TIME = 10000
MAX_ELEMENT = 50
MIN_REQUIRED_EFFECT_SIZE = 3
generator = ATPG(circuit_size = 5, gate_set = gate_set, alpha = 0.99, beta = 0.999, grid_slice = GRID_SLICE, search_time = SEARCH_TIME, sample_time = SAMPLE_TIME, max_element = MAX_ELEMENT, min_required_effect_size = MIN_REQUIRED_EFFECT_SIZE)

coupling_map = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
value = 0.05*np.pi
single_fault_list, two_fault_list = generator.get_fault_list(coupling_map = coupling_map, two_qubit_faults = [[value, value, value, value, value, value], [value, value, -value, value, value, -value], [value, -value, value, value, -value, value] , [value, -value, -value, value, -value, -value],
		[-value, value, value, -value, value, value], [-value, value, -value, -value, value, -value], [-value, -value, value, -value, -value, value] , [-value, -value, -value, -value, -value, -value]])

configuration_list = generator.get_test_configuration(single_fault_list, two_fault_list)
