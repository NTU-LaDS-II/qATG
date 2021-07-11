from ATPG import *
import numpy as np
# import qiskit
import qiskit.circuit.library as Qgate
# import sys
from warnings import filterwarnings
filterwarnings('ignore')
# initial_state = np.dot(np.kron(qiskit.extensions.standard.u2.U2Gate(0, np.pi).to_matrix(), qiskit.extensions.standard.u2.U2Gate(0, np.pi).to_matrix()),
#      qiskit.extensions.standard.x.CnotGate().to_matrix())
# initial_state = np.dot(initial_state, np.array([1, 0, 0, 0]))
# initial_state = np.array([1,0])
# initial_state = np.dot(qiskit.extensions.standard.u2.U2Gate(0, np.pi).to_matrix(), initial_state)
# print(qiskit.extensions.standard.x.CnotGate().to_matrix())
# print(initial_state)

generator = ATPG(circuit_size = 5, gate_set = [Qgate.U3Gate, Qgate.U2Gate, Qgate.U1Gate])
generator.alpha = 0.99
generator.beta = 0.999

coupling_map = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
fault_list = generator.get_fault_list(coupling_map)

configuration_list = generator.get_test_configuration(fault_list[0], fault_list[1])