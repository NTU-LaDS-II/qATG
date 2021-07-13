import qiskit.circuit.library as Qgate
from qiskit import QuantumCircuit
from qiskit import transpile
from numpy import pi

U_params = [0.25*pi, 0.25*pi, 0.25*pi]
gate_set = [Qgate.RZGate, Qgate.SXGate]

basis_gates = [gate.__name__[:-4].lower() for gate in gate_set]
q = QuantumCircuit(1)
q.u(*U_params, 0)
result_ckt = transpile(q, basis_gates = basis_gates, optimization_level = 3)
result_gates = [gate for gate, _, _ in result_ckt.data]
# DO NOT REMOVE THE COMMENT
# another more safe method
# result_gates = []
# for gate, _, _ in result_ckt.data:
	# new_params = [param.__float__() for param in gate.params]
	# new_gate = gate_set[basis_gates.index(gate.__class__.__name__[:-4].lower())]
	# result_gates.append(new_gate(*new_params))
# return result_gates
print(result_gates)