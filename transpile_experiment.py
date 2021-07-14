import qiskit.circuit.library as Qgate
from qiskit import QuantumCircuit
from qiskit import transpile
from numpy import pi

def get_qiskit_gate_type(gate_type):
	if gate_type == "RGate":
		return Qgate.RGate
	elif gate_type == "RXGate":
		return Qgate.RXGate
	elif gate_type == "RXXGate":
		return Qgate.RXXGate
	elif gate_type == "RYGate":
		return Qgate.RYGate
	elif gate_type == "RZGate":
		return Qgate.RZGate
	elif gate_type == "RZZGate":
		return Qgate.RZZGate
	elif gate_type == "XGate":
		return Qgate.XGate
	elif gate_type == "YGate":
		return Qgate.YGate
	elif gate_type == "ZGate":
		return Qgate.ZGate
	elif gate_type == "U1Gate":
		return Qgate.U1Gate
	elif gate_type == "U2Gate":
		return Qgate.U2Gate
	elif gate_type == "U3Gate":
		return Qgate.U3Gate
	elif gate_type == "SXGate":
		return Qgate.SXGate


def get_gate_set():
	U_params = [0.25*pi, 0.25*pi, 0.25*pi]
	print("num of gate type")
	num_of_gate_type = int(input())
	gate_set = []
	for i in range(num_of_gate_type):
		gate_type = input()
		gate_set.append(get_qiskit_gate_type(gate_type))
	# gate_set = [Qgate.RZGate, Qgate.SXGate]

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
	return result_gates , gate_set