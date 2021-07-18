import qiskit.circuit.library as Qgate
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit import Parameter
from numpy import pi


# U_params = [0.25*pi, 0.25*pi, 0.25*pi]
# gate_set = [Qgate.RZGate, Qgate.SXGate]

# basis_gates = [gate.__name__[:-4].lower() for gate in gate_set]
# q = QuantumCircuit(1)
# q.u(*U_params, 0)
# result_ckt = transpile(q, basis_gates = basis_gates, optimization_level = 3)
# result_gates = [gate for gate, _, _ in result_ckt.data]


# DO NOT REMOVE THE COMMENT
# another more safe method
# result_gates = []
# for gate, _, _ in result_ckt.data:
	# new_params = [param.__float__() for param in gate.params]
	# new_gate = gate_set[basis_gates.index(gate.__class__.__name__[:-4].lower())]
	# result_gates.append(new_gate(*new_params))
# return result_gates


# print(result_gates)

# -------

# ready_to_solve_list = []
# ready_to_solve_list.append(parameter_list)

# def annealing(ready_to_solve_list):
# 	best_score = INT_MIN
# 	best_solution = None
# 	for solution in ready_to_solve_list:
# 		score(solution)
# 		up_score(solution)
# 		down_score(solution)

# 		# up
# 		if up_score > score:
# 			ready_to_solve_list.append(up_solution)
# 		else:
# 			delta_t = up_score - score # < 0
# 			if random([0~1]) < exp(delta_t / T):
# 				ready_to_solve_list.append(up_solution)
# 		# down similar

# 		T /= r

# 		if T < threshold:
# 			break

# 	# how to find best solution

theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')

q = QuantumCircuit(1)
q.u(theta, phi, lam, 0)
ckt = transpile(q, basis_gates = ['sx', 'rz'], optimization_level = 3)
# new_ckt = ckt.bind_parameters({theta: pi/2, phi: pi/2, lam: pi/2})


print(new_ckt.data)

