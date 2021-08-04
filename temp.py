def get_single_optimal_method(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
		# get best parameter list for activation gate
		#for 1 qubit gate
		#faultfree is a QuantumGate
		faultfree = self.get_activation_gate(fault)
		#faulty is a  QuantumGate
		faulty = fault.get_faulty_gate(faultfree)
		#faulty = self.get_activation_gate(fault)

		
		# print("Start:",faulty[0][0].params)
		faultfree_matrix = faultfree.gate.to_matrix()
		faulty_matrix = faulty.gate.to_matrix()
		# parameter list for activation gate
		parameter_list = [0, 0, 0]

		# print(faultfree[0])
		parameter_list = self.single_explore(faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		parameter_list = self.single_gradient(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		# self.single_explore(faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		# parameter_list = self.single_annealing_3_dir(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		# parameter_list = self.single_deterministic(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		# print("after annealing: ", parameter_list)
		# score = vector_distance(faultfree_quantum_state, faulty_quantum_state)

		faulty_parameter = self.faulty_activation_gate(fault, parameter_list)
		gate1 = faulty.gate
		gate2 = Qgate.U3Gate(*faulty_parameter)
		# faulty_gate_list.append(Qgate.U3Gate(faulty_parameter[0], faulty_parameter[1], faulty_parameter[2]))
		# faulty_gate_list = faulty_gate_list + self.U_to_gate_set_transpiler_to_gate_list(faulty_parameter)
		# faulty_gate_list.append(faulty.gate)
		
		# faultfree_gate_list.append(Qgate.U3Gate(parameter_list[0], parameter_list[1], parameter_list[2]))
		faultfree_gate_list = faultfree_gate_list + self.U_to_gate_set_transpiler_to_gate_list(parameter_list)
		faultfree_gate_list.append(faultfree.gate)
		# print(faulty_matrix)
		faultfree_quantum_state = matrix_operation([U3(parameter_list), faultfree_matrix], faultfree_quantum_state, max_size=2)
		faulty_quantum_state = matrix_operation([U3(self.faulty_activation_gate(fault, parameter_list)), faulty_matrix], faulty_quantum_state, max_size=2)

		# print(faultfree_quantum_state, faulty_quantum_state)
		faulty_quantum_state_ = to_probability(faulty_quantum_state)
		faultfree_quantum_state_ = to_probability(faultfree_quantum_state)
		# faulty_quantum_state_, faultfree_quantum_state_ = compression(faulty_quantum_state_, faultfree_quantum_state_)
		repetition, boundary = compute_repetition(faulty_quantum_state_, faultfree_quantum_state_, self.alpha, self.beta)
		# print(parameter_list, repetition)
		# print("repetition:", parameter_list, repetition)
		# print(np.array(faultfree_quantum_state*np.conj(faultfree_quantum_state)), np.array(faulty_quantum_state*np.conj(faulty_quantum_state)))
		# print(faultfree_quantum_state, faulty_quantum_state)

		# gate1 gate2 type = Qgate.U3Gate(parameter_list)
		return (gate1 , gate2 , faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition)