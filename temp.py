def overall_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
	
	# notice that the fault might be single/CNOT
	# you can pass the CNOT case first
	# do not return stuff
	# please apply the changes directly in the gate lists
	def score(faulty_parameter , faultyfree_parameter):
		return vector_distance(np.matmul(U3(fauty_parameter) , faulty_quantum_state) , np.matmul(U3(faultfree_parameter) , faultfree_quantum_state))
			
	faulty_matrix = np.array([[1 , 0],
							[0 , 1]])
	faultfree_matrix = np.array([[1 , 0],
							[0 , 1]])
	
	if fault == CNOT_variation_fault:
		pass;
	else:
		for gate in faulty_gate_list:
			faulty_matrix = np.matmul(faulty_matrix , gate.to_matrix())
		for gate in faulty_gate_list:
			faultyfree_matrix = np.matmul(faultfree_matrix , gate.to_matrix())

		faulty_theta = 2 * np.acos(faulty_matrix[0][0])
		faulty_lam = cmath.log(-faulty_matrix[0][1] / cmath.sin(faulty_theta / 2)) / 1j
		faulty_phi = cmath.log(faulty_matrix[1][0] / cmath.sin(faulty_theta / 2)) / 1j

		faultfree_theta = 2 * np.acos(faultfree_matrix[0][0])
		faultfree_lam = cmath.log(-faultfree_matrix[0][1] / cmath.sin(faultfree_theta / 2)) / 1j
		faultfree_phi = cmath.log(faultfree_matrix[1][0] / cmath.sin(faultfree_theta / 2)) / 1j

		results = []
		for theta in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
					faulty_parameter = [faulty_theta + theta , faulty_phi + phi , faulty_lam + lam]
					faultfree_parameter = [faultfree_theta + theta , faultfree_phi + phi , faultfree_lam + lam]
					results.append([faulty_parameter , faultfree_parameter , score(faulty_parameter , faultfree_parameter)])

		best_faulty_parameter = max(results, key = lambda x: x[2])[0]
		best_faultfree_parameter = max(results, key = lambda x: x[2])[1]

		faulty_ckt = QuantumCircuit(1)
		faultfree_ckt = QuantumCircuit(1)

		faulty_ckt.u3(theta = best_faulty_parameter[0] , phi = best_faulty_parameter[1] , lam = best_faulty_parameter[2] , qubit = 0)
		faultfree_ckt.u3(theta = best_faultfree_parameter[0] , phi = best_faultfree_parameter[1] , lam = best_faultfree_parameter[2] , qubit = 0)
