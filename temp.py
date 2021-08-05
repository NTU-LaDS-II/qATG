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



def single_gradient_for_overall_gradient(self , U_and_faulty_matrix , U_and_faultfree_matrix):
	def score:

	for i in range(SEARCH_TIME):

def get_U_and_gate_list(self , result_gate_list):
	U_and_gate_list = []
	for i in range(0 , len(result_gate_list) , 6):
		temp = np.array([[1 , 0],
						[0 , 1]])
		for j in range(5):
			temp = np.matmul(temp , result_gate_list[i + j].to_matrix())
		theta = 2 * np.acos(temp[0][0])
		lam = cmath.log(-temp[0][1] / cmath.sin(theta / 2)) / 1j
		phi = cmath.log(temp[1][0] / cmath.sin(theta / 2)) / 1j

		U_and_gate_list.append(Qgate.U3Gate(theta , phi , lam))
		U_and_gate_list.append(result_gate_list[i + 5])
	return U_and_gate_list


def overall_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
	def score(faulty_matrix , faultyfree_matrix):
		return vector_distance(np.matmul(faulty_matrix , faulty_quantum_state) , np.matmul(faultfree_matrix , faultfree_quantum_state))
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

		faulty_q = QuantumCircuit(1)
		faultfree_q = QuantumCircuit(1)

		faulty_q.unitary(data = faulty_matrix , [0] , label = None)
		faultfree_q.unitary(data = faultfree_matrix , [0] , label = None)

		result_faulty_ckt = transpile(faulty_q, basis_gates = self.gate_set , optimization_level = 3)
		result_faultfree_ckt = transpile(faultfree_q , basis_gates = self.gate_set , optimization_level = 3)
 		
 		# qiskit gate
		result_faulty_gate_list = [gate for gate, _, _ in result_faulty_ckt.data]
		result_faultfree_gate_list = [gate for gate, _, _ in result_faultfree_ckt.data]

		# qiskit gate
		U_and_faulty_gate_list = self.get_U_and_gate_list(result_faulty_gate_list)
		U_and_faultfree_gate_list = self.get_U_and_gate_list(result_faultfree_gate_list)

		single_gradient_for_overall_gradient(U_and_faulty_gate_list , U_and_faultfree_gate_list)




		# grid search
		# results = []
		# for theta in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
		# 	for phi in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
		# 		for lam in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
		# 			# using U = RZ*RX*RZ*RX*RZ
		# 			for i in range(0 , len(result_faulty_gate_set) , 6):
		# 				temp = np.array([[1 , 0],
		# 								[0 , 1]])
		# 				for j in range(5):
		# 					temp = np.matmul(temp , result_faulty_gate_set[i + j].to_matrix())
		# 				faulty_gate_set_transpile_to_U.append(temp)
		# 				faulty_gate_set_transpile_to_U.append(result_faulty_gate_set[i + 5].to_matrix())

		# 			for i in range(0 , len(result_faultfree_gate_set) , 6):
		# 				temp = np.array([[1 , 0],
		# 								[0 , 1]])
		# 				for j in range(5):
		# 					temp = np.matmul(temp , result_faultfree_gate_set[i + j].to_matrix())
		# 				faultfree_gate_set_transpile_to_U.append(temp)
		# 				faultfree_gate_set_transpile_to_U.append(result_faultfree_gate_set[i + 5].to_matrix())










					faulty_matrix = np.array([[1 , 0],
											[0 , 1]])
					faultfree_matrix = np.array([[1 , 0],
												[0 , 1]])
					for matrix in faulty_gate_set_transpile_to_U:
						faulty_matrix = np.matmul(faulty_matrix, matrix)
					for matrix in faultfree_gate_set_transpile_to_U:
						faultfree_matrix = np.matmul(faultfree_matrix, matrix)
					results.append(faulty_matrix, faultfree_matrix , score(faulty_matrix , faultfree_matrix))
