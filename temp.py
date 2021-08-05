

def single_gradient_for_overall_gradient(self , U_and_faulty_gate_list , U_and_faultfree_gate_list , faulty_quantum_state, faultfree_quantum_state):
	def score():
		temp1 = temp2 = np.array([[1 , 0],
								[0 , 1]])
		 
		for gate in U_and_faulty_gate_list:
			temp1 = np.matmul(temp1 , gate.to_matrix())
		for gate in U_and_faultfree_gate_list:
			temp2 = np.matmul(temp2 , gate.to_matrix())
		return vector_distance(np.matmul(temp1 , faulty_quantum_state) , np.matmul(temp2 , faultfree_quantum_state))
	def parameter_up(U_and_faulty_gate_list , U_and_faultfree_gate_list , index , time , step):
		for i in range(0 , len(U_and_faulty_gate_list) , 2):
			U_and_faulty_gate_list[i].params[index] += time * step
		for i in range(0 , len(U_and_faultfree_gate_list) , 2):
			U_and_faultfree_gate_list[i].params[index] += time * step
		return 

	def parameter_down(U_and_faulty_gate_list , U_and_faultfree_gate_list , index , time , step):
		for i in range(0 , len(U_and_faulty_gate_list) , 2):
			U_and_faulty_gate_list[i].params[index] -= time * step
		for i in range(0 , len(U_and_faultfree_gate_list) , 2):
			U_and_faultfree_gate_list[i].params[index] -= time * step
		return 

	for i in range(SEARCH_TIME):
		for j in range(3):
			current_score = score()
			parameter_up(U_and_faulty_gate_list , U_and_faultfree_gate_list , j , 1 , self.step)
			up_score = score()
			parameter_down(U_and_faulty_gate_list , U_and_faultfree_gate_list , j , 2 , self.step)
			down_score = score()
			parameter_up(U_and_faulty_gate_list , U_and_faultfree_gate_list , j , 1 , self.step)

			if(up_score > current_score and up_score >= down_score):
				parameter_up(U_and_faulty_gate_list , U_and_faultfree_gate_list , j , 1 , self.step*(up_score - current_score))
			elif(down_score > current_score and down_score >= up_score):
				parameter_down(U_and_faulty_gate_list , U_and_faultfree_gate_list , j , 1 , self.step*(down_score - current_score))
	return


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

def get_result_gate_list(self , faulty_gate_list , faultfree_gate_list):
	faulty_matrix = faultfree_matrix = np.array([[1 , 0],
												[0 , 1]])
	for gate in faulty_gate_list:
		faulty_matrix = np.matmul(faulty_matrix , gate.to_matrix())
	for gate in faultfree_gate_list:
		faultfree_matrix = np.matmul(faultfree_matrix , gate.to_matrix())

	faulty_q = QuantumCircuit(1)
	faultfree_q = QuantumCircuit(1)

	faulty_q.unitary(data = faulty_matrix , [0] , label = None)
	faultfree_q.unitary(data = faultfree_matrix , [0] , label = None)

	result_faulty_ckt = transpile(faulty_q, basis_gates = self.gate_set , optimization_level = 3)
	result_faultfree_ckt = transpile(faultfree_q , basis_gates = self.gate_set , optimization_level = 3)

	return [gate for gate, _, _ in result_faulty_ckt.data] , [gate for gate, _, _ in result_faultfree_ckt.data]

def transpile_U_and_gate_list_to_template(self , U_and_faulty_gate_list , U_and_faultfree_gate_list):
	best_faulty_gate_list , best_faultfree_gate_list = self.get_result_gate_list(U_and_faulty_gate_list , U_and_faultfree_gate_list)
	return best_faulty_gate_list , best_faultfree_gate_list

def overall_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
	
	if fault == CNOT_variation_fault:
		pass;
	else:
		# qiskit gate
		result_faulty_gate_list , result_faultfree_gate_list = self.get_result_gate_list(faulty_gate_list , faultfree_gate_list)

		# qiskit gate
		U_and_faulty_gate_list = self.get_U_and_gate_list(result_faulty_gate_list)
		U_and_faultfree_gate_list = self.get_U_and_gate_list(result_faultfree_gate_list)

		single_gradient_for_overall_gradient(U_and_faulty_gate_list , U_and_faultfree_gate_list)

		# transpile U_and_gate_list to template
		faulty_gate_list , faultfree_gate_list = self.transpile_U_and_gate_list_to_template(U_and_faulty_gate_list , U_and_faultfree_gate_list)



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

