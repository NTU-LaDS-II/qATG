	# func below for overall gradient
	def get_element_len(self):
		return len(self.effective_u_ckt.data)

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


	def get_U_and_gate_list(self , result_gate_list , element_len):

		if element_len == 2:
			return result_gate_list

		U_and_gate_list = []
		# using U = RZ*RX*RZ*RX*RZ
		for i in range(0 , len(result_gate_list) , element_len):
			temp = np.array([[1 , 0],
							[0 , 1]])
			for j in range(element_len - 1):
				temp = np.matmul(temp , result_gate_list[i + j].to_matrix())

			theta = 2 * cmath.acos(temp[0][0])
			lam = cmath.log(-temp[0][1] / cmath.sin(theta / 2)) / 1j
			phi = cmath.log(temp[1][0] / cmath.sin(theta / 2)) / 1j

			U_and_gate_list.append(Qgate.U3Gate(np.real(theta) , np.real(phi) , np.real(lam)))
			U_and_gate_list.append(result_gate_list[i + element_len - 1])
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

		faulty_q.unitary(faulty_matrix , [0] , label = None)
		faultfree_q.unitary(faultfree_matrix , [0] , label = None)

		result_faulty_ckt = transpile(faulty_q, basis_gates = self.basis_gates , optimization_level = 3)
		result_faultfree_ckt = transpile(faultfree_q , basis_gates = self.basis_gates , optimization_level = 3)

		return [gate for gate, _, _ in result_faulty_ckt.data] , [gate for gate, _, _ in result_faultfree_ckt.data]

	def transpile_U_and_gate_list_to_template(self , U_and_faulty_gate_list , U_and_faultfree_gate_list):
		best_faulty_gate_list , best_faultfree_gate_list = self.get_result_gate_list(U_and_faulty_gate_list , U_and_faultfree_gate_list)
		return best_faulty_gate_list , best_faultfree_gate_list

	def overall_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
		
		if fault == CNOT_variation_fault:
			pass;
		else:

			element_len = self.get_element_len()

			# qiskit gate
			result_faulty_gate_list , result_faultfree_gate_list = self.get_result_gate_list(faulty_gate_list , faultfree_gate_list)

			# qiskit gate
			U_and_faulty_gate_list = self.get_U_and_gate_list(result_faulty_gate_list , element_len)
			U_and_faultfree_gate_list = self.get_U_and_gate_list(result_faultfree_gate_list , element_len)

			single_gradient_for_overall_gradient(U_and_faulty_gate_list , U_and_faultfree_gate_list)

			# transpile U_and_gate_list to template
			faulty_gate_list , faultfree_gate_list = self.transpile_U_and_gate_list_to_template(U_and_faulty_gate_list , U_and_faultfree_gate_list)
	def transpile_to_U_and_faulty_gate(ckt , element_len):
		if 





	def overall_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
		
		if fault == CNOT_variation_fault:
			pass;
		else:	
			element_len = len(self.effective_u_ckt.data)
			faulty_element_list = list(np.array_split(faulty_gate_list, len(faulty_gate_list) / element_len))
			faultfree_element_list = list(np.array_split(faultfree_gate_list, len(faultfree_gate_list) / element_len))

			U_and_faulty_pair_gate_list = get_U_and_gate_pair_list(faulty_element_list , element_len)
			U_and_faultfree_pair_gate_list = get_U_and_gate_pair_list(faultfree_element_list , element_len)

			# 
	def get_U_and_gate_pair_list(element_list , element_len):
		if element_len == 2:
			return element_list

		result = []
		for element in element_list:
			q = QuantumCircuit(1)
			for i in range(element_len - 1):
				q.append(element[i] , [0])
				ckt_to_u = transpile(q , basis_gates = ['u3'] , optimization_level = 3)

			result.append([ckt_to_u[0][0] , element[element_len - 1]])


			


			