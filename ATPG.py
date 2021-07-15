import numpy as np
# import qiskit
# import math
from Fault import *
from Gate import *
from scipy.stats import chi2, ncx2
import qiskit.circuit.library as Qgate
from qiskit.circuit.quantumregister import Qubit
# from qiskit.quantum_info import process_fidelity
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError
from qiskit import Aer
from qiskit import execute, transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
# import statsmodels.stats.power as smp
from qiskit import transpile
from numpy import pi
import random
from QuantumGate import *
from util import *

random.seed(427427)
# INT_MIN = 1E-100
# INT_MAX = 1E15
# SEARCH_TIME = 700
# sample_time = 10000
# threshold = 0.01
# r = 0.1

# fuck that noise model

class Configuration():
	def __init__(self, qc_faulty_list, qc_faultfree, length, fault_list):
		self.qc_faulty_list = qc_faulty_list
		self.qc_faultfree = qc_faultfree
		self.length = length
		self.fault_list = fault_list
		self.repetition_list = []
		self.max_repetition = INT_MAX
		self.sim_faulty_distribution_list = []
		self.sim_faultfree_distribution = []
		self.sim_overkill = INT_MAX
		self.sim_testescape = []
		self.template = []
		self.real_faulty_distribution = []
		self.boundary = 0

		#if 1 is simulation result
		self.real_faulty_distribution_list_1 = []
		self.real_faultfree_distribution_1 = []
		self.real_overkill_1 = INT_MAX
		self.real_testescape_1 = []
		#if 2 is simulation result
		self.real_faulty_distribution_list_2 = []
		self.real_faultfree_distribution_2 = []
		self.real_overkill_2 = INT_MAX
		self.real_testescape_2 = []

	def __str__(self):
		target_fault = ""
		myself = ""
		for i in self.fault_list:
			target_fault += "\n"+str(i)
		myself += "Target fault:"+target_fault+"\n"
		myself += "Length:"+str(self.length)+"     Repetition:"+str(self.max_repetition)+"      Cost:"+str(self.length*self.max_repetition)+"\n"
		myself += "overkill:"+str(self.sim_overkill)+"\n"
		myself += "testescape:"+str(self.sim_testescape)+"\n"
		# myself += str(self.faultfree_distribution)+"\n"
		# myself += str(self.faulty_distribution_list[0])+"\n"

		# self.check_bit()
		return myself

	def cal_testescape_new(self, expectd_faulty_distribution_list, observed_faulty_distribution_list, fault_index=[], alpha=0.99):
		max_repetition = self.max_repetition
		if not fault_index:
			fault_index = []
			for f in self.fault_list:
				fault_index.append(f.index)
		
		testescape_list = []
		for i in range(len(expectd_faulty_distribution_list)):
			testescape = 0
			expected_vector, observed_vector = compression_forfault(expectd_faulty_distribution_list[i], observed_faulty_distribution_list[i], fault_index[i])
				
			expected_data = expected_vector * max_repetition
			vector_length = len(expected_vector)
			chi_value = self.boundary#chi2.ppf(alpha, vector_length-1)

			for _ in range(sample_time):
				observed_sampled_data = random.choices(range(observed_vector.shape[0]), weights=observed_vector, cum_weights=None, k=max_repetition)
				observed_data = np.zeros(observed_vector.shape[0])
				for d in observed_sampled_data:
					observed_data[d]+=1

				delta_square = np.square(expected_data - observed_data)
				chi_statistic = np.sum(delta_square/(expected_data+INT_MIN))
				if(chi_statistic>chi_value):
					testescape+=1
			testescape_list.append(testescape/sample_time)
		return testescape_list

	def cal_overkill_new(self, faultfree_distribution, faulty_distribution_list, fault_index=[], alpha=0.99):
		max_repetition = self.max_repetition
		if not fault_index:
			fault_index = []
			for f in self.fault_list:
				fault_index.append(f.index)

		overkill = 0
		expected_data = []
		observed_vector = []
		for i in range(len(faulty_distribution_list)):
			expected_vector_, observed_vector_ = compression_forfault(faulty_distribution_list[i], faultfree_distribution, fault_index[i])
			expected_data.append(expected_vector_ * max_repetition)
			observed_vector.append(observed_vector_)

		chi_value = self.boundary#chi2.ppf(alpha, observed_vector[0].shape[0]-1)
		for _ in range(sample_time):
			for i in range(len(faulty_distribution_list)):
				observed_sampled_data = random.choices(range(observed_vector[i].shape[0]), weights=observed_vector[i], cum_weights=None, k=max_repetition)
				observed_data = np.zeros(observed_vector[i].shape[0])
				for d in observed_sampled_data:
					observed_data[d]+=1

				delta_square = np.square(expected_data[i] - observed_data)
				chi_statistic = np.sum(delta_square/(expected_data[i]+INT_MIN))
				if(chi_statistic<=chi_value):
					overkill+=1
					break
		return overkill/sample_time

	def simulate_real_circuit(self, qc, backend, shots, times, circuit_size=5):
		summantion = {}
		for i in range(2**circuit_size):
			summantion['{0:b}'.format(i).zfill(circuit_size)] = 0

		for t in range(times):
			job_sim = execute(qc, backend, shots=shots)
			counts = job_sim.result().get_counts()
			for i in counts:
				summantion[i] += counts[i]
			print("finish ", t, " times")

		probability = np.zeros(len(summantion))
		for i in summantion:
			probability[int(i, 2)] = summantion[i]
		probability = probability/np.sum(probability)
		return(probability)

	def simulate_faulty_real_circuit(self, backend):
		for i in range(len(self.qc_faulty_list)):
			cut_list = []
			for _ in range(75):
				cut_list.append(self.qc_faulty_list[i])
			jobs = execute(cut_list, backend, shots=self.max_repetition)

			for d in range(75):
				summantion = {}
				for s in range(2**5):
					summantion['{0:b}'.format(s).zfill(5)] = 0
				counts = jobs.result().get_counts(d)
				for c in counts:
					summantion[c] += counts[c]
				self.real_faulty_distribution_list_2[i].append(to_np(summantion))
			print("finish", i, "faulty circuit, # of data:", len(self.real_faulty_distribution_list_2[i]))

	def simulate_faultfree_real_circuit(self, backend):
		cut_list = []
		for _ in range(75):
			cut_list.append(self.qc_faultfree)
		jobs = execute(cut_list, backend, shots=self.max_repetition)

		for d in range(75):
			summantion = {}
			for s in range(2**5):
				summantion['{0:b}'.format(s).zfill(5)] = 0
			counts = jobs.result().get_counts(d)
			for c in counts:
				summantion[c] += counts[c]
			self.real_faultfree_distribution_2.append(to_np(summantion))
		print(" # of data:", len(self.real_faultfree_distribution_2))

class ATPG():
	def __init__(self, circuit_size, gate_set, qr_name='q', cr_name='c'):
		self.circuit_size = circuit_size
		self.gate_set = get_gate_set(gate_set)
		
		self.qr_name = qr_name
		self.cr_name = cr_name
		self.quantumregister = QuantumRegister(self.circuit_size, self.qr_name)
		self.classicalregister = ClassicalRegister(self.circuit_size, self.cr_name)
		self.noise_model = self.get_noise_model()
		self.backend = Aer.get_backend('qasm_simulator')
		self.step = 0.01
		self.configuration_list = []
		self.alpha = 0.99
		self.beta = 0.9999
		self.gate_set = gate_set
		return
	def get_gate_set(gate_type_list):

		U_params = [0.25*pi, 0.25*pi, 0.25*pi]
		# gate_set = [Qgate.RZGate, Qgate.SXGate]
		gate_set = gate_type_list
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
		return result_gates

	def get_fault_list(self , coupling_map):
		single_fault_list = []
		two_fault_list = []

		# first insert single_fault_list
		for gate_type in self.gate_set:
			ratio_list , bias_list , threshold_list = get_params_list(gate_type)

			for ratio in ratio_list:
				V_fault = []
				for qb in range(self.circuit_size):
					V_fault.append(Variation_fault(gate_type, [qb], ratio=ratio))
			single_fault_list.append(V_fault)

			for bias in bias_list:
				V_fault = []
				for qb in range(self.circuit_size):
					V_fault.append(Variation_fault(gate_type, [qb], bias=bias))
			single_fault_list.append(V_fault)

			for threshold in threshold_list:
				T_fault = []
				for qb in range(self.circuit_size):
					T_fault.append(Threshold_lopa(gate_type, [qb], threshold=threshold))
			single_fault_list.append(T_fault)

		value = 0.05*np.pi
		f = [[value, value, value, value, value, value], [value, value, -value, value, value, -value], [value, -value, value, value, -value, value] , [value, -value, -value, value, -value, -value],
		[-value, value, value, -value, value, value], [-value, value, -value, -value, value, -value], [-value, -value, value, -value, -value, value] , [-value, -value, -value, -value, -value, -value]]
		# f = [[value, value, value, value, value, value]]
		for value in f:
			one_type_fault = []
			drop_fault = [] 
			while len(drop_fault) != len(coupling_map):
				CNOT_v_fault = []
				for i in range(len(coupling_map)):
					if coupling_map[i] in drop_fault:
						continue
					else:
						push_fault = True
						for fault in CNOT_v_fault:
							if fault.index[0] in coupling_map[i] or fault.index[1] in coupling_map[i]:
								push_fault = False
								break
						if push_fault:
							CNOT_v_fault.append(CNOT_variation_fault(coupling_map[i], value=value))
							drop_fault.append(coupling_map[i])
				one_type_fault.append(CNOT_v_fault)
			# for i in range(len(coupling_map)):
			#     one_type_fault.append([CNOT_variation_fault(coupling_map[i], value=value)])
			two_fault_list.append(deepcopy(one_type_fault))
		return (single_fault_list, two_fault_list)
		# before retrun list before [gate , index , []]
		# now return QuantumGate

	def get_quantum_gate(self, gate_type, index, parameter=[]):
		if type(index) != list :
			index = [index]

		if(gate_type in self.gate_set):
			return QuantumGate(gate_type(*parameter) , [Qubit(QuantumRegister(self.circuit_size, self.qr_name) , index[0])] , [])
			
		elif(gate_type == Qgate.CXGate):
			return QuantumGate(gate_type() , [Qubit(QuantumRegister(self.circuit_size, self.qr_name), index[0]) , Qubit(QuantumRegister(self.circuit_size, self.qr_name), index[1])], [])
			
		elif(gate_type == Qgate.Barrier):
			return QuantumGate(gate_type(1) , [Qubit(QuantumRegister(self.circuit_size, self.qr_name), index[0])] , [])
			
		else:
			print("Get Quantum Gate Error")
			print(gate_type, gate_type.__name__)

	def get_test_configuration(self, single_fault_list, two_fault_list, initial_state=np.array([1, 0])):
		configuration_list = []
		#single_fault_list = [ratio_fault_list , bias_fault_list , threshold_fault_list]，所以應該要
		for fault_type in single_fault_list:
			for i in range(self.circuit_size):
				template = self.generate_test_template(fault_type[i], np.array([1, 0]), self.get_single_gradient, cost_ratio=2)
				configuration = self.build_single_configuration(template, fault_type)
				self.simulate_configuration(configuration)
				configuration_list.append(configuration)
		print("finish build single configuration")

		if two_fault_list:
			for fault_type in two_fault_list:
				template = self.generate_test_template(fault_type[0][0], np.array([1, 0, 0, 0]), self.get_CNOT_gradient, cost_ratio=2)
				for fault_list in fault_type:
					configuration = self.build_two_configuration(template, fault_list)
					self.simulate_configuration(configuration)
					configuration_list.append(configuration)
				# break
		print("finish build two configuration")

		test_cost = 0
		initial_time = 0
		overkill = []
		testescape = []
		for configuration in configuration_list:
			test_cost += configuration.max_repetition*configuration.length
			initial_time += configuration.max_repetition
			overkill.append(configuration.sim_overkill)
			for i in configuration.sim_testescape:
				testescape.append(i)
		overall_overkill = 1
		for i in overkill:
			overall_overkill *= 1-i
		print("Alpha:", self.alpha)
		print("Beta:", self.beta)
		print("Total overkill:", 1-overall_overkill)
		print("Total testescape:", np.mean(testescape))
		print("Total test cost:", test_cost)
		print("Initial time:", initial_time)
		print("Number of configuration:", len(configuration_list))
		return configuration_list

	def getQgate(self , gate_type, index, parameter):
		#return qiskit 要求的 list 格式 
		buff = self.get_quantum_gate(gate_type = gate_type, index = index, parameter = parameter)
		return [buff.gate , buff.pos , []]

	def build_two_configuration(self, template, fault_list):
		qc_faulty_list = []
		qc_faultfree = QuantumCircuit(self.quantumregister, self.classicalregister)
		length = template[2]
		index_array = []
		for fault in fault_list:
			index_array.append(fault.index)
			# print(fault.index)
		# print(len(fault_list), len(index_array))
		for index_pair in index_array:  ##這組電路fault在哪裡
			qc_faulty = QuantumCircuit(self.quantumregister, self.classicalregister)
			for fault_index in index_array: ##這組電路幾個pair
				gate_index = 0
				if fault_index == index_pair:
					while gate_index < len(template[0]):
						qc_faulty._data.append(self.getQgate(gate_type=type(template[0][gate_index][1]), index=fault_index[0], parameter=template[0][gate_index][1].params))                
						qc_faulty._data.append(self.getQgate(gate_type=type(template[0][gate_index][0]), index=fault_index[1], parameter=template[0][gate_index][0].params))

						qc_faulty._data.append(self.getQgate(gate_type=type(template[0][gate_index+1]), index=fault_index[0], parameter=template[0][gate_index+1].params))
				
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[0], parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[1], parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=type(template[0][gate_index+2]), index=fault_index, parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[0], parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[1], parameter=[]))

						qc_faulty._data.append(self.getQgate(gate_type=type(template[0][gate_index+3]), index=fault_index[1], parameter=template[0][gate_index+3].params))
						gate_index += 4
				else:
					while gate_index < len(template[1]):
						# print(fault_index)
						qc_faulty._data.append(self.getQgate(gate_type=type(template[1][gate_index][1]), index=fault_index[0], parameter=template[1][gate_index][1].params))                
						qc_faulty._data.append(self.getQgate(gate_type=type(template[1][gate_index][0]), index=fault_index[1], parameter=template[1][gate_index][0].params))
				
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[0], parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[1], parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=type(template[1][gate_index+1]), index=fault_index, parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[0], parameter=[]))
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault_index[1], parameter=[]))
						gate_index += 2
			
			qc_faulty.measure(self.quantumregister, self.classicalregister)
			qc_faulty_list.append(qc_faulty)
			# print("faulty")
			# print(qc_faulty)
		gate_index = 0
		while gate_index < len(template[1]):
			for fault in fault_list:##這組電路幾個pair
				qc_faultfree._data.append(self.getQgate(gate_type=type(template[1][gate_index][1]), index=fault.index[0], parameter=template[1][gate_index][1].params))                
				qc_faultfree._data.append(self.getQgate(gate_type=type(template[1][gate_index][0]), index=fault.index[1], parameter=template[1][gate_index][0].params))
				
				qc_faultfree._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault.index[0], parameter=[]))
				qc_faultfree._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault.index[1], parameter=[]))
				qc_faultfree._data.append(self.getQgate(gate_type=type(template[1][gate_index+1]), index=fault.index, parameter=[]))
				qc_faultfree._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault.index[0], parameter=[]))
				qc_faultfree._data.append(self.getQgate(gate_type=Qgate.Barrier, index=fault.index[1], parameter=[]))
			gate_index += 2
		qc_faultfree.measure(self.quantumregister, self.classicalregister)
		# print(length)
		# print("faultfree")
		# print(qc_faultfree)
		new_configuration = Configuration(qc_faulty_list, qc_faultfree, length, fault_list)
		for tt in fault_list:
			print(tt.index)
		new_configuration.template = template
		# print(new_configuration)
		return new_configuration

	def build_single_configuration(self, template, fault_list):
		# template = [faulty_gate_list , faultfree_gate_list , ]
		length = template[2]
		qc_faulty_list = []
		for num_circuit in range(self.circuit_size):
			#qc_faulty 存 qiskit Qgate
			qc_faulty = QuantumCircuit(self.quantumregister, self.classicalregister)
			for gate in template[0]:
				qc_faulty._data.append(self.getQgate(gate_type=type(gate), index=num_circuit, parameter=gate.params))
				qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=num_circuit, parameter=[]))

			for gate in template[1]:
				for n in range(self.circuit_size):
					if n != num_circuit:
						qc_faulty._data.append(self.getQgate(gate_type=type(gate), index=n, parameter=gate.params))
						qc_faulty._data.append(self.getQgate(gate_type=Qgate.Barrier, index=n, parameter=[]))

			qc_faulty.measure(self.quantumregister, self.classicalregister)
			qc_faulty_list.append(qc_faulty)
			# print("faulty")
			# print(qc_faulty)
		qc_faultfree = QuantumCircuit(self.quantumregister, self.classicalregister)
		for gate in template[1]:
			for n in range(self.circuit_size):
				qc_faultfree._data.append(self.getQgate(gate_type=type(gate), index=n, parameter=gate.params))
				qc_faultfree._data.append(self.getQgate(gate_type=Qgate.Barrier, index=n, parameter=[]))
		qc_faultfree.measure(self.quantumregister, self.classicalregister)
		# print("faultfree")
		# print(qc_faultfree)
		new_configuration = Configuration(qc_faulty_list, qc_faultfree, length, fault_list)
		return new_configuration

	def generate_test_template(self, fault, quantum_state, activate_function, cost_ratio=1):
		faulty_gate_list = []
		faultfree_gate_list = []
		faulty_quantum_state, faultfree_quantum_state = deepcopy(quantum_state), deepcopy(quantum_state)

		faulty_gate_list, faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition = activate_function(fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list)
		effectsize = cal_effect_size(to_probability(faulty_quantum_state), to_probability(faultfree_quantum_state))
		for time in range(20):
			faulty_gate_list, faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition = activate_function(fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list)
			effectsize = cal_effect_size(to_probability(faulty_quantum_state), to_probability(faultfree_quantum_state))
			if repetition >= INT_MAX:
				print()
				print("error")
				print(fault)
				print(time, gate_list, faulty_quantum_state, faultfree_quantum_state)
				print()

			if effectsize>5:
				break

		print(fault, " repetition:", repetition, " len:", (len(faultfree_gate_list)), "effectsize", effectsize)
		print("ideal:", to_probability(faulty_quantum_state), to_probability(faultfree_quantum_state))
		return (faulty_gate_list, faultfree_gate_list, (len(faultfree_gate_list)))

	def get_CNOT_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
		# for 2 qubit gate
		# faultfre is a QuantumGate
		faultfree = self.get_quantum_gate(gate_type = fault.gate_type, index = fault.index, parameter = [])
		# faulty is a list of QuantumGate
		faulty = fault.get_faulty_gate(faultfree)
		# this is a list 中間的參數是index
		# faulty = self.get_quantum_gate(gate_type=fault.gate_type, index=fault.index, parameter=[])

		# QuantumGate member: gate , index , []
		faultfree_matrix = faultfree.gate.to_matrix()
		faulty_matrix = fault.gate_type().to_matrix()
		faulty_matrix = np.dot(np.kron(faulty[0].gate.to_matrix(), np.eye(2)), faulty_matrix)
		faulty_matrix = np.dot(faulty_matrix, np.kron(np.eye(2), faulty[2].gate.to_matrix()))

		parameter_list = [0, 0, 0, 0, 0, 0]
		for i in range(SEARCH_TIME):
			self.two_gradient(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state)
		faultfree_gate_list.append([Qgate.U3Gate(parameter_list[0], parameter_list[1], parameter_list[2]), Qgate.U3Gate(parameter_list[3], parameter_list[4], parameter_list[5])])
		faultfree_gate_list.append(Qgate.CXGate())
		
		faulty_gate_list.append([Qgate.U3Gate(parameter_list[0], parameter_list[1], parameter_list[2]), Qgate.U3Gate(parameter_list[3], parameter_list[4], parameter_list[5])])
		# for qiskit func, so need to append QuantumGate not QuantumGate.gate
		faulty_gate_list.append(faulty[0].gate)
		faulty_gate_list.append(Qgate.CXGate())
		faulty_gate_list.append(faulty[2].gate)

		faultfree_quantum_state = matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faultfree_matrix], faultfree_quantum_state)
		faulty_quantum_state = matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faulty_matrix], faulty_quantum_state)

		faulty_quantum_state_ = to_probability(faulty_quantum_state)
		faultfree_quantum_state_ = to_probability(faultfree_quantum_state)
		repetition, boundary = compute_repetition(faulty_quantum_state_, faultfree_quantum_state_, self.alpha, self.beta)
		return (faulty_gate_list, faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition)

	def two_gradient(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state):
		score = vector_distance(
				matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faultfree_matrix], faultfree_quantum_state), 
				matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faulty_matrix], faulty_quantum_state))
		new_parameter_list = [0]*len(parameter_list)

		for i in range(len(parameter_list)):
			parameter_list[i] += self.step
			up_score = vector_distance(
				matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faultfree_matrix], faultfree_quantum_state), 
				matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faulty_matrix], faulty_quantum_state))
			parameter_list[i] -= 2*self.step

			down_score = vector_distance(
				matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faultfree_matrix], faultfree_quantum_state), 
				matrix_operation([[U3(parameter_list[0:3]), U3(parameter_list[3:6])], faulty_matrix], faulty_quantum_state))
			parameter_list[i] += self.step

			if score == up_score == down_score:
				new_parameter_list[i] += self.step
			elif(up_score > score and up_score >= down_score):
				new_parameter_list[i] += self.step
			elif(down_score > score and down_score >= up_score):
				new_parameter_list[i] -= self.step
		
		for i in range(len(parameter_list)):
			parameter_list[i] += new_parameter_list[i]

		return 
	
	def get_single_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
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
		for i in range(SEARCH_TIME):
			self.single_gradient(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		
		# score = vector_distance(faultfree_quantum_state, faulty_quantum_state)
		faulty_parameter = self.check_fault(fault, parameter_list)
		faulty_gate_list.append(Qgate.U3Gate(faulty_parameter[0], faulty_parameter[1], faulty_parameter[2]))
		faulty_gate_list.append(faulty.gate)
		
		faultfree_gate_list.append(Qgate.U3Gate(parameter_list[0], parameter_list[1], parameter_list[2]))
		faultfree_gate_list.append(faultfree.gate)
		# print(faulty_matrix)
		faultfree_quantum_state = matrix_operation([U3(parameter_list), faultfree_matrix], faultfree_quantum_state, max_size=2)
		faulty_quantum_state = matrix_operation([U3(self.check_fault(fault, parameter_list)), faulty_matrix], faulty_quantum_state, max_size=2)
		# print(faultfree_quantum_state, faulty_quantum_state)
		faulty_quantum_state_ = to_probability(faulty_quantum_state)
		faultfree_quantum_state_ = to_probability(faultfree_quantum_state)
		# faulty_quantum_state_, faultfree_quantum_state_ = compression(faulty_quantum_state_, faultfree_quantum_state_)
		repetition, boundary = compute_repetition(faulty_quantum_state_, faultfree_quantum_state_, self.alpha, self.beta)
		# print(parameter_list, repetition)
		# print("repetition:", parameter_list, repetition)
		# print(np.array(faultfree_quantum_state*np.conj(faultfree_quantum_state)), np.array(faulty_quantum_state*np.conj(faulty_quantum_state)))
		# print(faultfree_quantum_state, faulty_quantum_state)
		return (faulty_gate_list, faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition)

	# 只對parameter list進行
	def single_gradient(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		# parameter_list = [0 , 0 , 0] 應該根據gate型態給予參數
		score = vector_distance(
				matrix_operation([U3(parameter_list), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.check_fault(fault, parameter_list)), faulty_matrix], faulty_quantum_state, max_size=2))
		new_parameter_list = [0]*len(parameter_list)
		# temp_value = 0
		for i in range(len(parameter_list)):
			parameter_list[i] += self.step
			up_score = vector_distance(
				matrix_operation([U3(parameter_list), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.check_fault(fault, parameter_list)), faulty_matrix], faulty_quantum_state, max_size=2))

			parameter_list[i] -= 2*self.step

			down_score = vector_distance(
				matrix_operation([U3(parameter_list), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.check_fault(fault, parameter_list)), faulty_matrix], faulty_quantum_state, max_size=2))
			parameter_list[i] += self.step


			if score == up_score == down_score:
				new_parameter_list[i] += self.step
			elif(up_score > score and up_score >= down_score):
				new_parameter_list[i] += self.step
			elif(down_score > score and down_score >= up_score):
				new_parameter_list[i] -= self.step
		for i in range(len(parameter_list)):
			parameter_list[i] += new_parameter_list[i]
		return 
	# fault.gate_type 要修正
	def check_fault(self, fault, parameter_list):
		if(type(fault) == Variation_fault and fault.gate_type in self.qiskit_gate_set):
			res = []
			for i in range(len(parameter_list)):
				res.append(parameter_list[i]*fault.ratio[i]+fault.bias[i])
			return res
		elif(type(fault) == Threshold_lopa and fault.gate_type in self.qiskit_gate_set):
			res = []
			for i in range(len(parameter_list)):
				res.append(fault.threshold[i] if parameter_list[i] > fault.threshold[i] else parameter_list[i])
			return res
		else:
			return parameter_list

	def get_activation_gate(self, fault):
		if(type(fault) == Variation_fault):
			ratio = []
			bias = []
			for r in fault.ratio:
				if r < 1:
					ratio.append(2 * np.pi)
				else:
					ratio.append(0)
				bias.append(0)
			return self.get_quantum_gate(gate_type=fault.gate_type, index=fault.index, parameter=ratio)
		elif(type(fault) == Threshold_lopa):
			threshold = []
			for t in fault.threshold:
				if t < (2*np.pi):
					threshold.append(2*np.pi - INT_MIN)
				else:
					threshold.append(0)
			return self.get_quantum_gate(gate_type=fault.gate_type, index=fault.index, parameter=threshold)
		elif(type(fault) == Qgate.CXGate):
			return self.get_quantum_gate(gate_type=fault.gate_type, index=fault.index, parameter= [])

	def simulate_configuration(self, configuration, shots=200000):
		configuration.repetition_list = []
		configuration.sim_faultfree_distribution = []
		configuration.sim_faulty_distribution_list = []
		configuration.max_repetition = 0
		configuration.boundary = 0 
		
		job_sim = execute(configuration.qc_faultfree, self.backend, noise_model=self.noise_model, shots=shots)
		
		summantion_free = {}
		for i in range(2**self.circuit_size):
			summantion_free['{0:b}'.format(i).zfill(self.circuit_size)] = 0
		counts = job_sim.result().get_counts()
		for i in counts:
			summantion_free[i] += counts[i]
		configuration.sim_faultfree_distribution = to_np(summantion_free)

		for number, qc_faulty in enumerate(configuration.qc_faulty_list):
			job_sim = execute(qc_faulty, self.backend, noise_model=self.noise_model, shots=shots)

			summantion_faulty = {}
			for i in range(2**self.circuit_size):
				summantion_faulty['{0:b}'.format(i).zfill(self.circuit_size)] = 0
			counts = job_sim.result().get_counts()
			for i in counts:
				summantion_faulty[i] += counts[i]
			configuration.sim_faulty_distribution_list.append(to_np(summantion_faulty)) 

			faulty_distribution, faultfree_distribution = compression_forfault(configuration.sim_faulty_distribution_list[-1], configuration.sim_faultfree_distribution, deepcopy(configuration.fault_list[number].index))
			repetition, boundary = compute_repetition(faulty_distribution, faultfree_distribution, self.alpha, self.beta)
			configuration.repetition_list.append((repetition, boundary))

		fault_index = []
		for f in configuration.fault_list:
			fault_index.append(f.index)

		temp_list = [0, 0]
		for c in configuration.repetition_list:
			if c[0] > temp_list[0]:
				temp_list[0] = c[0]
				temp_list[1] = c[1]

		configuration.max_repetition = temp_list[0]
		configuration.boundary = temp_list[1]

		configuration.sim_overkill = configuration.cal_overkill_new(configuration.sim_faultfree_distribution, configuration.sim_faulty_distribution_list, fault_index, alpha=self.alpha)
		configuration.sim_testescape = configuration.cal_testescape_new(configuration.sim_faulty_distribution_list, configuration.sim_faulty_distribution_list, fault_index, alpha=self.alpha)
		
		print(configuration)

		return

	def simulate_circuit(self, qc, backend, shots, noise_model = False):
		if noise_model:
			noise_model = self.noise_model
			job_sim = execute(qc, backend, noise_model=noise_model, shots=shots)
		else:
			job_sim = execute(qc, backend, shots=shots)

		summantion = {}
		for i in range(2**self.circuit_size):
			summantion['{0:b}'.format(i).zfill(self.circuit_size)] = 0
		counts = job_sim.result().get_counts()
		for i in counts:
			summantion[i] += counts[i]
		
		return( to_np(summantion) )

	def get_noise_model(self):
		# Error probabilities
		prob_1 = 0.001  # 1-qubit gate
		prob_2 = 0.01   # 2-qubit gate

		# Depolarizing quantum errors
		error_1 = standard_errors.depolarizing_error(prob_1, 1)
		error_2 = standard_errors.depolarizing_error(prob_2, 2)
		error_3 = ReadoutError([[0.985,0.015],[0.015,0.985]])

		# Add errors to noise model
		noise_model = NoiseModel()
		noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
		noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
		noise_model.add_all_qubit_readout_error(error_3)

		return noise_model
		