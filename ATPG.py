import numpy as np
# import qiskit
import math
import cmath
from Fault import *
from Gate import *
from scipy.stats import chi2, ncx2
import qiskit.circuit.library as Qgate
from qiskit.circuit import Parameter
from qiskit.circuit.quantumregister import Qubit
# from qiskit.quantum_info import process_fidelity
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError
from qiskit import Aer
from qiskit import execute, transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions import *
# import statsmodels.stats.power as smp
from qiskit import transpile
from numpy import pi
import random
from QuantumGate import *
from util import *

random.seed(427427)
# INT_MIN = 1E-100
# INT_MAX = 1E15
# S = 700
# sample_time = 10000
# threshold = 0.01
# r = 0.1

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
		self.gate_set = gate_set
		
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

		

		self.basis_gates = [gate.__name__[:-4].lower() for gate in self.gate_set]
		q = QuantumCircuit(1)
		self.theta = Parameter('theta')
		self.phi = Parameter('phi')
		self.lam = Parameter('lam')
		q.u(self.theta, self.phi, self.lam, 0)
		self.effective_u_ckt = transpile(q, basis_gates = self.basis_gates, optimization_level = 3)

		# temporary
		self.SERIAL_NUMBER = 0
		
		return
	
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
			# for i in range(self.circuit_size):
			# TODO Potential error
			for i in range(1):
				template = self.generate_test_template(fault_type[i], np.array([1, 0]), self.get_single_optimal_method, cost_ratio=2)
				configuration = self.build_single_configuration(template, fault_type)
				self.simulate_configuration(configuration)
				configuration_list.append(configuration)
		print("finish build single configuration")

		# CAUTION: UNCOMMENT
		# if two_fault_list:
		# 	for fault_type in two_fault_list:
		# 		template = self.generate_test_template(fault_type[0][0], np.array([1, 0, 0, 0]), self.get_CNOT_optimal_method, cost_ratio=2)
		# 		for fault_list in fault_type:
		# 			configuration = self.build_two_configuration(template, fault_list)
		# 			self.simulate_configuration(configuration)
		# 			configuration_list.append(configuration)
		# 		# break
		# print("finish build two configuration")

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

		# faulty_gate_list, faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition = activate_function(fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list)
		# effectsize = cal_effect_size(to_probability(faulty_quantum_state), to_probability(faultfree_quantum_state))
		for time in range(MAX_ELEMENT):
			faulty_gate_list, faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition = activate_function(fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list)
			effectsize = cal_effect_size(to_probability(faulty_quantum_state), to_probability(faultfree_quantum_state))
			if repetition >= INT_MAX:
				print()
				print("error")
				print(fault)
				print(time, gate_list, faulty_quantum_state, faultfree_quantum_state)
				print()

			if effectsize > MIN_REQUIRED_EFFECT_SIZE:
				break

		# overall gradient descent
		faulty_quantum_state, faultfree_quantum_state = deepcopy(quantum_state), deepcopy(quantum_state)
		self.overall_gradient(fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list)

		print(fault, " repetition:", repetition, " len:", (len(faultfree_gate_list)), "effectsize", effectsize)
		print("ideal:", to_probability(faulty_quantum_state), to_probability(faultfree_quantum_state))
		print("faulty_gate_list: ")
		for faulty_gates in faulty_gate_list:
			if type(faulty_gates) == list:
				print("\t[", end = "")
				for faulty_gate in faulty_gates:
					print(faulty_gate.__class__.__name__, [param.__float__() for param in faulty_gate.params], ", ", end = "")
				print("]")
				# print("\t", faulty_gates)
			else:
				print("\t", faulty_gates.__class__.__name__, [param.__float__() for param in faulty_gates.params])
		print("faultfree_gate_list: ")
		for faultfree_gates in faultfree_gate_list:
			if type(faultfree_gates) == list:
				print("\t[", end = "")
				for faultfree_gate in faultfree_gates:
					print(faultfree_gate.__class__.__name__, [param.__float__() for param in faultfree_gate.params], ", ", end = "")
				print("]")
				# print("\t", faulty_gates)
			else:
				print("\t", faultfree_gates.__class__.__name__, [param.__float__() for param in faultfree_gates.params])
		return (faulty_gate_list, faultfree_gate_list, (len(faultfree_gate_list)))

	def get_CNOT_optimal_method(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):
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
		# for i in range(SEARCH_TIME):
			# self.two_gradient(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state)
		self.two_annealing(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state)
		# faultfree_gate_list.append([Qgate.U3Gate(parameter_list[0], parameter_list[1], parameter_list[2]), Qgate.U3Gate(parameter_list[3], parameter_list[4], parameter_list[5])])
		faultfree_gate_list.append(self.U_to_gate_set_transpiler_to_gate_list(parameter_list[0:3]) + self.U_to_gate_set_transpiler_to_gate_list(parameter_list[3:]))
		faultfree_gate_list.append(Qgate.CXGate())
		
		# faulty_gate_list.append([Qgate.U3Gate(parameter_list[0], parameter_list[1], parameter_list[2]), Qgate.U3Gate(parameter_list[3], parameter_list[4], parameter_list[5])])
		faulty_gate_list.append(self.U_to_gate_set_transpiler_to_gate_list(parameter_list[0:3]) + self.U_to_gate_set_transpiler_to_gate_list(parameter_list[3:]))
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

	def two_annealing(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state):
		def score(parameters):
			return vector_distance(
				matrix_operation([[U3(parameters[0:3]), U3(parameters[3:6])], faultfree_matrix], faultfree_quantum_state), 
				matrix_operation([[U3(parameters[0:3]), U3(parameters[3:6])], faulty_matrix], faulty_quantum_state))

		current_sol = parameter_list
		best_sol = None
		best_score = INT_MIN
		T = T_init
		step = self.step
		anneal_times = 0

		while T > T_min:
			current_score = score(current_sol)
			for i in range(len(current_sol)):
				current_sol[i] += step
				up_score = score(current_sol)
				current_sol[i] -= step*2
				down_score = score(current_sol)
				current_sol[i] += step

				if up_score == current_score and down_score == current_score:
					luck = random.random()
					if luck < 1/3:
						current_sol[i] -= step
					elif luck > 2/3:
						current_sol[i] += step
				elif up_score >= current_score and up_score > down_score:
					current_sol[i] += step
					step *= step_ratio
				elif down_score >= current_score and down_score > up_score:
					current_sol[i] -= step
					step *= step_ratio
				elif up_score >= down_score:
					if random.random() < math.exp((up_score - current_score) / T):
						current_sol[i] += step
				else:
					if random.random() < math.exp((down_score - current_score) / T):
						current_sol[i] -= step

			current_score = score(current_sol)
			if current_score > best_score:
				best_score = current_score
				best_sol = deepcopy(current_sol)

			T *= T_ratio
			anneal_times += 1

		parameter_list = best_sol
		# print("anneal_times: ", anneal_times)
	
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
		# parameter_list = self.single_annealing_3_dir(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		# parameter_list = self.single_deterministic(parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault)
		# print("after annealing: ", parameter_list)
		# score = vector_distance(faultfree_quantum_state, faulty_quantum_state)

		faulty_parameter = self.faulty_activation_gate(fault, parameter_list)
		# faulty_gate_list.append(Qgate.U3Gate(faulty_parameter[0], faulty_parameter[1], faulty_parameter[2]))
		faulty_gate_list = faulty_gate_list + self.U_to_gate_set_transpiler_to_gate_list(faulty_parameter)
		faulty_gate_list.append(faulty.gate)
		
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
		return (faulty_gate_list, faultfree_gate_list, faulty_quantum_state, faultfree_quantum_state, repetition)

	# 只對parameter list進行
	def single_gradient(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		# parameter_list = [0 , 0 , 0] 應該根據gate型態給予參數
		def score(parameters):
			return vector_distance(
					matrix_operation([U3(parameters), faultfree_matrix], faultfree_quantum_state, max_size=2), 
					matrix_operation([U3(self.faulty_activation_gate(fault, parameters)), faulty_matrix], faulty_quantum_state, max_size=2))
		# print("score: ", score)

		for j in range(SEARCH_TIME):
			new_parameter_list = [0]*len(parameter_list)
			for i in range(len(parameter_list)):
				current_score = score(parameter_list)
				parameter_list[i] += self.step
				up_score = score(parameter_list)
				parameter_list[i] -= 2*self.step
				down_score = score(parameter_list)
				parameter_list[i] += self.step

				if(up_score > current_score and up_score >= down_score):
					# new_parameter_list[i] += self.step
					new_parameter_list[i] += self.step*(up_score - current_score)
				elif(down_score > current_score and down_score >= up_score):
					# new_parameter_list[i] -= self.step
					new_parameter_list[i] -= self.step*(down_score - current_score)
			if new_parameter_list == [0, 0, 0]:
				break
			for i in range(len(parameter_list)):
				parameter_list[i] += new_parameter_list[i]

		print("score: ", score(parameter_list))
		return parameter_list

	def single_annealing_8_dir(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		anneal_times = 0
		class candidate_solution():
			def __init__(self, parameter_list, using_temp, using_step):
				self.parameter_list = deepcopy(parameter_list)
				self.using_temp = using_temp
				self.using_step = using_step
				self.from_direction = -1

		def score(parameters):
			return vector_distance(
				matrix_operation([U3(parameters), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.faulty_activation_gate(fault, parameters)), faulty_matrix], faulty_quantum_state, max_size=2))

		candidate_solution_list = [candidate_solution(parameter_list, T_init, 2*self.step)]
		best_score = INT_MIN
		best_solution = None

		# start annealing
		while candidate_solution_list:
			target = candidate_solution_list[0]
			candidate_solution_list = candidate_solution_list[1:]
			# stop criteria
			if target.using_temp < T_min:
				continue
			current_score = score(target.parameter_list)
			# print("current_score: ", current_score)
			# one processed
			anneal_times += 1
			# compare
			if current_score > best_score:
				best_score = current_score
				best_solution = target.parameter_list

			# there should be 8 candidate-candidates
			candidate_candidates = [candidate_solution(target.parameter_list, target.using_temp*T_ratio, target.using_step*step_ratio) for _ in range(8)]
			adding_new_candidate = 0
			for i in range(len(candidate_candidates)):
				candidate_candidates[i].from_direction = i
				for j in range(len(target.parameter_list)):
					if (i >> j) % 2:
						candidate_candidates[i].parameter_list[j] = target.parameter_list[j] + target.using_step
					else:
						candidate_candidates[i].parameter_list[j] = target.parameter_list[j] - target.using_step
				# preventing going back
				if i + target.from_direction == 8:
					continue
				# cal score and decide
				this_score = score(candidate_candidates[i].parameter_list)
				# print("this_score: ", this_score)
				# if random.random() < math.exp((this_score - current_score) / target.using_temp):
				if this_score >= current_score:
					candidate_solution_list.append(candidate_candidates[i])
					adding_new_candidate += 1

			# print("adding_new_candidate: ", adding_new_candidate)
			# print("size of list: ", len(candidate_solution_list))

		parameter_list = best_solution
		print("Annealing times: ", anneal_times)
		return best_solution

	def single_annealing_3_dir(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		def score(parameters):
			return vector_distance(
				matrix_operation([U3(parameters), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.faulty_activation_gate(fault, parameters)), faulty_matrix], faulty_quantum_state, max_size=2))

		current_sol = parameter_list
		best_sol = None
		best_score = INT_MIN
		T = T_init
		step = self.step
		anneal_times = 0

		while T > T_min:
			current_score = score(current_sol)
			for i in range(len(current_sol)):
				current_sol[i] += step
				up_score = score(current_sol)
				current_sol[i] -= step*2
				down_score = score(current_sol)
				current_sol[i] += step

				if up_score == current_score and down_score == current_score:
					luck = random.random()
					if luck < 1/3:
						current_sol[i] -= step
					elif luck > 2/3:
						current_sol[i] += step
				elif up_score >= current_score and up_score > down_score:
					current_sol[i] += step
					step *= step_ratio
				elif down_score >= current_score and down_score > up_score:
					current_sol[i] -= step
					step *= step_ratio
				elif up_score >= down_score:
					if random.random() < math.exp((up_score - current_score) / T):
						current_sol[i] += step
				else:
					if random.random() < math.exp((down_score - current_score) / T):
						current_sol[i] -= step

			current_score = score(current_sol)
			if current_score > best_score:
				best_score = current_score
				best_sol = deepcopy(current_sol)

			T *= T_ratio
			anneal_times += 1

		return best_sol
		# print("anneal_times: ", anneal_times)

	def single_annealing_3_dir_enhance(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		def score(parameters):
			return vector_distance(
				matrix_operation([U3(parameters), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.faulty_activation_gate(fault, parameters)), faulty_matrix], faulty_quantum_state, max_size=2))

		current_sol = parameter_list
		best_sol = None
		best_score = INT_MIN
		T = T_init
		step = self.step
		anneal_times = 0

		while T > T_min:
			current_score = score(current_sol)
			for i in range(len(current_sol)):
				current_sol[i] += step
				up_score = score(current_sol)
				current_sol[i] -= step*2
				down_score = score(current_sol)
				current_sol[i] += step

				if up_score == current_score and down_score == current_score:
					luck = random.random()
					if luck < 1/3:
						current_sol[i] -= step
					elif luck > 2/3:
						current_sol[i] += step
				elif up_score >= current_score and up_score > down_score:
					current_sol[i] += step
					step *= step_ratio
				elif down_score >= current_score and down_score > up_score:
					current_sol[i] -= step
					step *= step_ratio
				elif up_score >= down_score:
					if random.random() < math.exp((up_score - current_score) / T):
						current_sol[i] += step
					elif random.random() < math.exp((down_score - current_score) / T):
						current_sol[i] -= step
				else:
					if random.random() < math.exp((down_score - current_score) / T):
						current_sol[i] -= step
					elif random.random() < math.exp((up_score - current_score) / T):
						current_sol[i] += step

			current_score = score(current_sol)
			if current_score > best_score:
				best_score = current_score
				best_sol = deepcopy(current_sol)

			T *= T_ratio
			anneal_times += 1

		return best_sol
		# print("anneal_times: ", anneal_times)

	def single_annealing_random_dir(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		def score(parameters):
			return vector_distance(
				matrix_operation([U3(parameters), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.faulty_activation_gate(fault, parameters)), faulty_matrix], faulty_quantum_state, max_size=2))

		current_sol = parameter_list
		current_score = score(current_sol)
		best_sol = None
		best_score = INT_MIN
		T = T_init
		step = self.step
		anneal_times = 0
		# past_dir = -1
		past_dir = [0, 0, 0]

		while T > T_min:
			# # 8 dir
			# # choose direction
			# while 1:
			# 	new_dir = random.randint(0, 7);
			# 	if new_dir + past_dir != 8:
			# 		break
			# # print("new dir: ", new_dir)

			# new_sol = [0, 0, 0]
			# for i in range(len(current_sol)):
			# 	if (new_dir >> i) % 2:
			# 		new_sol[i] = current_sol[i] + step
			# 	else:
			# 		new_sol[i] = current_sol[i] - step
			# new_score = score(new_sol)
			# past_dir = new_dir

			def get_dir():
				# choice = [-1, 0, 1] # 27 dir
				choice = [-1, 1] # 8 dir
				while 1:
					rt_dir = [random.choice(choice) for _ in range(3)]
					if (not all(d == 0 for d in rt_dir)) and (not all(past_dir[i]+rt_dir[i] == 0 for i in range(3))):
						# prevent not moving
						# prevent moving backwards
						break
				# print("new dir: ", rt_dir)
				return rt_dir
			passing = False

			while not passing:
				new_dir = get_dir()
				new_sol = [current_sol[i] + new_dir[i]*step for i in range(3)]
				new_score = score(new_sol)

				if new_score > current_score:
					current_sol = deepcopy(new_sol)
					current_score = new_score
					step *= step_ratio
					passing = True
				elif random.random() < math.exp((new_score - current_score) / T):
					current_sol = deepcopy(new_sol)
					current_score = new_score
					passing = True

			if current_score > best_score:
				best_sol = deepcopy(current_sol)
				best_score = current_score

			T *= T_ratio
			anneal_times += 1
			past_dir = deepcopy(new_dir)

		# print("best_sol: ", best_sol)
		# parameter_list = deepcopy(best_sol)
		# parameter_list = best_sol
		# print("anneal_times: ", anneal_times)

		return best_sol

	def single_deterministic(self, parameter_list, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		pass
		RFO = np.matmul(faulty_matrix.transpose(), faultfree_matrix)
		w, v = np.linalg.eig(RFO)
		# w size should be two
		if np.abs(w[0]) > np.abs(w[1]):
			v0 = v[0][0]
			v1 = v[1][0]
		else:
			v0 = v[0][1]
			v1 = v[1][1]
		p0 = faultfree_quantum_state[0]
		p1 = faultfree_quantum_state[1]

		best_theta = 0
		best_phi = 1j
		best_lam = 1j
		for theta in np.linspace(-np.pi, np.pi, SEARCH_TIME):
			phi = 1j*np.log((np.sqrt(2)*p0 - v0) / v1 + 0j)
			lam = -1j*np.log((p0 - np.sqrt(2)*v0) / p1 + 0j)

			if math.isnan(phi) or math.isnan(lam):
				continue

			if np.abs(np.imag(phi)*np.imag(lam)) < np.abs(np.imag(best_phi)*np.imag(best_lam)):
				best_theta = theta
				best_phi = phi
				best_lam = lam

		parameter_list = [best_theta, np.real(best_phi), np.real(best_lam)]
		return parameter_list

	def single_explore(self, faulty_matrix, faultfree_matrix, faulty_quantum_state, faultfree_quantum_state, fault):
		def score(parameters):
			return vector_distance(
				matrix_operation([U3(parameters), faultfree_matrix], faultfree_quantum_state, max_size=2), 
				matrix_operation([U3(self.faulty_activation_gate(fault, parameters)), faulty_matrix], faulty_quantum_state, max_size=2))

		# with open("./4D_plot/" + str(self.SERIAL_NUMBER) + ".csv", 'w') as f:
		# 	# explore the parameters
		# 	f.write("Fault: , " + fault.description + "\n")
		# 	f.write("faulty_matrix, " + np.array2string(faulty_matrix).replace('\n', '').replace(',', ';') + "\n")
		# 	f.write("faultfree_matrix, " + np.array2string(faultfree_matrix).replace('\n', '').replace(',', ';') + "\n")
		# 	f.write("faulty_quantum_state, " + np.array2string(faulty_quantum_state).replace('\n', '').replace(',', ';') + "\n")
		# 	f.write("faultfree_quantum_state, " + np.array2string(faultfree_quantum_state).replace('\n', '').replace(',', ';') + "\n")
		# 	f.write("theta, phi, lam, score \n")
		# 	for theta in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
		# 		for phi in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
		# 			for lam in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
		# 				f.write(str(theta) + ", " + str(phi) + ", " + str(lam) + ", " + str(score([theta, phi, lam])) + "\n")

		# self.SERIAL_NUMBER += 1

		results = []
		for theta in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=21, endpoint = True):
					results.append([[theta, phi, lam], score([theta, phi, lam])])
		return max(results, key = lambda x: x[1])[0]

	def faulty_activation_gate(self, fault, parameter_list):
		# first transpile
		transpile_result_ckt = self.U_to_gate_set_transpiler(parameter_list)
		if type(fault) == Variation_fault:
			for gate, _, _ in transpile_result_ckt.data:
				if type(gate) == fault.gate_type:
					for i in range(len(gate.params)):
						# potential bug: ParameterExpression force turn to float
						gate.params[i] = gate.params[i].__float__()*fault.ratio[i]+fault.bias[i]
		elif type(fault) == Threshold_lopa:
			for gate, _, _ in transpile_result_ckt.data:
				if type(gate) == fault.gate_type:
					for i in range(len(gate.params)):
						# potential bug: ParameterExpression force turn to float
						gate.params[i] = fault.threshold[i] if gate.params[i].__float__() > fault.threshold[i] else gate.params[i].__float__()
		# transpile back
		return self.ckt_to_U_params_transpiler(transpile_result_ckt)

	def U_to_gate_set_transpiler_to_gate_list(self, U_params):
		ckt = self.U_to_gate_set_transpiler(U_params)
		return [gate for gate, _, _ in ckt.data]

	def U_to_gate_set_transpiler(self, U_params):
		# basis_gates = [gate.__name__[:-4].lower() for gate in self.gate_set]
		# q = QuantumCircuit(1)
		# q.u(*U_params, 0)
		# result_ckt = transpile(q, basis_gates = basis_gates, optimization_level = 3)
		result_ckt = self.effective_u_ckt.bind_parameters({self.theta: U_params[0], self.phi: U_params[1], self.lam: U_params[2]})
		# result_gates = [gate for gate, _, _ in result_ckt.data]
		return result_ckt

	def ckt_to_U_params_transpiler(self, ckt):
		result_ckt = transpile(ckt, basis_gates = ['u3'], optimization_level = 3)
		# should be only one gate
		# possible no gate: [0, 0, 0]
		if len(result_ckt.data) == 0:
			# print("Insert [0, 0, 0] alarm")
			return [0, 0, 0]
		# potential bug
		return [param.__float__() for param in result_ckt.data[0][0].params]

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

	def simulate_configuration(self, configuration, shots = 200000):
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
		noise_model.add_all_qubit_quantum_error(error_1, [gate.__name__[:-4].lower() for gate in self.gate_set])
		noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
		noise_model.add_all_qubit_readout_error(error_3)

		return noise_model


	
	# func below for overall gradient
	def overall_gradient(self, fault, faulty_quantum_state, faultfree_quantum_state, faulty_gate_list, faultfree_gate_list):

		def score(which_element , faulty_parameter_list , faultfree_parameter_list):
			U_and_faulty_pair_gate_list[which_element][0].params = self.faulty_activation_gate(fault , faulty_parameter_list)
			U_and_faultfree_pair_gate_list[which_element][0].params = faultfree_parameter_list
			faulty_matrix = faultfree_matrix = np.eye(2)

			for pair in U_and_faulty_pair_gate_list:
				faulty_matrix = np.matmul(faulty_matrix , pair[0].to_matrix())
				faulty_matrix = np.matmul(faulty_matrix , pair[1].to_matrix())
			for pair in U_and_faultfree_pair_gate_list:
				faultfree_matrix = np.matmul(faultfree_matrix , pair[0].to_matrix())
				faultfree_matrix = np.matmul(faultfree_matrix , pair[1].to_matrix())

			return vector_distance(np.matmul(faulty_matrix , faulty_quantum_state) , np.matmul(faultfree_matrix , faultfree_quantum_state))


		if fault == CNOT_variation_fault:
			pass;
		else:	
			# print("faulty gate list before gradient")
			# for gate in faulty_gate_list:
			# 	print(gate.params)

			element_len = len(self.effective_u_ckt.data) + 1
			# faulty_element_list = list(np.array_split(faulty_gate_list, len(faulty_gate_list) / element_len))
			# faultfree_element_list = list(np.array_split(faultfree_gate_list, len(faultfree_gate_list) / element_len))
			faulty_element_list = []
			faultfree_element_list = []
			for i in range(0 , len(faulty_gate_list) , element_len):
				temp1 = []
				temp2 = []
				for j in range(element_len):
					temp1.append(faulty_gate_list[i + j])
					temp2.append(faultfree_gate_list[i + j])
				faulty_element_list.append(temp1)
				faultfree_element_list.append(temp2)

			# [ [U , faulty_gate] , []]
			U_and_faulty_pair_gate_list = self.get_U_and_gate_pair_list(faulty_element_list , element_len)
			U_and_faultfree_pair_gate_list = self.get_U_and_gate_pair_list(faultfree_element_list , element_len)

			#　do on element gradient one time
			for k in range(len(U_and_faulty_pair_gate_list)):
				faulty_parameter_list = [param.__float__() for param in U_and_faulty_pair_gate_list[k][0].params]
				faultfree_parameter_list = [param.__float__() for param in U_and_faultfree_pair_gate_list[k][0].params]
				# print("faulty_parameter_list = ", faulty_parameter_list)
				# print("faultfree_parameter_list", faultfree_parameter_list)
				for i in range(SEARCH_TIME):
					new_parameter_list = [0 , 0 , 0]
					for j in range(3):
						current_score = score(k , faulty_parameter_list , faultfree_parameter_list)
						faulty_parameter_list[j] += self.step
						faultfree_parameter_list[j] += self.step

						up_score = score(k , faulty_parameter_list , faultfree_parameter_list)
						faulty_parameter_list[j] -= 2*self.step
						faultfree_parameter_list[j] -= 2*self.step

						down_score = score(k , faulty_parameter_list , faultfree_parameter_list)
						faulty_parameter_list[j] += self.step
						faultfree_parameter_list[j] += self.step

						if up_score > current_score and up_score >= down_score:
							# print("up is better")
							new_parameter_list[j] += self.step*(up_score - current_score)
						elif down_score > current_score and down_score >= up_score:
							# print("down is better")
							new_parameter_list[j] -= self.step*(down_score - current_score)
					# print("current_score = ",current_score)
					# print("up_score = ",up_score)
					# print("down_score = ",down_score)
					if new_parameter_list == [0 , 0 , 0]:
						break
					for j in range(3):
						faulty_parameter_list[j] += new_parameter_list[j] 
						faultfree_parameter_list[j] += new_parameter_list[j]
				U_and_faulty_pair_gate_list[k][0].params = faulty_parameter_list
				U_and_faultfree_pair_gate_list[k][0].params = faultfree_parameter_list


			faulty_gate_list = self.transpile_U_and_gate_pair_list_to_gate_list(U_and_faulty_pair_gate_list)
			faultfree_gate_list = self.transpile_U_and_gate_pair_list_to_gate_list(U_and_faultfree_pair_gate_list)
			# print("faulty gate list after gradient")
			# for gate in faulty_gate_list:
			# 	print(gate.params)
	def get_U_and_gate_pair_list(self , element_list , element_len):
		if element_len == 2:
			return element_list

		result = []
		# print(element_list)
		for element in element_list:
			q = QuantumCircuit(1)

			for i in range(element_len - 1):
				q.append(element[i] , [0])

			ckt_to_u = transpile(q , basis_gates = ['u3'] , optimization_level = 3)
			# print("*************************")
			# print(ckt_to_u)
			result.append([ckt_to_u.data[0][0] , element[element_len - 1]])

		return result
			

	def transpile_U_and_gate_pair_list_to_gate_list(self , U_and_gate_pair_list):
		q = QuantumCircuit(1)
		for pair in U_and_gate_pair_list:
			q.append(pair[0] , [0])
			q.append(pair[1] , [0])
		result_ckt = transpile(q , basis_gates = self.basis_gates , optimization_level = 3)
		return [gate for gate, _, _ in result_ckt.data]
			


			