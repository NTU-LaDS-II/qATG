from math import ceil
import numpy as np
from copy import deepcopy
from QuantumGate import *
from scipy.stats import chi2, ncx2

INT_MIN = 1E-100
INT_MAX = 1E15
SEARCH_TIME = 700
sample_time = 10000
threshold = 0.01
r = 0.1
def get_qiskit_gate(gate_tye):
	
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

def get_params_list(gate):
	# gate is a gate
	amount = gate.__init__.__code__.co_argcount - len(gate.__init__.__defaults__) - 1
	# -1 for self
	ratio_list = []
	bias_list = []
	threshold_list = []
	for i in range(amount):
		ratio_candidate = [1 for _ in range(amount)]
		bias_candidate = [0 for _ in range(amount)]
		threshold_candidate = [2*np.pi for _ in range(amount)]
		ratio_candidate[i] = 1-r
		bias_candidate[i] = r*np.pi
		threshold_candidate[i] = 1.5*np.pi
		ratio_list.append(ratio_candidate)
		bias_list.append(bias_candidate)
		threshold_list.append(threshold_candidate)
	return ratio_list , bias_list , threshold_list

def compression_forfault(expected_vector, observed_vector, fault_index):
	if not fault_index:
		return np.array(expected_vector), np.array(observed_vector)
	fault_index = deepcopy(fault_index)
	fault_index.sort()
	expected_vector_ = np.zeros(2**len(fault_index))
	observed_vector_ = np.zeros(2**len(fault_index))
	index_list = []
	if (len(fault_index)==2):
		index_list.append(0)
		index_list.append(2**fault_index[0])
		index_list.append(2**fault_index[1])
		index_list.append(2**fault_index[0] + 2**fault_index[1])
		# count = np.zeros(4)
		for n in range(len(expected_vector)):
			if(n & index_list[1] and n & index_list[2]):
				expected_vector_[3] += expected_vector[n]
				observed_vector_[3] += observed_vector[n]
				# count[3] += 1
			elif(n & index_list[2]):
				expected_vector_[2] += expected_vector[n]
				observed_vector_[2] += observed_vector[n]
				# count[2] += 1
			elif(n & index_list[1]):
				expected_vector_[1] += expected_vector[n]
				observed_vector_[1] += observed_vector[n]
				# count[1] += 1
			else:
				expected_vector_[0] += expected_vector[n]
				observed_vector_[0] += observed_vector[n]
				# count[0] += 1
		# print(count)

	else:
		index_list.append(0)
		index_list.append(2**fault_index[0])

		for n in range(len(expected_vector)):
			if(n & index_list[1]):
				expected_vector_[1] += expected_vector[n]
				observed_vector_[1] += observed_vector[n]
			else:
				expected_vector_[0] += expected_vector[n]
				observed_vector_[0] += observed_vector[n]
	# if(np.sum(expected_vector_) != 1 or np.sum(observed_vector_)!=1):
	#     print("EEEEEEEEEEE")
	#     print(expected_vector_, observed_vector_)
	# print(index_list)
	# print(observed_vector_)
	return np.array(expected_vector_), np.array(observed_vector_)

def to_np(vector):
	if(type(vector)==dict or type(vector)==list):
		probability = np.zeros(len(vector))
		for i in vector:
			probability[int(i, 2)] = vector[i]
	else:
		probability = vector

	probability = probability/np.sum(probability)
	return probability

def cal_effect_size(expected_vector, observed_vector):
	delta_square = np.square(expected_vector - observed_vector)
	effectsize = np.sum(delta_square/(expected_vector+INT_MIN))
	effectsize = np.sqrt(effectsize)
	if(effectsize<0.1):
		# print(effectsize)
		effectsize = 0.1
	return effectsize

def compute_repetition(faulty_data, faultfree_data, alpha, beta):
	if faultfree_data.shape != faulty_data.shape:
		print('input shape not consistency')
		return

	degree_freedom = faultfree_data.shape[0]-1
	effect_size = cal_effect_size(faulty_data, faultfree_data)
	if(effect_size>0.8):
		lower_bound_effect_size = 0.8
	else:
		lower_bound_effect_size = effect_size

	repetition = chi2.ppf(alpha, degree_freedom)/(lower_bound_effect_size**2)
	non_centrality = repetition*(effect_size**2)
	chi2_value = chi2.ppf(alpha, degree_freedom)
	non_chi2_value = ncx2.ppf(1-beta, degree_freedom, non_centrality)
	# print(non_chi2_value, chi2_value, repetition, effect_size, non_centrality)
	# if(chi2_value > non_chi2_value):
	# print("original", repetition)
	while(non_chi2_value<chi2_value):
		# print(non_chi2_value, chi2_value, repetition)
		repetition += 1
		non_centrality = repetition*(effect_size**2)
		chi2_value = chi2.ppf(alpha, degree_freedom)
		non_chi2_value = ncx2.ppf(1-beta, degree_freedom, non_centrality)
	# print("after", repetition, non_chi2_value, chi2_value)
	
	# repetition += 5
	# print(repetition, type(repetition))
	boundary = (non_chi2_value*0.3+chi2_value*0.7)
	# print(chi2_value, non_chi2_value, boundary, non_centrality, effect_size)
	if(repetition >= INT_MAX or repetition <= 0):
		# print("repetition error")
		# print(faultfree)
		# print(faulty)
		# print()
		return INT_MAX
	else:
		return ceil(repetition), boundary

def matrix_operation(matrix_list, quantum_state=[], max_size=4):
	matrix = np.eye(max_size)
	for i in range(len(matrix_list)):
		if type(matrix_list[i]) == list:
			for j in range(len(matrix_list[i])-1):
				matrix_list[i][j] = np.kron(matrix_list[i][j] , matrix_list[i][j+1])
			matrix_list[i] = matrix_list[i][-2]
		matrix = np.dot(matrix_list[i], matrix)
	if type(quantum_state) != list:
		return np.dot(matrix, quantum_state)
	else:
		return matrix

def to_probability(probability):
	return np.array(probability*np.conj(probability), dtype=float)

def vector_distance(vector1, vector2):
	return np.sum(np.square(np.abs(np.subtract(to_probability(vector1), to_probability(vector2)))))
