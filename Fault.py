import qiskit
import qiskit.circuit.library as Qgate
import numpy as np
from QuantumGate import *
from copy import deepcopy

class Fault():
	def __init__(self, index, gate_type, description):
		self.index = index if type(index)==list else [index]
		self.gate_type = gate_type
		self.description = description

	def __str__(self):
		return self.description

	def get_faulty_gate(self, gate_type):
		f = deepcopy(gate_type)
		#override
		return f

	
############ CNOT ratio & bias ############
class CNOT_variation_fault(Fault):
	def __init__(self, index, value=[0, 0, 0, 0, 0, 0]):
		if(type(index)!=list or len(index)!=2):
			print("type of index should be list or size of index should be 2")
			exit()
		if(len(value)!=6):
			print("size of value should be 6")
			exit() 
		description = 'CNOT variation fault at control qubit '+str(index[0])+' and target qubit '+str(index[1])+', parameter:'
		for i in value:
			description += ' '+str(i)
		super().__init__(index, Qgate.CXGate, description)
		self.value = value

	def get_faulty_gate(self, gate_info):
		gate_list = []
		gate_list.append(QuantumGate(Qgate.U3Gate(self.value[0], self.value[1], self.value[2]) , gate_info.pos , []))
		gate_list.append(deepcopy(gate_info))
		gate_list.append(QuantumGate(Qgate.U3Gate(self.value[3], self.value[4], self.value[5]) , gate_info.pos , []))
		#一維陣列 element is QuantumGate
		return gate_list

############ U ratio & bias ############      
#ratio and bias are list and their length depend on which type of gate it is
class U_variation_fault(Fault):
	def __init__(self, index, ratio=[1, 1, 1], bias=[0, 0, 0]):
		#if(len(ratio)!=3):
		#    print("size of ratio should be 3")
		#    exit() 
		#if(len(bias)!=3):
		#   print("size of bias should be 3")
		#    exit() 
		description = 'U3 variation fault at '+str(index[0])+', ratio parameter:'
		for i in ratio:
		    description += ' '+str(i)
		description += ', bias parameter:'
		for i in bias:
		    description += ' '+str(i)
		super().__init__(index, Qgate.U3Gate, description)
		self.ratio = ratio
		self.bias = bias
		self.description = description
		#彈性空間，由ratio的len決定
		#gate_info is class QuantumGate
	def get_faulty_gate(self , gate_info):
		faulty_gate = deepcopy(gate_info)
		for i in range(len(self.ratio)):
			faulty_gate.gate.params[i] = self.ratio[i]*faulty_gate.gate.params[i] + self.bias[i]
		return faulty_gate


############ U low pass ############
class U_threshold_lopa(Fault):
	def __init__(self, index, threshold=[np.pi*2, np.pi*2, np.pi*2]):

		#if(len(threshold)!=3):
		#    print("size of threshold should be 3")
		#    exit() 
		description = 'U3 threshold fault at '+str(index[0])+', threshold parameter:'
		for i in threshold:
		    description += ' '+str(i)

		super().__init__(index, Qgate.U3Gate, description)
		self.threshold = threshold
		
	def get_faulty_gate(self, gate_info):
		#gate_info is class QuantumGate
		faulty_gate = deepcopy(gate_info)
		for i in range(len(self.threshold)):
			faulty_gate.gate.params[i] = self.threshold[i] if faulty_gate.gate.params[i] > self.threshold[i] else faulty_gate.gate.params[i]
		return faulty_gate