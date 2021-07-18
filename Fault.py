# import qiskit
import qiskit.circuit.library as Qgate
from numpy import pi
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
class Variation_fault(Fault):
	def __init__(self, gate_type, index, ratio = None, bias = None):
		require_length = gate_type.__init__.__code__.co_argcount - len(gate_type.__init__.__defaults__) - 1
		if not ratio and not bias:
			print("Variation Fault: init error with two none value")
			exit()
		elif not ratio:
			# check bias
			if len(bias) != require_length:
				print("Variation Fault: init error with wrong bias length")
				exit()
			ratio = [1] * len(bias)
		elif not bias:
			# check ratio
			if len(ratio) != require_length:
				print("Variation Fault: init error with wrong ratio length")
				exit()
			bias = [0] * len(ratio)
		else:
			# check ratio
			if len(ratio) != require_length:
				print("Variation Fault: init error with wrong ratio length")
				exit()
			# check bias
			if len(bias) != require_length:
				print("Variation Fault: init error with wrong bias length")
				exit()
		description = gate_type.__name__
		description += ' variation fault at '+str(index[0])+', ratio parameter:'
		for i in ratio:
			description += ' '+str(i)
		description += ', bias parameter:'
		for i in bias:
			description += ' '+str(i)
		super().__init__(index, gate_type, description)
		self.ratio = ratio
		self.bias = bias
		self.description = description
		self.gate_type = gate_type
		# 彈性空間，由ratio的len決定
		# gate_info is class QuantumGate
	def get_faulty_gate(self , gate_info):
		faulty_gate = deepcopy(gate_info)
		for i in range(len(self.ratio)):
			faulty_gate.gate.params[i] = self.ratio[i]*faulty_gate.gate.params[i] + self.bias[i]
		return faulty_gate


############ U low pass ############
class Threshold_lopa(Fault):
	def __init__(self, gate_type, index, threshold = None):
		require_length = gate_type.__init__.__code__.co_argcount - len(gate_type.__init__.__defaults__) - 1
		if not threshold:
			threshold = [2*pi] * require_length
		else:
			# check threshold
			if len(threshold) != require_length:
				print("Threshold lowpass error with wrong threshold length")
				exit()
		description = gate_type.__name__
		description += ' threshold fault at '+str(index[0])+', threshold parameter:'
		for i in threshold:
			description += ' '+str(i)

		super().__init__(index, gate_type, description)
		self.threshold = threshold
		self.description = description
		self.gate_type = gate_type
	def get_faulty_gate(self, gate_info):
		# gate_info is class QuantumGate
		faulty_gate = deepcopy(gate_info)
		for i in range(len(self.threshold)):
			faulty_gate.gate.params[i] = self.threshold[i] if faulty_gate.gate.params[i] > self.threshold[i] else faulty_gate.gate.params[i]
		return faulty_gate
