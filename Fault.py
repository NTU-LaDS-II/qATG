import qiskit
import qiskit.circuit.library as Qgate
import numpy as np
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

    def is_faulty_type(self, gate_info):
        if type(gate_info[0]) !=  self.gate_type:
            return False
        else:
            for i in range(len(gate_info[1])):
                if gate_info[1][i].index != self.index[i]:
                    return False
        return True

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
        gate_list.append((Qgate.U3Gate(self.value[0], self.value[1], self.value[2]),[gate_info[1][0]], []))
        gate_list.append(deepcopy(gate_info))
        gate_list.append((Qgate.U3Gate(self.value[3], self.value[4], self.value[5]),[gate_info[1][1]], []))

        return gate_list

############ U3 ratio & bias ############
class U3_variation_fault(Fault):
    def __init__(self, index, ratio=[1, 1, 1], bias=[0, 0, 0]):
        if(len(ratio)!=3):
            print("size of ratio should be 3")
            exit() 
        if(len(bias)!=3):
            print("size of bias should be 3")
            exit() 
        description = 'U3 variation fault at '+str(index[0])+', ratio parameter:'
        for i in ratio:
            description += ' '+str(i)
        description += ', bias parameter:'
        for i in bias:
            description += ' '+str(i)
        super().__init__(index, Qgate.U3Gate, description)
        self.ratio = ratio
        self.bias = bias
        
    def get_faulty_gate(self, gate_info):
        faulty_gate = deepcopy(gate_info)
        faulty_gate[0].params[0] = self.ratio[0]*faulty_gate[0].params[0] + self.bias[0]
        faulty_gate[0].params[1] = self.ratio[1]*faulty_gate[0].params[1] + self.bias[1]
        faulty_gate[0].params[2] = self.ratio[2]*faulty_gate[0].params[2] + self.bias[2]
        return [faulty_gate]


############ U3 low pass ############
class U3_threshold_lopa(Fault):
    def __init__(self, index, threshold=[np.pi*2, np.pi*2, np.pi*2]):

        if(len(threshold)!=3):
            print("size of threshold should be 3")
            exit() 
        description = 'U3 threshold fault at '+str(index[0])+', threshold parameter:'
        for i in threshold:
            description += ' '+str(i)

        super().__init__(index, Qgate.U3Gate, description)
        self.threshold = threshold
        
    def get_faulty_gate(self, gate_info):
        faulty_gate = deepcopy(gate_info)
        faulty_gate[0].params[0] = self.threshold[0] if faulty_gate[0].params[0] > self.threshold[0] else faulty_gate[0].params[0]
        faulty_gate[0].params[1] = self.threshold[1] if faulty_gate[0].params[1] > self.threshold[1] else faulty_gate[0].params[1]
        faulty_gate[0].params[2] = self.threshold[2] if faulty_gate[0].params[2] > self.threshold[2] else faulty_gate[0].params[2]
        return [faulty_gate]

############ U2 ratio & bias ############
class U2_variation_fault(Fault):
    def __init__(self, index, ratio=[1, 1], bias=[0, 0]):
        if(len(ratio)!=2):
            print("size of ratio should be 2")
            exit() 
        if(len(bias)!=2):
            print("size of bias should be 2")
            exit()
        description = 'U2 variation fault at '+str(index[0])+', ratio parameter:'
        for i in ratio:
            description += ' '+str(i)
        description += ', bias parameter:'
        for i in bias:
            description += ' '+str(i) 
        super().__init__(index, Qgate.U2Gate, description)
        self.ratio = ratio
        self.bias = bias
        
    def get_faulty_gate(self, gate_info):
        faulty_gate = deepcopy(gate_info)
        faulty_gate[0].params[0] = self.ratio[0]*faulty_gate[0].params[0] + self.bias[0]
        faulty_gate[0].params[1] = self.ratio[1]*faulty_gate[0].params[1] + self.bias[1]
        return [faulty_gate]

############ U2 low pass ############
class U2_threshold_lopa(Fault):
    def __init__(self, index, threshold=[np.pi*2, np.pi*2]):

        if(len(threshold)!=2):
            print("size of threshold should be 2")
            exit()
        description = 'U2 threshold fault at '+str(index[0])+', threshold parameter:'
        for i in threshold:
            description += ' '+str(i)
        super().__init__(index, Qgate.U2Gate, description)
        self.threshold = threshold
        
    def get_faulty_gate(self, gate_info):
        faulty_gate = deepcopy(gate_info)
        faulty_gate[0].params[0] = self.threshold[0] if faulty_gate[0].params[0] > self.threshold[0] else faulty_gate[0].params[0]
        faulty_gate[0].params[1] = self.threshold[1] if faulty_gate[0].params[1] > self.threshold[1] else faulty_gate[0].params[1]
        return [faulty_gate]

############ U1 ratio & bias ############
class U1_variation_fault(Fault):
    def __init__(self, index, ratio=[1], bias=[0]):
        if(len(ratio)!=1):
            print("size of ratio should be 1")
            exit()
        if(len(bias)!=1):
            print("size of bias should be 1")
            exit()
        description = 'U1 variation fault at '+str(index[0])+', ratio parameter:'
        for i in ratio:
            description += ' '+str(i)
        description += ', bias parameter:'
        for i in bias:
            description += ' '+str(i)
        super().__init__(index, Qgate.U1Gate, description)
        self.ratio = ratio
        self.bias = bias
        
    def get_faulty_gate(self, gate_info):
        faulty_gate = deepcopy(gate_info)
        faulty_gate[0].params[0] = self.ratio[0]*faulty_gate[0].params[0] + self.bias[0]
        return [faulty_gate]

############ U1 low pass ############
class U1_threshold_lopa(Fault):
    def __init__(self, index, threshold=[np.pi*4]):
        if(len(threshold)!=1):
            print("size of threshold should be 1")
            exit()
        description = 'U1 threshold fault at '+str(index[0])+', threshold parameter:'
        for i in threshold:
            description += ' '+str(i)
        super().__init__(index, Qgate.U1Gate, description)
        self.threshold = threshold
        
    def get_faulty_gate(self, gate_info):
        faulty_gate = deepcopy(gate_info)
        faulty_gate[0].params[0] = self.threshold[0] if faulty_gate[0].params[0] > self.threshold[0] else faulty_gate[0].params[0]
        return [faulty_gate]