import numpy as np
import qiskit.circuit.library as qGate
from qiskit import qasm3

from qatg import QATG
from qatg import QATGFault

class myRXFault(QATGFault):
	def __init__(self, param):
		super(myRXFault, self).__init__(qGate.RXGate, 0, f"gateType: RX, qubits: 0, param: {param}")
		self.param = param
	def createOriginalGate(self):
		return qGate.RXGate(self.param)
	def createFaultyGate(self, faultfreeGate):
		return qGate.RXGate(faultfreeGate.params[0] - 0.1*np.pi) # bias fault
	
class myRZFault(QATGFault):
	def __init__(self, param):
		super(myRZFault, self).__init__(qGate.RZGate, 0, f"gateType: RZ, qubits: 0, param: {param}")
		self.param = param
	def createOriginalGate(self):
		return qGate.RZGate(self.param)
	def createFaultyGate(self, faultfreeGate):
		return qGate.RZGate(faultfreeGate.params[0] - 0.1*np.pi) # bias fault

generator = QATG(circuitSize = 1, basisSingleQubitGateSet = [qGate.RXGate, qGate.RZGate], circuitInitializedStates = {1: [1, 0]}, minRequiredStateFidelity = 0.1)
configurationList = generator.createTestConfiguration([myRXFault(np.pi), myRZFault(np.pi)])

for idx, configuration in enumerate(configurationList):
    print(configuration)
    with open(f'my_fault{idx+1}.qasm', 'w') as f:
    	qasm3.dump(configuration.circuit, f)
input()
