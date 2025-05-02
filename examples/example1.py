import numpy as np
import qiskit.circuit.library as qGate

from qatg import QATG
from qatg import QATGFault

class myUFault(QATGFault):
	def __init__(self, params):
		super(myUFault, self).__init__(qGate.UGate, 0, f"gateType: U, qubits: 0, params: {params}")
		self.params = params
	def createOriginalGate(self):
		return qGate.UGate(*self.params)
	def createFaultyGate(self, faultfreeGate):
		return qGate.UGate(faultfreeGate.params[0] - 0.1*np.pi, faultfreeGate.params[1], faultfreeGate.params[2]) # bias fault on theta

generator = QATG(circuitSize = 1, basisSingleQubitGateSet = [qGate.UGate], circuitInitializedStates = {1: [1, 0]}, minRequiredStateFidelity = 0.1)
configurationList = generator.createTestConfiguration([myUFault([np.pi, np.pi, np.pi])])

for configuration in configurationList:
    print(configuration)
    print(configuration.circuit)
input()
