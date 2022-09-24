import numpy as np
import qiskit.circuit.library as qGate

from qatg import qatg
from qatgFault import qatgFault

class myUFault(qatgFault):
	def __init__(self, params):
		super(myUFault, self).__init__(qGate.UGate, 0, f"gateType: U, qubits: 0, params: {params}")
		self.params = params
	def createOriginalGate(self):
		return qGate.UGate(*self.params)
	def createFaultyGate(self, faultfreeGate):
		return qGate.UGate(faultfreeGate.params[0] - 0.1*np.pi, faultfreeGate.params[1], faultfreeGate.params[2])

generator = qatg(circuitSize = 1, basisGateSet = [qGate.UGate], circuitInitializedStates = {1: [1, 0]}, minRequiredEffectSize = 2)
configurationList = generator.createTestConfiguration([myUFault([np.pi, np.pi, np.pi])])

print(configurationList[0])
configurationList[0].circuit.draw('mpl').show()
input()
