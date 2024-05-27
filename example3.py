import numpy as np
import qiskit.circuit.library as qGate
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from qatg import QATG
from qatg import QATGFault

class myCNOTFault(QATGFault):
	def __init__(self):
		super(myCNOTFault, self).__init__(qGate.CXGate, [0, 1], f"gateType: CX, qubits: 0-1")
	def createOriginalGate(self):
		return qGate.CXGate()
	def createFaultyGate(self, faultfreeGate):
		if not isinstance(faultfreeGate, qGate.CXGate):
			raise TypeError("what is this faultfreeGate")
		matrix = qGate.CXGate().to_matrix()
		UF = qGate.UGate(0.05*np.pi, 0.05*np.pi, 0.05*np.pi)
		matrix = np.matmul(np.kron(np.eye(2), UF), matrix)
		matrix = np.matmul(matrix, np.kron(UF, np.eye(2)))
		# UnitaryGate(matrix).draw()
		return UnitaryGate(matrix)

# qc = QuantumCircuit(2)
# faulty_cx_gate = myCNOTFault().createFaultyGate(qGate.CXGate())
# qc.append(faulty_cx_gate, [0, 1])
# qc.draw('mpl')
# print(qc)
generator = QATG(circuitSize = 2, basisSingleQubitGateSet = [qGate.UGate], circuitInitializedStates = {2: [1, 0, 0, 0]}, minRequiredStateFidelity = 0.1)
configurationList = generator.createTestConfiguration([myCNOTFault()])


for configuration in configurationList:
    print(configuration)
    configuration.circuit.draw('mpl')
input()
