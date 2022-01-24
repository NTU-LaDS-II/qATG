import numpy as np
import qiskit.circuit.library as qGate
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_gate

from qatg import qatg
from qatgFault import qatgFault
from qatgUtil import U3

# Faults
# single faults
singleFaultList = []
# two faults
class myCNOTFault(qatgFault):
	"""my CNOT fault"""
	def __init__(self, qubit, sixParameters):
		super(myCNOTFault, self).__init__(qGate.CXGate, qubit, f"gateType: CNOT, qubits: {qubit}, params: {sixParameters}")

		sub_q = QuantumRegister(2)
		sub_circ = QuantumCircuit(sub_q, name='myCNOTFault')
		sub_circ.u(*sixParameters[0:3], 0)
		sub_circ.cx(0, 1)
		sub_circ.u(*sixParameters[3:6], 1)
		self.faultyGate = circuit_to_gate(sub_circ)

		# fix this for qiskit
		resultArray = np.kron(U3(sixParameters[0:3]), np.eye(2))
		resultArray = np.matmul(qGate.CXGate().to_matrix(), resultArray)
		resultArray = np.matmul(np.kron(np.eye(2), U3(sixParameters[3:6])), resultArray)

		self.faultyGate.__array__ = lambda dtype = None: np.array(resultArray, dtype = dtype)

	def getOriginalGateParameters(self):
		return []

	def getFaultyGate(self, parameters):
		if len(parameters) != 0:
			raise ValueError("No parameters for CNOT!")
		return self.faultyGate

couplingMap = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
twoFaultList = [myCNOTFault(coupling, [0.05] * 6) for coupling in couplingMap]

		
generator = qatg(circuitSize = 5, basisGateSet = [qGate.RZGate, qGate.RXGate])
generator.configurationSimulationSetup()
configurationList = generator.getTestConfiguration(singleFaultList, twoFaultList)

for configuration in configurationList:
	print(configuration)