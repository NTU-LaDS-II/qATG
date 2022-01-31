import numpy as np
import qiskit.circuit.library as qGate
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_gate

from qatg import qatg
from qatgFault import qatgFault
from qatgUtil import U3

# Faults
# single faults
class myUFault(qatgFault):
	def __init__(self, qubit, threeParameter, faultList):
		super(myUFault, self).__init__(qGate.UGate, qubit, f"gateType: U, qubits: {qubit}, params: {threeParameter}")
		self.threeParameter = threeParameter
		self.faultList = faultList

	def getOriginalGateParameters(self):
		return self.threeParameter

	def createFaulty(self, parameters):
		if len(parameters) == 0:
			raise ValueError("No parameters for U!")
		return qGate.UGate(*[parameter + fault for parameter, fault in zip(parameters, self.faultList)])

singleFaultList = [myUFault(0, [2*np.pi] * 3, fault) for fault in [[-0.1*np.pi, 0, 0], [0, -0.1*np.pi, 0], [0, 0, -0.1*np.pi]]]

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

	def createFaulty(self, parameters):
		if len(parameters) != 0:
			raise ValueError("No parameters for CNOT!")
		return self.faultyGate

# couplingMap = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
# couplingMap = [[0, 1]]
# twoFaultList = [myCNOTFault(coupling, [0.05 * np.pi] * 6) for coupling in couplingMap]
twoFaultList = []

		
generator = qatg(circuitSize = 5, basisGateSet = [qGate.UGate])
generator.configurationSimulationSetup()
configurationList = generator.createTestConfiguration(singleFaultList, twoFaultList)

for configuration in configurationList:
	print(configuration)