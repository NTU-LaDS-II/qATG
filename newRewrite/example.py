import numpy as np
import qiskit.circuit.library as qGate
from qiskit import QuantumRegister, QuantumCircuit

from qatg import qatg
from qatgFault import qatgFault

# Faults
# single faults
singleFaultList = []
# two faults
class myCNOTFault(qatgFault):
	"""my CNOT fault"""
	def __init__(self, qubit, sixParameters):
		super().__init__(qGate.CXGate, qubit, f"gateType: CNOT, qubits: {qubit}, params: {sixParameters}")

		sub_q = QuantumRegister(2)
		sub_circ = QuantumCircuit(sub_q, name='myCNOTFault')
		sub_circ.u(*sixParameters[0:3], 0)
		sub_circ.cx(0, 1)
		sub_circ.u(*sixParameters[0:3], 1)
		self.faultyGate = sub_circ.to_instruction()

	def getOriginalGateParameters(self):
		return []

	def getFaulty(self, parameters):
		if len(parameters) != 0:
			raise ValueError("No parameters for CNOT!")
		return self.faultyGate

couplingMap = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
twoFaultList = [myCNOTFault(coupling, [0.05] * 6) for coupling in couplingMap]

		
generator = qatg(circuitSize = 5, basisGateSet = [Qgate.RZGate, Qgate.RXGate])
generator.configurationSimulationSetup()
configurationList = generator.getTestConfiguration(singleFaultList, twoFaultList)

for configuration in configurationList:
	print(configuration)