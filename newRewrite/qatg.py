import numpy as np
from copy import deepcopy
from qiskit import Aer
from qiskit import execute
from qiskit import transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
import qiskit.circuit.library as qGate

class qatg():
	def __init__(self, circuitSize: int, basisGateSet: list[qGate], \
			quantumRegisterName: str = 'q', classicalRegisterName: str = 'c', \
			targetAlpha: float = 0.99, targetBeta: float = 0.999, \
			gridSlice: int = 21, \
			gradientDescentSearchTime: int = 800, gradientDescentStep: float = 0.01, \
			maxTestTemplateSize: int = 50, minRequiredEffectSize: float = 3, \
			testSampleTime: int = 10000):
		if not isinstance(circuitSize, int):
			raise TypeError('circuitSize must be int')
		if circuitSize <= 0:
			raise ValueError('circuitSize must be positive')
		self.circuitSize = circuitSize

		# list[qGate], list['str' with available gates]
		# np.array[qGate], np.array['str' with available gates]
		# list[self def gates], np.array[self def gates]
		# self def gates must be 2x2, ...
		self.basisGateSet = basisGateSet
		self.quantumRegisterName = quantumRegisterName
		self.classicalRegisterName = classicalRegisterName
		self.targetAlpha = targetAlpha
		self.targetBeta = targetBeta
		self.gridSlice = gridSlice
		self.gradientDescentSearchTime = gradientDescent
		self.gradientDescentStep = gradientDescentStep
		self.maxTestTemplateSize = maxTestTemplateSize
		self.minRequiredEffectSize = minRequiredEffectSize
		self.testSampleTime = testSampleTime

		self.quantumRegister = QuantumRegister(self.circuitSize, self.quantumRegisterName)
		self.classicalRegister = ClassicalRegister(self.circuitSize, self.classicalRegisterName)
		self.backend = Aer.get_backend('qasm_simulator')
		self.basisGateSetString = [gate.__name__[:-4].lower() for gate in self.basisGateSet]
		q = QuantumCircuit(1)
		self.qiskitParameterTheta = Parameter('theta')
		self.qiskitParameterPhi = Parameter('phi')
		self.qiskitParameterLambda = Parameter('lam')
		q.u(self.qiskitParameterTheta, self.qiskitParameterPhi, self.qiskitParameterLambda, 0)
		try:
			self.effectiveUGateCircuit = transpile(q, basis_gates = self.basisGateSetString, optimization_level = 3)
		except 

		return

	def getTestConfiguration(self, singleFaults, twoFaults, initialState: np.array = np.array([1, 0]), simulateConfiguration: bool = True):
		# single faults are: each element is a single fault
		# a single fault is a function(qbNum, gateType)
		# faultfree: gateType, might be qGate?
		# faulty: singleFault(qbNum, gateType), return qGate or 2x2 numpy matrix?
		# simulateConfiguration: True, simulate the configuration and generate test repetition
		# false: don't simulate and repetition = NaN

		configurationList = []

		for singleFault in singleFaults:
			for basisGate in basisGateSet:
				for qubit in range(self.circuitSize):
					template = self.generateTestTemplate(faultfree = basisGate, faulty = singleFault(qubit, basisGate), \
						initialState = initialState)
					configuration = self.buildSingleConfiguration
		pass
