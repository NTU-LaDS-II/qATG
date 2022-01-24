import numpy as np
from copy import deepcopy
from qiskit import Aer
from qiskit import execute
from qiskit import transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
import qiskit.circuit.library as qGate

from qatgFault import qatgFault

class qatg():
	def __init__(self, circuitSize: int = None, qubitIds: list[int] = None, basisGateSet: list[qGate], couplingMap: list[list], \
			quantumRegisterName: str = 'q', classicalRegisterName: str = 'c', \
			targetAlpha: float = 0.99, targetBeta: float = 0.999, \
			gridSlice: int = 21, \
			gradientDescentSearchTime: int = 800, gradientDescentStep: float = 0.01, \
			maxTestTemplateSize: int = 50, minRequiredEffectSize: float = 3, \
			testSampleTime: int = 10000):
		if not circuitSize and not qubitIds:
			raise ValueError('unknown circuitSize and qubitIds')

		if qubitIds:
			# monkey
			if circuitSize:
				if len(qubitIds) != circuitSize:
					raise ValueError('unmatched circuitSize and qubitIds size')
			self.qubitIds = qubitIds
		elif circuitSize:
			if not isinstance(circuitSize, int):
				raise TypeError('circuitSize must be int')
			if circuitSize <= 0:
				raise ValueError('circuitSize must be positive')
			self.qubitIds = range(circuitSize)
		
		# list[qGate]
		self.basisGateSet = basisGateSet
		self.couplingMap = couplingMap
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
		except Exception as e:
			raise e
		return

	def getTestConfiguration(self, singleFaultList, twoFaultList, \
			singleInitialState: np.array = np.array([1, 0]), twoInitialState: array = np.array([1, 0, 0, 0]), simulateConfiguration: bool = True):
		# simulateConfiguration: True, simulate the configuration and generate test repetition
		# false: don't simulate and repetition = NaN

		# singleFaultList: a list of singleFault
		# singleFault: a class object inherit class Fault
		# gateType: faultObject.getGateType()
		# original gate parameters: faultObject.getOriginalGateParameters(target)
		# faulty: faultObject.getFaulty(faultfreeParameters, target)

		configurationList = []

		for singleFault in singleFaultList:
			if not issubclass(singleFault, qatgFault):
				raise TypeError(f"{singleFault} should be subclass of qatgFault")
			for qubit in self.qubitIds:
				template = self.generateTestTemplate(faultObject = singleFault, target = qubit, initialState = singleInitialState, findActivationGate = singleActivationGate)
				

		for twoFault in twoFaultList:
			if not issubclass(singleFault, qatgFault):
				raise TypeError(f"{twoFault} should be subclass of qatgFault")
			for couple in couplingMap:
				template = self.generateTestTemplate(faultObject = twoFault, target = couple, initialState = twoInitialState, findActivationGate = twoActivationGate)

		pass

	def getTestTemplate(self, faultObject, target, initialState, findActivationGate):
		templateGateList = [] # list of qGate

		faultyQuantumState = deepcopy(intialState)
		faultfreeQuantumState = deepcopy(intialState)

		for element in range(self.maxTestTemplateSize):
			newElement, faultyQuantumState, faultfreeQuantumState = findActivationGate(faultObject = faultObject, target = target, faultyQuantumState = faultyQuantumState, faultfreeQuantumState = faultfreeQuantumState)
			# newElement: list[np.array(gate)]
			templateGateList = np.concatenate([templateGateList, newElement])
			effectSize = calEffectSize(faultyQuantumState, faultfreeQuantumState)
			if effectsize > self.minRequiredEffectSize:
				break

		# print?

		return templateGateList

	def singleActivationGate(self, faultObject, target, faultyQuantumState, faultfreeQuantumState):
		newElement = [] # list of qGate

		# optimize activation gate
		originalGateParameters = faultObject.getOriginalGateParameters(target) # list of parameters
		originalGateMatrix = faultObject.getOriginalGate(target).to_matrix()
		faultyGateMatrix = faultObject.getFaulty(originalGateParameters, target).to_matrix() # np.array(gate)
		# TODO
		# grid search
		# remember to compare gate.type and faultObject.getGateType()
		# gate.to_matrix() -> 2x2
		# gate.params -> list[Parameter]
		# gradient descent
		# U to gateSet

		# faultObject.getOriginalGate(target): np.array(gate)
		newElement.append(faultObject.getOriginalGate(target))
		return newElement

	def twoActivationGate(self, faultObject, target, faultyQuantumState, faultfreeQuantumState):
		# TODO
		pass

	@staticmethod
	def calEffectSize(faultyQuantumState, faultfreeQuantumState):
		# TODO
		pass

	@staticmethod
	def U2GateSetsTranspile(UParameters):
		# to gate list directly
		resultCircuit = self.effectiveUGateCircuit.bind_parameters({self.qiskitParameterTheta: UParameters[0], \
			self.qiskitParameterPhi: UParameters[1], self.qiskitParameterLambda: UPartParameters[2]})
		return [gate for gate, _, _ in resultCircuit.data] # a list of qGate