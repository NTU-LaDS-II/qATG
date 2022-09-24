import numpy as np
from copy import deepcopy
from qiskit import transpile
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
import qiskit.circuit.library as qGate

from qatgFault import qatgFault
from qatgUtil import *
from qatgConfiguration import qatgConfiguration

class qatg():
	"""qatg main class"""
	def __init__(self, \
			circuitSize: int, basisGateSet: list[qGate], circuitInitializedStates: dict, \
			quantumRegisterName: str = 'q', classicalRegisterName: str = 'c', \
			gridSlice: int = 11, gradientDescentMaxIteration: int = 800, \
			gradientDescentStep: float = 0.5, gradientMeasureStep: float = 0.0001, gradientDeltaThreshold: float = 1e-8, \
			maxTestTemplateSize: int = 50, minRequiredEffectSize: float = 3, \
			oneQubitErrorProb = 0.001, twoQubitErrorProb = 0.1, \
			zeroReadoutErrorProb = [0.985, 0.015], oneReadoutErrorProb = [0.015, 0.985], \
			targetAlpha: float = 0.99, targetBeta: float = 0.999, \
			simulationShots: int = 200000, testSampleTime: int = 10000):
		# init + config setup
		if not isinstance(circuitSize, int):
			raise TypeError('circuitSize must be int')
		if circuitSize <= 0:
			raise ValueError('circuitSize must be positive')
		self.circuitSize = circuitSize
		
		# list[qGate]
		self.basisGateSet = basisGateSet
		self.basisGateSetString = [gate.__name__[:-4].lower() for gate in self.basisGateSet]

		# dict{0: [], 1: [1, 0], 2: [1, 0, 0, 0]} etc.
		self.circuitInitializedStates = circuitInitializedStates

		self.quantumRegisterName = quantumRegisterName
		self.classicalRegisterName = classicalRegisterName
		self.circuitSetup = {}
		self.circuitSetup['circuitSize'] = self.circuitSize
		self.circuitSetup['basisGateSet'] = self.basisGateSet
		self.circuitSetup['basisGateSetString'] = self.basisGateSetString
		self.circuitSetup['circuitInitializedStates'] = self.circuitInitializedStates
		self.circuitSetup['quantumRegisterName'] = self.quantumRegisterName
		self.circuitSetup['classicalRegisterName'] = self.classicalRegisterName

		self.gridSlice = gridSlice
		self.gradientDescentMaxIteration = gradientDescentMaxIteration
		self.gradientDescentStep = gradientDescentStep # suggest not too big
		self.gradientMeasureStep = gradientMeasureStep # suggest not too big
		self.gradientDeltaThreshold = gradientDeltaThreshold
		self.maxTestTemplateSize = maxTestTemplateSize
		self.minRequiredEffectSize = minRequiredEffectSize
		
		q = QuantumCircuit(1)
		self.qiskitParameterTheta = Parameter('theta')
		self.qiskitParameterPhi = Parameter('phi')
		self.qiskitParameterLambda = Parameter('lam')
		q.u(self.qiskitParameterTheta, self.qiskitParameterPhi, self.qiskitParameterLambda, 0)
		try:
			self.effectiveUGateCircuit = transpile(q, basis_gates = self.basisGateSetString, optimization_level = 3)
		except Exception as e:
			raise e

		self.simulationSetup = {}
		self.simulationSetup['oneQubitErrorProb'] = oneQubitErrorProb
		self.simulationSetup['twoQubitErrorProb'] = twoQubitErrorProb
		self.simulationSetup['zeroReadoutErrorProb'] = zeroReadoutErrorProb
		self.simulationSetup['oneReadoutErrorProb'] = oneReadoutErrorProb
		self.simulationSetup['targetAlpha'] = targetAlpha
		self.simulationSetup['targetBeta'] = targetBeta
		self.simulationSetup['simulationShots'] = simulationShots
		self.simulationSetup['testSampleTime'] = testSampleTime

		return

	def getTestConfiguration(self, faultList, simulateConfiguration: bool = True):
		# simulateConfiguration: True, simulate the configuration and generate test repetition
		# false: don't simulate and repetition = NaN

		configurationList = [qatgConfiguration(self.circuitSetup, self.simulationSetup, fault) for fault in faultList]

		for k in range(len(faultList)):
			fault = faultList[k]
			if not issubclass(type(fault), qatgFault):
				raise TypeError(f"{fault} should be subclass of qatgFault")
			initialState = self.circuitInitializedStates[len(fault.getQubits())]
			template, effectSize = self.generateTestTemplate(faultObject = fault, initialState = initialState)
			configurationList[k].setTemplate(template, effectSize)
			if simulateConfiguration:
				configurationList[k].simulate()

		return configurationList

	def generateTestTemplate(self, faultObject, initialState):
		# list of "qGates"
		# a member is either a gate (for all qubits) or a list of gate (for each qubit)
		# one activation gate, one original/faulty gate
		templateGateList = []

		faultyQuantumState = deepcopy(initialState)
		faultfreeQuantumState = deepcopy(initialState)

		for _ in range(self.maxTestTemplateSize):
			newElement, faultyQuantumState, faultfreeQuantumState = self.findNewElement(faultObject, faultyQuantumState, faultfreeQuantumState)
			templateGateList += newElement
			effectSize = calEffectSize(faultyQuantumState, faultfreeQuantumState)
			print(effectSize)
			if effectSize > self.minRequiredEffectSize:
				break

		return templateGateList, effectSize

	def findNewElement(self, faultObject, faultyQuantumState, faultfreeQuantumState):
		# find new element
		originalGate = faultObject.createOriginalGate()
		faultyGateMatrix = faultObject.createFaultyGate(originalGate).to_matrix()
		originalGateMatrix = originalGate.to_matrix()

		def parameterSet2ActivationMatrix(parameterSet):
			faultfreeGateMatrixList = [qatgU3(parameter) for parameter in parameterSet] # vertical
			# faulty: u->transpile->gate set, and then insert fault
			faultyGateMatrixList = [] # should be vertical
			for parameter in parameterSet:
				effectiveCktGateList = self.U2GateSetsTranspile(parameter)
				faultyEffectiveGateMatrix = np.eye(*effectiveCktGateList[0].to_matrix().shape)
				for gate in effectiveCktGateList:
					if faultObject.isSameGateType(gate):
						faultyEffectiveGateMatrix = np.matmul(faultObject.createFaultyGate(gate).to_matrix(), faultyEffectiveGateMatrix)
					else:
						faultyEffectiveGateMatrix = np.matmul(gate.to_matrix(), faultyEffectiveGateMatrix)
				faultyGateMatrixList.append(faultyEffectiveGateMatrix)
			# kron all together
			faultfreeActivation = np.array([1])
			faultyActivation = np.array([1])
			for k in range(len(faultfreeGateMatrixList)):
				faultfreeActivation = np.kron(faultfreeGateMatrixList[k], faultfreeActivation)
				faultyActivation = np.kron(faultyGateMatrixList[k], faultyActivation)
			
			return faultfreeActivation, faultyActivation

		def score(parameterSet):
			# parameterSet: [list of first U, list of second U, ...]
			faultfreeActivation, faultyActivation = parameterSet2ActivationMatrix(parameterSet)
			return vectorDistance(
				np.dot(np.matmul(originalGateMatrix, faultfreeActivation), faultfreeQuantumState), 
				np.dot(np.matmul(faultyGateMatrix, faultyActivation), faultyQuantumState))

		# 1. find best parameters
		# grid search
		qubitSize = len(faultObject.getQubits())
		optimalParameterSet = [[0, 0, 0] for _ in range(qubitSize)]
		for k in range(qubitSize):
			results = []
			for theta in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
				for phi in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
					for lam in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
						optimalParameterSet[k] = [theta, phi, lam]
						results.append([[theta, phi, lam], score(optimalParameterSet)])
			optimalParameterSet[k] = max(results, key = lambda x: x[1])[0]
		print(f"Before: {score(optimalParameterSet)}")

		# gradient
		for k in range(self.gradientDescentMaxIteration):
			deltaParameter = 0
			currentOptimalScore = score(optimalParameterSet)
			tempParameterSet = deepcopy(optimalParameterSet)
			maxDelta = 0
			for m in range(qubitSize):
				for n in range(3):
					# find gradient
					tempParameterSet[m][n] += self.gradientMeasureStep
					currentTempScore = score(tempParameterSet)
					deltaParameter = (currentTempScore - currentOptimalScore) / self.gradientMeasureStep
					tempParameterSet[m][n] -= self.gradientMeasureStep
					optimalParameterSet[m][n] += self.gradientDescentStep * deltaParameter # + since concave?
					maxDelta = abs(deltaParameter) if abs(deltaParameter) > maxDelta else maxDelta
			print(score(optimalParameterSet))
			if maxDelta < self.gradientDeltaThreshold:
				break
		print(f"After : {score(optimalParameterSet)}")
		print(k)

		# 2. transpile, append
		# construct faultfree template element
		newElement = [self.U2GateSetsTranspile(parameter) for parameter in optimalParameterSet]
		# iterate by qubit
		newElement = [list(x) for x in zip(*newElement)] # transpose
		# iterate by order
		newElement.append(originalGate)

		faultfreeActivation, faultyActivation = parameterSet2ActivationMatrix(optimalParameterSet)
		faultfreeQuantumState = np.dot(np.matmul(originalGateMatrix, faultfreeActivation), faultfreeQuantumState)
		faultyQuantumState = np.dot(np.matmul(faultyGateMatrix, faultyActivation), faultyQuantumState)

		return newElement, faultyQuantumState, faultfreeQuantumState

	def U2GateSetsTranspile(self, UParameters):
		# to gate list directly
		resultCircuit = self.effectiveUGateCircuit.bind_parameters({ \
			self.qiskitParameterTheta: UParameters[0], \
			self.qiskitParameterPhi: UParameters[1], \
			self.qiskitParameterLambda: UParameters[2]})
		for cktInstruction in resultCircuit.data:
			cktInstruction.operation.params = [qatgWrapToPi(float(param)) for param in cktInstruction.operation.params]
		return [cktInstruction.operation for cktInstruction in resultCircuit.data]
		# return [gate for gate, _, _ in resultCircuit.data] # old version of qiskit
		# potential bug: parameters might have something such as "3pi"
		# how to restrict range?

