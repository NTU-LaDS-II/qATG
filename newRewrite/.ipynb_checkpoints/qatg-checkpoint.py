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
	def __init__(self, circuitSize: int, basisGateSet: list[qGate], \
			quantumRegisterName: str = 'q', classicalRegisterName: str = 'c', \
			gridSlice: int = 21, gradientDescentSearchTime: int = 800, gradientDescentStep: float = 0.01, \
			maxTestTemplateSize: int = 50, minRequiredEffectSize: float = 3):
		if not isinstance(circuitSize, int):
			raise TypeError('circuitSize must be int')
		if circuitSize <= 0:
			raise ValueError('circuitSize must be positive')
		self.circuitSize = circuitSize
		
		# list[qGate]
		self.basisGateSet = basisGateSet
		# self.couplingMap = couplingMap # not used
		self.quantumRegisterName = quantumRegisterName
		self.classicalRegisterName = classicalRegisterName
		self.gridSlice = gridSlice
		self.gradientDescentSearchTime = gradientDescentSearchTime
		self.gradientDescentStep = gradientDescentStep
		self.maxTestTemplateSize = maxTestTemplateSize
		self.minRequiredEffectSize = minRequiredEffectSize

		self.quantumRegister = QuantumRegister(self.circuitSize, self.quantumRegisterName)
		self.classicalRegister = ClassicalRegister(self.circuitSize, self.classicalRegisterName)
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

		self.simulationSetup = None
		return

	def configurationSimulationSetup(self, oneQubitErrorProb = 0.001, twoQubitErrorProb = 0.1, \
			zeroReadoutErrorProb = [0.985, 0.015], oneReadoutErrorProb = [0.015, 0.985], \
			targetAlpha: float = 0.99, targetBeta: float = 0.999, \
			simulationShots: int = 200000, testSampleTime: int = 10000):
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

	def getTestConfiguration(self, singleFaultList, twoFaultList, \
			singleInitialState: np.array = np.array([1, 0]), twoInitialState: np.array = np.array([1, 0, 0, 0]), simulateConfiguration: bool = True):
		# simulateConfiguration: True, simulate the configuration and generate test repetition
		# false: don't simulate and repetition = NaN

		if simulateConfiguration and not self.simulationSetup:
			raise NotImplementedError("pls call configurationSimulationSetup() first")

		# singleFaultList: a list of singleFault
		# singleFault: a class object inherit class Fault
		# gateType: faultObject.getGateType()
		# original gate parameters: faultObject.getOriginalGateParameters()
		# faulty: faultObject.getFaulty(faultfreeParameters)

		configurationList = []

		for singleFault in singleFaultList:
			if not issubclass(type(singleFault), qatgFault):
				raise TypeError(f"{singleFault} should be subclass of qatgFault")
			template = self.generateTestTemplate(faultObject = singleFault, initialState = singleInitialState, findActivationGate = self.findSingleElement)
			configuration = self.buildSingleConfiguration(template, singleFault)
			if simulateConfiguration:
				configuration.simulate()
			configurationList.append(configuration)
			print(configuration)

		for twoFault in twoFaultList:
			if not issubclass(type(twoFault), qatgFault):
				raise TypeError(f"{twoFault} should be subclass of qatgFault")
			template = self.generateTestTemplate(faultObject = twoFault, initialState = twoInitialState, findActivationGate = self.findTwoElement)
			configuration = self.buildTwoConfiguration(template, twoFault)
			if simulateConfiguration:
				configuration.simulate()
			configurationList.append(configuration)
		
		return configurationList

	def buildSingleConfiguration(self, template, singleFault):
		qcFaultFree = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		qbIndex = singleFault.getQubit()[0]
		for gate in template:
			# for qbIndex in range(self.circuitSize):
			qcFaultFree.append(gate, [qbIndex])
			qcFaultFree.append(qGate.Barrier(qbIndex))
		qcFaultFree.measure(self.quantumRegister, self.classicalRegister)

		qcFaulty = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		faultyGateList = [singleFault.getFaulty(gate.params) if isinstance(gate, singleFault.getGateType()) else gate for gate in template]
		# add faulty gate to the qbIndex row
		for gate in faultyGateList:
			qcFaulty.append(gate, [qbIndex])
			qcFaulty.append(qGate.Barrier(qbIndex))
		qcFaulty.measure(self.quantumRegister, self.classicalRegister)

		return qatgConfiguration(self.circuitSize, self.basisGateSet, self.simulationSetup, \
			singleFault, qcFaultFree, qcFaulty)

	def buildTwoConfiguration(self, template, twoFault):
		qcFaultFree = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		qbIndex = twoFault.getQubit()
		controlQubit = qbIndex[0]
		targetQubit = qbIndex[1]
		for activationGatePair in template:
			if isinstance(activationGatePair, list):
				qcFaultFree.append(activationGatePair[0], [controlQubit])
				qcFaultFree.append(activationGatePair[1], [targetQubit])
				
			else:
				qcFaultFree.append(activationGatePair, [controlQubit, targetQubit])
			
			qcFaultFree.append(qGate.Barrier(controlQubit))
			qcFaultFree.append(qGate.Barrier(targetQubit))
		
		qcFaultFree.measure(self.quantumRegister, self.classicalRegister)
		qcFaulty = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		faultyGateList = [activationGatePair if isinstance(activationGatePair, list) else twoFault.getFaulty(activationGatePair.params) for activationGatePair in template]
		
		for activationGatePair in faultyGateList:
			if isinstance(activationGatePair, list):
				qcFaulty.append(activationGatePair[0], [controlQubit])
				qcFaulty.append(activationGatePair[1], [targetQubit])
				
			else:
				qcFaulty.append(activationGatePair, [controlQubit, targetQubit])

			qcFaulty.append(qGate.Barrier(controlQubit))
			qcFaulty.append(qGate.Barrier(targetQubit))
		qcFaulty.measure(self.quantumRegister, self.classicalRegister)
	
		return qatgConfiguration(self.circuitSize, self.basisGateSet, self.simulationSetup, \
			twoFault, qcFaultFree, qcFaulty)

	def generateTestTemplate(self, faultObject, initialState, findActivationGate):
		templateGateList = [] # list of qGate

		faultyQuantumState = deepcopy(initialState)
		faultfreeQuantumState = deepcopy(initialState)

		for element in range(self.maxTestTemplateSize):
			newElement, faultyQuantumState, faultfreeQuantumState = findActivationGate(faultObject = faultObject, faultyQuantumState = faultyQuantumState, faultfreeQuantumState = faultfreeQuantumState)
			templateGateList += newElement
			effectSize = calEffectSize(faultyQuantumState, faultfreeQuantumState)
			if effectSize > self.minRequiredEffectSize:
				break

		return templateGateList

	def findSingleElement(self, faultObject, faultyQuantumState, faultfreeQuantumState):
		# optimize activation gate
		optimalParameterList, faultyQuantumState, faultfreeQuantumState = self.singleActivationOptimization(faultyQuantumState, faultfreeQuantumState, faultObject)
		newElement = U2GateSetsTranspile(optimalParameterList) # a list of qGate
		newElement.append(faultObject.getOriginalGate())
		return newElement, faultyQuantumState, faultfreeQuantumState

	def singleActivationOptimization(self, faultyQuantumState, faultfreeQuantumState, faultObject):
		originalGateParameters = faultObject.getOriginalGateParameters() # list of parameters
		originalGateMatrix = faultObject.getOriginalGate().to_matrix()
		faultyGateMatrix = faultObject.getFaulty(originalGateParameters).to_matrix() # np.array(gate)

		def score(parameters):
			return vectorDistance(
				matrixOperation([U3(parameters), originalGateMatrix], faultfreeQuantumState), 
				matrixOperation(np.concatenate([self.insertFault2GateList(self.U2GateSetsTranspile(parameters), faultObject), [faultyGateMatrix]]), faultyQuantumState))

		results = []
		for theta in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
					results.append([[theta, phi, lam], score([theta, phi, lam])])
		parameterList = max(results, key = lambda x: x[1])[0]

		for time in range(self.gradientDescentSearchTime):
			newParameterList = [0]*len(parameterList)
			for i in range(len(parameterList)):
				currentScore = score(parameterList)
				parameterList[i] += self.gradientDescentStep
				upScore = score(parameterList)
				parameterList[i] -= 2*self.gradientDescentStep
				downScore = score(parameterList)
				parameterList[i] += self.gradientDescentStep

				if(upScore > currentScore and upScore >= downScore):
					newParameterList[i] += self.gradientDescentStep
				elif(downScore > currentScore and downScore >= upScore):
					newParameterList[i] -= self.gradientDescentStep
				elif upScore == currentScore == downScore:
					newParameterList[i] += self.gradientDescentStep
			if newParameterList == [0, 0, 0]:
				break
			for i in range(len(parameterList)):
				parameterList[i] += newParameterList[i]

		faultyQuantumState = matrixOperation(np.concatenate([self.insertFault2GateList(self.U2GateSetsTranspile(parameterList), faultObject), [faultyGateMatrix]]), faultyQuantumState)
		faultfreeQuantumState = matrixOperation([U3(parameterList), originalGateMatrix], faultfreeQuantumState)

		return parameterList, faultyQuantumState, faultfreeQuantumState

	def findTwoElement(self, faultObject, faultyQuantumState, faultfreeQuantumState):
		
		optimalParameterList, faultyQuantumState, faultfreeQuantumState = self.twoActivationOptimization(faultyQuantumState, faultfreeQuantumState, faultObject)
		aboveActivationGate = self.U2GateSetsTranspile(optimalParameterList[0:3])
		belowActivationGate = self.U2GateSetsTranspile(optimalParameterList[3:6])
		toalActivationGate = [[aboveGate, belowGate] for aboveGate, belowGate in zip(aboveActivationGate, belowActivationGate)]
		toalActivationGate.append(faultObject.getOriginalGate())
		return toalActivationGate, faultyQuantumState, faultfreeQuantumState

	def twoActivationOptimization(self, faultyQuantumState, faultfreeQuantumState, faultObject):
		# for only CNOT have fault
		originalGateParameters = faultObject.getOriginalGateParameters()
		originalGateMatrix = faultObject.getOriginalGate().to_matrix()
		faultyGateMatrix = faultObject.getFaulty(originalGateParameters).to_matrix()
		
		def score(parameters):
			return vectorDistance(
				matrixOperation([np.kron(U3(parameters[0:3]), U3(parameters[3:6])), originalGateMatrix], faultfreeQuantumState), 
				matrixOperation([np.kron(U3(parameters[0:3]), U3(parameters[3:6])), faultyGateMatrix], faultyQuantumState))
		# 3+3 method
		results = []
		for theta in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
					results.append([[theta, phi, lam], score([theta, phi, lam, 0, 0, 0])])
		first_three = max(results, key = lambda x: x[1])[0]

		results = []
		for theta in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
					results.append([[theta, phi, lam], score(first_three + [theta, phi, lam])])
		next_three = max(results, key = lambda x: x[1])[0]
	
		parameterList = np.concatenate([first_three, next_three])

		for time in range(self.gradientDescentSearchTime):
			newParameterList = [0]*len(parameterList)
			for i in range(len(parameterList)):
				currentScore = score(parameterList)
				parameterList[i] += self.gradientDescentStep
				upScore = score(parameterList)
				parameterList[i] -= 2*self.gradientDescentStep
				downScore = score(parameterList)
				parameterList[i] += self.gradientDescentStep

				if(upScore > currentScore and upScore >= downScore):
					newParameterList[i] += self.gradientDescentStep
				elif(downScore > currentScore and downScore >= upScore):
					newParameterList[i] -= self.gradientDescentStep
				elif upScore == currentScore == downScore:
					newParameterList[i] += self.gradientDescentStep
			if newParameterList == [0, 0, 0, 0, 0, 0]:
				break
			for i in range(len(parameterList)):
				parameterList[i] += newParameterList[i]

		print("score: ", score(parameterList))
		faultyQuantumState = matrixOperation([np.kron(U3(parameterList[0:3]), U3(parameterList[3:6])), faultyGateMatrix], faultyQuantumState)
		faultfreeQuantumState = matrixOperation([np.kron(U3(parameterList[0:3]), U3(parameterList[3:6])), originalGateMatrix], faultfreeQuantumState)

		return parameterList, faultyQuantumState, faultfreeQuantumState

	def U2GateSetsTranspile(self, UParameters):
		# to gate list directly
		resultCircuit = self.effectiveUGateCircuit.bind_parameters({self.qiskitParameterTheta: UParameters[0], \
			self.qiskitParameterPhi: UParameters[1], self.qiskitParameterLambda: UParameters[2]})
		return [gate for gate, _, _ in resultCircuit.data] # a list of qGate

	@staticmethod
	def insertFault2GateList(gateList, faultObject):
		return [faultObject.getFaulty(gate.params).to_matrix() if isinstance(gate, faultObject.getGateType()) else gate.to_matrix() for gate in gateList]
		