import numpy as np
from copy import deepcopy
from qiskit import Aer
from qiskit import execute
from qiskit import transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
import qiskit.circuit.library as qGate

from qatgFault import qatgFault
from qatgUtil import U3, CNOT

class qatg():
	def __init__(self, circuitSize: int = None, basisGateSet: list[qGate], couplingMap: list[list], \
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
			for qubit in range(self.circuitSize):
				template = self.generateTestTemplate(faultObject = singleFault, target = qubit, initialState = singleInitialState, findActivationGate = findSingleElement)
				

		for twoFault in twoFaultList:
			if not issubclass(singleFault, qatgFault):
				raise TypeError(f"{twoFault} should be subclass of qatgFault")
			for couple in couplingMap:
				template = self.generateTestTemplate(faultObject = twoFault, target = couple, initialState = twoInitialState, findActivationGate = findTwoElement)

		pass

	def getTestTemplate(self, faultObject, target, initialState, findActivationGate):
		templateGateList = [] # list of qGate

		faultyQuantumState = deepcopy(initialState)
		faultfreeQuantumState = deepcopy(initialState)

		for element in range(self.maxTestTemplateSize):
			newElement, faultyQuantumState, faultfreeQuantumState = findActivationGate(faultObject = faultObject, target = target, faultyQuantumState = faultyQuantumState, faultfreeQuantumState = faultfreeQuantumState)
			# newElement: list[np.array(gate)]
			templateGateList = np.concatenate([templateGateList, newElement])
			effectSize = calEffectSize(faultyQuantumState, faultfreeQuantumState)
			if effectsize > self.minRequiredEffectSize:
				break	

		# print?

		return templateGateList

	def findSingleElement(self, faultObject, target, faultyQuantumState, faultfreeQuantumState):
		
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
		optimalParameterList = self.singleGridSearch(faultyGateMatrix , originalGateMatrix , faultyQuantumState , faultfreeQuantumState , faultObject)
		# optimalParameterList = self.singleGradientDescent(optimalParameterList, faultyGateMatrix , originalGateMatrix , faultyQuantumState , faultfreeQuantumState , faultObject)
		newElement.append(U2GateSetsTranspile(optimalParameterList))
		newElement.append(faultObject.getOriginalGate(target))
		return newElement

	def singleGridSearch(self, faultyGateMatrix, originalGateMatrix, faultyQuantumState, faultfreeQuantumState):
		def score(parameters):
			return vectorDistance(
				matrixOperation([U3(parameters), originalGateMatrix], faultfreeQuantumState), 
				matrixOperation(np.concatenate([insertFault2GateList(U2GateSetsTranspile(parameters), faultObject, target), [faultyGateMatrix]]), faultyQuantumState))
		results = []
		for theta in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=self.gridSlice, endpoint = True):
					results.append([[theta, phi, lam], score([theta, phi, lam])])
		return max(results, key = lambda x: x[1])[0]

	def findTwoElement(self, faultObject, target, faultyQuantumState, faultfreeQuantumState):
		# TODO
		newElement = [] # list of list of 2 qGate

		originalGateParameters = faultObject.getOriginalGateParameters(target) # list of 6 parameters
		originalGateMatrix = faultObject.getOriginalGate(target).to_matrix()
		faultyGateMatrix = faultObject.getFaultyGate(originalGateParameters, target).to_matrix() # np.array(gate)
		
		optimalParameterList = self.twoGridSearch(faultyGateMatrix , originalGateMatrix , faultyQuantumState , faultfreeQuantumState , faultObject)
		# optimalParameterList = self.twoGradientDescent(optimalParameterList, faultyGateMatrix , originalGateMatrix , faultyQuantumState , faultfreeQuantumState , faultObject)
		aboveActivationGate = U2GateSetsTranspile(optimalParameterList[0:3])
		belowActivationGate = U2GateSetsTranspile(optimalParameterList[3:6])
		toalActivationGate = [[aboveGate, belowGate] for aboveGate, belowGate in zip(aboveActivationGate, belowActivationGate)]
		return toalActivationGate + [originalGateMatrix]

	def twoGridSearch(self, faultyGateMatrix, originalGateMatrix, faultyQuantumState, faultfreeQuantumState):
		# for only CNOT have fault
		def score(parameters):
			return vectorDistance(
				matrixOperation([U3(parameters[0:3]), U3(parameters[3:6]), originalGateMatrix], faultfreeQuantumState), 
				matrixOperation([U3(parameters[0:3]), U3(parameters[3:6]), faultyGateMatrix], faultyQuantumState))

		# 3+3 method
		results = []
		for theta in np.linspace(-np.pi, np.pi, num=self.grid_slice, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=self.grid_slice, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=self.grid_slice, endpoint = True):
					results.append([[theta, phi, lam], score([theta, phi, lam, 0, 0, 0])])
		first_three = max(results, key = lambda x: x[1])[0]

		results = []
		for theta in np.linspace(-np.pi, np.pi, num=self.grid_slice, endpoint = True):
			for phi in np.linspace(-np.pi, np.pi, num=self.grid_slice, endpoint = True):
				for lam in np.linspace(-np.pi, np.pi, num=self.grid_slice, endpoint = True):
					results.append([[theta, phi, lam], score(first_three + [theta, phi, lam])])
		next_three = max(results, key = lambda x: x[1])[0]
	
		return first_three + next_three
	
	def twoGradientDescent(self, parameterList, faultyGateMatrix, originalGateMatrix, faultyQuantumState, faultfreeQuantumState):
		# parameterList = [0, 0, 0, 0, 0, 0]
		def score(parameters):
			return vectorDistance(
				matrixOperation([[U3(parameters[0:3]), U3(parameters[3:6])], originalGateMatrix], faultfreeQuantumState), 
				matrixOperation([[U3(parameters[0:3]), U3(parameters[3:6])], faultyGateMatrix], faultyQuantumState))

		for j in range(self.search_time):
			newParameterList = [0]*len(parameterList)
			for i in range(len(parameterList)):
				currentScore = score(parameterList)
				parameterList[i] += self.step
				upScore = score(parameterList)
				parameterList[i] -= 2*self.step
				downScore = score(parameterList)
				parameterList[i] += self.step

				if(upScore > currentScore and upScore >= downScore):
					newParameterList[i] += self.step
					# newParameterList[i] += self.step*(upScore - currentScore)
				elif(downScore > currentScore and downScore >= upScore):
					newParameterList[i] -= self.step
					# newParameterList[i] -= self.step*(downScore - currentScore)
				elif upScore == currentScore == downScore:
					newParameterList[i] += self.step
			if newParameterList == [0, 0, 0, 0, 0, 0]:
				break
			for i in range(len(parameterList)):
				parameterList[i] += newParameterList[i]

		print("score: ", score(parameterList))
		return parameterList
	def singleGradientDescent(self, parameterList, faultyGateMatrix, originalGateMatrix, faultyQuantumState, faultfreeQuantumState, fault):
		# parameterList = [0 , 0 , 0] 應該根據gate型態給予參數
		def score(parameters):
			return vectorDistance(
				matrixOperation([U3(parameters), originalGateMatrix], faultfreeQuantumState), 
				matrixOperation(np.concatenate([insertFault2GateList(U2GateSetsTranspile(parameters), faultObject, target), [faultyGateMatrix]]), faultyQuantumState))
		# print("score: ", score)

		for j in range(self.search_time):
			newParameterList = [0]*len(parameterList)
			for i in range(len(parameterList)):
				currentScore = score(parameterList)
				parameterList[i] += self.step
				upScore = score(parameterList)
				parameterList[i] -= 2*self.step
				downScore = score(parameterList)
				parameterList[i] += self.step

				if(upScore > currentScore and upScore >= downScore):
					# for gradient descent only
					newParameterList[i] += self.step
					# for grid search
					# newParameterList[i] += self.step*(upScore - currentScore)
				elif(downScore > currentScore and downScore >= upScore):
					# for gradient descent only
					newParameterList[i] -= self.step
					# for grid search
					# newParameterList[i] -= self.step*(downScore - currentScore)
				# for gradient descent only
				elif upScore == currentScore == downScore:
					newParameterList[i] += self.step
			if newParameterList == [0, 0, 0]:
				break
			for i in range(len(parameterList)):
				parameterList[i] += newParameterList[i]

		print("score: ", score(parameterList))
		return parameterList
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

	@staticmethod
	def insertFault2GateList(gateList, faultObject, target):
		return [faultObject.getFaulty(gate.params, target).to_matrix() if isinstance(gate, faultObject.getGateType()) else gate.to_matrix() for gate in gateList]
		