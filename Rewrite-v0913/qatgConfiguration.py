import random
import numpy as np
from math import ceil
from scipy.stats import chi2, ncx2
from qiskit import Aer
from qiskit import execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import qiskit.circuit.library as qGate
from qiskit.circuit.gate import Gate
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError

from qatgUtil import *
random.seed(114514)

class qatgConfiguration():
	"""the return results of qatg is described as qatgConfiguration objects"""
	def __init__(self, circuitSetup: dict, simulationSetup: dict, faultObject):
		# circuitSetup: circuitSize, basisGateSet, quantumRegisterName, classicalRegisterName, circuitInitializedStates
		
		self.circuitSize = circuitSetup['circuitSize']
		self.basisGateSet = circuitSetup['basisGateSet']
		self.basisGateSetString = circuitSetup['basisGateSetString']
		self.circuitInitializedStates = circuitSetup['circuitInitializedStates']
		self.backend = Aer.get_backend('qasm_simulator')

		self.oneQubitErrorProb = simulationSetup['oneQubitErrorProb']
		self.twoQubitErrorProb = simulationSetup['twoQubitErrorProb']
		self.zeroReadoutErrorProb = simulationSetup['zeroReadoutErrorProb']
		self.oneReadoutErrorProb = simulationSetup['oneReadoutErrorProb']

		self.targetAlpha = simulationSetup['targetAlpha']
		self.targetBeta = simulationSetup['targetBeta']

		self.simulationShots = simulationSetup['simulationShots']
		self.testSampleTime = simulationSetup['testSampleTime']

		self.faultObject = faultObject

		quantumRegisterName = circuitSetup['quantumRegisterName']
		classicalRegisterName = circuitSetup['classicalRegisterName']
		self.quantumRegister = QuantumRegister(self.circuitSize, quantumRegisterName)
		self.classicalRegister = ClassicalRegister(self.circuitSize, classicalRegisterName)

		self.faultfreeQCKT = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		self.faultyQCKT = QuantumCircuit(self.quantumRegister, self.classicalRegister)

		self.faultfreeDistribution = []
		self.faultyDistribution = []
		self.repetition = np.nan
		self.boundary = np.nan
		self.simulatedOverkill = np.nan
		self.simulatedTestescape = np.nan
		self.cktDepth = np.nan

		self.noiseModel = self.getNoiseModel()

	def __str__(self):
		rt = ""
		rt += "Target fault: { " + str(self.faultObject) + " }\n"
		rt += "Length: " + str(self.cktDepth)
		rt += "\tRepetition: " + str(self.repetition)
		rt += "\tCost: " + str(self.cktDepth * self.repetition) + "\n"
		rt += "Overkill: "+str(self.simulatedOverkill)
		rt += "\tTest Escape: " + str(self.simulatedTestescape) + "\n"
		# rt += "Circuit: \n" + str(self.faultfreeQCKT)

		return rt

	def getNoiseModel(self):
		# Depolarizing quantum errors
		oneQubitError = standard_errors.depolarizing_error(self.oneQubitErrorProb, 1)
		twoQubitError = standard_errors.depolarizing_error(self.twoQubitErrorProb, 2)
		qubitReadoutError = ReadoutError([self.zeroReadoutErrorProb, self.oneReadoutErrorProb])

		# Add errors to noise model
		noiseModel = NoiseModel()
		noiseModel.add_all_qubit_quantum_error(oneQubitError, self.basisGateSetString)
		noiseModel.add_all_qubit_quantum_error(twoQubitError, ['cx'])
		noiseModel.add_all_qubit_readout_error(qubitReadoutError)

		return noiseModel

	def setTemplate(self, template):
		# template itself is faultfree

		qbIndexes = self.faultObject.getQubits()

		for gates in template:
			# in template, a list for seperate qubits and a gate for all qubits
			if isinstance(gates, list):
				for k in range(len(gates)):
					self.faultfreeQCKT.append(gates[k], [qbIndexes[k]])
					if self.faultObject.isSameGateType(gates[k]):
						self.faultyQCKT.append(self.faultObject.createFaultyGate(gates[k]), [qbIndexes[k]])
					else:
						self.faultyQCKT.append(gates[k], [qbIndexes[k]])
			elif issubclass(type(gates), Gate):
				self.faultfreeQCKT.append(gates, qbIndexes)
				if self.faultObject.isSameGateType(gates):
					self.faultyQCKT.append(self.faultObject.createFaultyGate(gates), qbIndexes)
				else:
					self.faultyQCKT.append(gates, qbIndexes)
			else:
				raise TypeError(f"Unknown object \"{gates}\" in template")

			for qb in qbIndexes:
				self.faultfreeQCKT.append(qGate.Barrier(qb))
				self.faultyQCKT.append(qGate.Barrier(qb))

		self.faultfreeQCKT.measure(self.quantumRegister, self.classicalRegister)
		self.faultyQCKT.measure(self.quantumRegister, self.classicalRegister)

		self.cktDepth = len(template)

		return

	def simulate(self):
		simulateJob = execute(self.faultfreeQCKT, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()
		self.faultfreeDistribution = [0] * (2 ** self.circuitSize)
		for k in counts:
			self.faultfreeDistribution[int(k, 2)] = counts[k]
		self.faultfreeDistribution = np.array(self.faultfreeDistribution / np.sum(self.faultfreeDistribution))

		simulateJob = execute(self.faultyQCKT, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()
		self.faultyDistribution = [0] * (2 ** self.circuitSize)
		for k in counts:
			self.faultyDistribution[int(k, 2)] = counts[k]
		self.faultyDistribution = np.array(self.faultyDistribution / np.sum(self.faultyDistribution))

		self.repetition, self.boundary = self.calRepetition()

		self.simulatedOverkill = self.calOverkill()
		self.simulatedTestescape = self.calTestEscape()
		
		return

	def calRepetition(self):
		if self.faultfreeDistribution.shape != self.faultyDistribution.shape:
			raise ValueError('input shape not consistency')

		degreeOfFreedom = self.faultfreeDistribution.shape[0] - 1
		effectSize = calEffectSize(self.faultyDistribution, self.faultfreeDistribution)
		lowerBoundEffectSize = 0.8 if effectSize > 0.8 else effectSize

		repetition = chi2.ppf(self.targetAlpha, degreeOfFreedom) / (lowerBoundEffectSize ** 2)
		nonCentrality = repetition * (effectSize ** 2)
		chi2Value = chi2.ppf(self.targetAlpha, degreeOfFreedom)
		nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		while nonChi2Value < chi2Value:
			repetition += 1
			# nonCentrality = repetition * (effectSize ** 2)
			nonCentrality += effectSize ** 2
			chi2Value = chi2.ppf(self.targetAlpha, degreeOfFreedom)
			nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		
		boundary = (nonChi2Value * 0.3 + chi2Value * 0.7)
		if repetition >= INT_MAX or repetition <= 0:
			return INT_MAX
		
		return ceil(repetition), boundary

	def calOverkill(self):
		overkill = 0
		expectedDistribution = self.faultfreeDistribution * self.repetition
		observedDistribution = self.faultyDistribution
		chiValue = self.boundary

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(observedDistribution.shape[0]), weights = observedDistribution, k = self.repetition)
			observedDistribution = np.zeros(observedDistribution.shape[0])
			for d in sampledData:
				observedDistribution[d] += 1

			deltaSquare = np.square(expectedDistribution - observedDistribution)
			chiStatistic = np.sum(deltaSquare/(expectedDistribution+INT_MIN))
			if chiStatistic <= chiValue:
				overkill += 1
		return overkill / self.testSampleTime

	def calTestEscape(self):
		testEscape = 0			
		expectedDistribution = self.faultyDistribution * self.repetition
		chiValue = self.boundary

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(self.faultyDistribution.shape[0]), weights = self.faultyDistribution, k = self.repetition)
			observedDistribution = np.zeros(self.faultyDistribution.shape[0])
			for d in sampledData:
				observedDistribution[d] += 1

			deltaSquare = np.square(expectedDistribution - observedDistribution)
			chiStatistic = np.sum(deltaSquare / (expectedDistribution + INT_MIN))
			if chiStatistic > chiValue:
				testEscape += 1
		return testEscape / self.testSampleTime
