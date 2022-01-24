import random
import numpy as np
from scipy.stats import chi2, ncx2
from qiskit import Aer
from qiskit import execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError

from qatgUtil import *
random.seed(114514)

class qatgConfiguration():
	def __init__(self, circuitSize: int, basisGateSet, simulationSetup: dict, \
			faultObject, faultfreeQuantumCircuit, faultyQuantumCircuit):
		self.circuitSize = circuitSize
		self.basisGateSet = basisGateSet
		self.basisGateSetString = [gate.__name__[:-4].lower() for gate in self.basisGateSet]
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
		self.faultfreeQuantumCircuit = faultfreeQuantumCircuit
		self.faultyQuantumCircuit = faultyQuantumCircuit

		self.faultfreeDistribution = []
		self.faultyDistribution = []
		self.repetition = np.nan
		self.boundary = np.nan
		self.simulatedOverkill = np.nan
		self.simulatedTestescape = np.nan

		self.noiseModel = self.getNoiseModel()

	def __str__(self):
		rt = ""
		rt += "Target fault: " + str(self.faultObject) + "\n"
		rt += "Length: " + str(len(self.faultfreeQuantumCircuit))
		rt += "\tRepetition: " + str(self.repetition)
		rt += "\tCost: " + str(len(self.faultfreeQuantumCircuit) * self.repetition) + "\n"
		rt += "Overkill: "+str(self.simulatedOverkill)
		rt += "\tTest Escape: " + str(self.simulatedTestescape) + "\n"
		rt += "Circuit: \n" + str(self.faultfreeQuantumCircuit)

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

	def simulate(self):
		simulateJob = execute(self.faultfreeQuantumCircuit, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()
		
		self.faultfreeDistribution = [0] * (2 ** self.circuitSize)
		for i in counts:
			self.faultfreeDistribution[int(i, 2)] = counts[i]
		self.faultfreeDistribution = np.array(self.faultfreeDistribution / np.sum(self.faultfreeDistribution))

		simulateJob = execute(self.faultyQuantumCircuit, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()

		self.faultyDistribution = [0] * (2 ** self.circuitSize)
		for i in counts:
			self.faultyDistribution[int(i, 2)] = counts[i]
		self.faultyDistribution = np.array(self.faultyDistribution / np.sum(self.faultyDistribution))

		self.repetition, self.boundary = self.calRepetition(self.faultyDistribution, self.faultfreeDistribution, self.targetAlpha, self.targetBeta)

		self.simulatedOverkill = self.calOverkill(self.faultfreeDistribution, self.faultyDistribution)
		self.simulatedTestescape = self.calTestEscape(self.faultyDistribution)
		
		return

	@staticmethod
	def calRepetition(faultyDistribution, faultfreeDistribution, alpha, beta):
		if faultfreeDistribution.shape != faultyDistribution.shape:
			raise ValueError('input shape not consistency')

		degreeOfFreedom = faultfreeDistribution.shape[0] - 1
		effectSize = calEffectSize(faultyDistribution, faultfreeDistribution)
		lowerBoundEffectSize = 0.8 if effectSize > 0.8 else effectSize

		repetition = chi2.ppf(alpha, degreeOfFreedom) / (lowerBoundEffectSize ** 2)
		while True:
			nonCentrality = repetition * (effectSize ** 2)
			chi2Value = chi2.ppf(alpha, degreeOfFreedom)
			nonChi2Value = ncx2.ppf(1 - beta, degreeOfFreedom, nonCentrality)
			if nonChi2Value >= chi2Value:
				break
			else:
				repetition += 1
		
		boundary = (nonChi2Value * 0.3 + chi2Value * 0.7)
		if repetition >= INT_MAX or repetition <= 0:
			return INT_MAX
		
		return ceil(repetition), boundary

	def calOverkill(self, faultfreeDistribution, faultyDistribution):
		overkill = 0
		expectedDistribution = faultfreeDistribution * self.repetition
		observedDistribution = faultyDistribution
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

	def calTestEscape(self, faultyDistribution):
		testEscape = 0			
		expectedDistribution = faultyDistribution * self.repetition
		chiValue = self.boundary

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(faultyDistribution.shape[0]), weights = faultyDistribution, k = self.repetition)
			observedDistribution = np.zeros(faultyDistribution.shape[0])
			for d in sampledData:
				observedDistribution[d] += 1

			deltaSquare = np.square(expectedDistribution - observedDistribution)
			chiStatistic = np.sum(deltaSquare / (expectedDistribution + INT_MIN))
			if chiStatistic > chiValue:
				testescape += 1
		return testescape / self.testSampleTime
