import random
import numpy as np
from math import ceil
from scipy.stats import chi2, ncx2
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import qiskit.circuit.library as qGate
from qiskit.circuit.gate import Gate

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import standard_errors, ReadoutError

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))

from qatgUtil import *

random.seed(114514)

class QATGConfiguration():
	"""the return results of qatg is described as qatgConfiguration objects"""
	def __init__(self, circuitSetup: dict, simulationSetup: dict, faultObject):
		# circuitSetup: circuitSize, basisGateSet, quantumRegisterName, classicalRegisterName, circuitInitializedStates
		
		self.circuitSize = circuitSetup['circuitSize']
		self.basisGateSet = circuitSetup['basisGateSet']
		self.basisGateSetString = circuitSetup['basisGateSetString']
		self.circuitInitializedStates = circuitSetup['circuitInitializedStates']
		self.backend = AerSimulator()

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
		self.OnestateFidelity = np.nan

		self.noiseModel = self.getNoiseModel()

	def __str__(self):
		rt = ""
		rt += "Target fault: { " + str(self.faultObject) + " }\n"
		rt += "Length: " + str(self.cktDepth)
		rt += "\tRepetition: " + str(self.repetition)
		rt += "\tCost: " + str(self.cktDepth * self.repetition) + "\n"
		rt += "Chi-Value boundary: " + str(self.boundary) + "\n"
		rt += "State Fidelity: " + str(self.OnestateFidelity) + "\n"
		rt += "Overkill: "+ str(self.simulatedOverkill)
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

	def setTemplate(self, template, OnestateFidelity):
		# template itself is faultfree
		self.OnestateFidelity = OnestateFidelity

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

			self.faultfreeQCKT.append(qGate.Barrier(len(qbIndexes)), qbIndexes)
			self.faultyQCKT.append(qGate.Barrier(len(qbIndexes)), qbIndexes)

		self.faultfreeQCKT.measure(self.quantumRegister, self.classicalRegister)
		self.faultyQCKT.measure(self.quantumRegister, self.classicalRegister)

		self.cktDepth = len(template)

		return

	def simulate(self):
		new_circuit = transpile(self.faultfreeQCKT, self.backend)
		simulateJob = self.backend.run(new_circuit, noise_model = self.noiseModel, seed_simulator = 1, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()
		self.faultfreeDistribution = [0] * (2 ** self.circuitSize)
		for k in counts:
			self.faultfreeDistribution[int(k, 2)] = counts[k]
		self.faultfreeDistribution = np.array(self.faultfreeDistribution / np.sum(self.faultfreeDistribution))

		new_circuit = transpile(self.faultyQCKT, self.backend)
		simulateJob = self.backend.run(new_circuit, noise_model = self.noiseModel, seed_simulator = 1, shots = self.simulationShots)
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

		effectSize = qatgCalEffectSize(self.faultyDistribution, self.faultfreeDistribution)
		
		lowerBoundEffectSize = 0.8 if effectSize > 0.8 else effectSize

		chi2Value = chi2.ppf(self.targetAlpha, degreeOfFreedom)
		repetition = ceil(chi2Value / (lowerBoundEffectSize ** 2))
		nonCentrality = repetition * (effectSize ** 2)
		nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		while nonChi2Value < chi2Value:
			repetition += 1
			nonCentrality += effectSize ** 2
			nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		
		boundary = (nonChi2Value * 0.3 + chi2Value * 0.7)
		if repetition >= qatgINT_MAX or repetition <= 0:
			raise ValueError("Error occured calculating repetition")
		
		return repetition, boundary

	def calOverkill(self):
		overkill = 0
		expectedDistribution = self.faultyDistribution
		observedDistribution = self.faultfreeDistribution

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(observedDistribution.shape[0]), weights = observedDistribution, k = self.repetition)
			sampledObservedDistribution = np.zeros(observedDistribution.shape[0])
			for d in sampledData:
				sampledObservedDistribution[d] += 1
			sampledObservedDistribution = sampledObservedDistribution / self.repetition

			deltaSquare = np.square(expectedDistribution - sampledObservedDistribution)
			chiStatistic = self.repetition * np.sum(deltaSquare/(expectedDistribution+qatgINT_MIN))

			# test should pass, chiStatistic should > boundary
			if chiStatistic <= self.boundary:
				overkill += 1

		return overkill / self.testSampleTime

	def calTestEscape(self):
		testEscape = 0
		expectedDistribution = self.faultyDistribution
		observedDistribution = self.faultyDistribution

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(observedDistribution.shape[0]), weights = observedDistribution, k = self.repetition)
			sampledObservedDistribution = np.zeros(observedDistribution.shape[0])
			for d in sampledData:
				sampledObservedDistribution[d] += 1
			sampledObservedDistribution = sampledObservedDistribution / self.repetition

			deltaSquare = np.square(expectedDistribution - sampledObservedDistribution)
			chiStatistic = self.repetition * np.sum(deltaSquare/(expectedDistribution+qatgINT_MIN))

			# test should fail, chiStatistic should <= boundary
			if chiStatistic > self.boundary:
				testEscape += 1

		return testEscape / self.testSampleTime

	@property
	def circuit(self):
		return self.faultfreeQCKT
	
