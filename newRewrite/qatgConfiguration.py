import random
from qiskit import Aer
from qiskit import execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError

random.seed(114514)

class qatgConfiguration():
	def __init__(self, circuitSize: int, basisGateSet, simulationSetup: dict, \
			faultObject, faultfreeQuantumCircuit, faultyQuantumCircuit):
		self.circuitSize = circuitSize
		self.basisGateSet = basisGateSet
		self.basisGateSetString = [gate.__name__[:-4].lower() for gate in self.basisGateSet]
		self.backend = Aer.get_backend('qasm_simulator')
		self.noiseModel = self.getNoiseModel()

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
		self.simulatedFaultfreeDistribution = []
		self.simulatedFaultyDistribution = []
		self.repetition = 0
		self.boundary = 0 
		
		simulateJob = execute(self.faultfreeQuantumCircuit, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()
		
		self.simulatedFaultfreeDistribution = [0] * (2 ** self.circuitSize)
		for i in counts:
			self.simulatedFaultfreeDistribution[int(i, 2)] = counts[i]
		self.simulatedFaultfreeDistribution = self.simulatedFaultfreeDistribution / np.sum(self.simulatedFaultfreeDistribution)

		simulateJob = execute(self.faultyQuantumCircuit, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		counts = simulateJob.result().get_counts()

		self.simulatedFaultyDistribution = [0] * (2 ** self.circuitSize)
		for i in counts:
			self.simulatedFaultyDistribution[int(i, 2)] = counts[i]
		self.simulatedFaultyDistribution = self.simulatedFaultyDistribution / np.sum(self.simulatedFaultyDistribution)

		self.repetition, self.boundary = compute_repetition(faulty_distribution, faultfree_distribution, self.alpha, self.beta)

		self.sim_overkill = self.calOverkill(self.simulatedFaultfreeDistribution, self.simulatedFaultyDistribution, fault_index, alpha=self.alpha)
		self.sim_testescape = self.calTestescape(self.simulatedFaultyDistribution, self.simulatedFaultyDistribution, fault_index, alpha=self.alpha)
		
		return

	def calTestescape(self):
		pass

	def calOverkill(self):
		pass