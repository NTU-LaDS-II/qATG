import random
from qiskit import Aer
from qiskit import execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError

random.seed(114514)

class qatgConfiguration():
	def __init__(self, circuitSize: int, basisGateSet, simulationSetup: dict, faultObject):
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
		self.repetition_list = []
		self.sim_faultfree_distribution = []
		self.sim_faulty_distribution_list = []
		self.max_repetition = 0
		self.boundary = 0 
		
		job_sim = execute(self.qc_faultfree, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		
		summantion_free = {}
		for i in range(2**self.circuit_size):
			summantion_free['{0:b}'.format(i).zfill(self.circuit_size)] = 0
		counts = job_sim.result().get_counts()
		for i in counts:
			summantion_free[i] += counts[i]
		self.sim_faultfree_distribution = to_np(summantion_free)

		for number, qc_faulty in enumerate(self.qc_faulty_list):
			job_sim = execute(qc_faulty, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)

			summantion_faulty = {}
			for i in range(2**self.circuit_size):
				summantion_faulty['{0:b}'.format(i).zfill(self.circuit_size)] = 0
			counts = job_sim.result().get_counts()
			for i in counts:
				summantion_faulty[i] += counts[i]
			self.sim_faulty_distribution_list.append(to_np(summantion_faulty)) 

			faulty_distribution, faultfree_distribution = compression_forfault(self.sim_faulty_distribution_list[-1], self.sim_faultfree_distribution, deepcopy(self.fault_list[number].index))
			repetition, boundary = compute_repetition(faulty_distribution, faultfree_distribution, self.alpha, self.beta)
			self.repetition_list.append((repetition, boundary))

		fault_index = []
		for f in self.fault_list:
			fault_index.append(f.index)

		temp_list = [0, 0]
		for c in self.repetition_list:
			if c[0] > temp_list[0]:
				temp_list[0] = c[0]
				temp_list[1] = c[1]

		self.max_repetition = temp_list[0]
		self.boundary = temp_list[1]

		self.sim_overkill = self.calOverkill(self.sim_faultfree_distribution, self.sim_faulty_distribution_list, fault_index, alpha=self.alpha)
		self.sim_testescape = self.calTestescape(self.sim_faulty_distribution_list, self.sim_faulty_distribution_list, fault_index, alpha=self.alpha)
		
		return

	def calTestescape(self):
		pass

	def calOverkill(self):
		pass