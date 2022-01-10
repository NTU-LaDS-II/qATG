from qiskit import Aer

class qatg():
	def __init__(self, circuitSize, basisGateSet, \
			quantumRegisterName = 'q', classicalRegisterName = 'c', \
			targetAlpha = 0.99, targetBeta = 0.999, \
			gridSlice = 21, \
			gradientDescentSearchTime = 800, gradientDescentStep = 0.01, \
			maxTestTemplateSize = 50, minRequiredEffectSize = 3, \
			testSampleTime = 10000):
		self.circuitSize = circuitSize
		self.gateSet = gateSet
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
		# test