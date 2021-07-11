

class QuantumGate():
	def __init__(self, QiskitGate , pos , buff):
		self.gate = QiskitGate # what Qgate
		self.pos = pos # which qubit 型別為list
		self.buff = buff # []
