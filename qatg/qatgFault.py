import abc
import qiskit.circuit.library as qGate
from qiskit.circuit.gate import Gate

class QATGFault(abc.ABC):
	def __init__(self, gateType, qubits, description = None):
		if not issubclass(gateType, Gate):
			raise TypeError('gateType must be one of qiskit.circuit.library')
		self.gateType = gateType
		if isinstance(qubits, int):
			qubits = [qubits]
		self.qubits = qubits
		if len(qubits) != gateType(*[0]*(gateType.__init__.__code__.co_argcount - len(gateType.__init__.__defaults__) - 1)).num_qubits:
			raise ValueError("the number of qubits the gate acts on and the number of input qubits does not match.")
		self.description = f"gateType: {self.gateType}, qubits: {self.qubits}" if not description else description

	def __str__(self):
		return self.description

	def getGateType(self):
		return self.gateType

	def getGateTypeStr(self):
		return self.gateType.__name__[:-4].lower()

	def isSameGateType(self, gate):
		return isinstance(gate, self.gateType)

	def getQubits(self):
		return self.qubits

	@abc.abstractmethod
	def createOriginalGate(self): # return self.gateType
		return NotImplemented

	@abc.abstractmethod
	def createFaultyGate(self, faultfreeGate): # return self.gateType
		return NotImplemented

	def getFaultyBehaviorFunction(self):
		return self.createFaultyGate
