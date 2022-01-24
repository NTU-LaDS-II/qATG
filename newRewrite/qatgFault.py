import abc
import numpy as np
import qiskit.circuit.library as qGate
from qiskit.circuit.gate import Gate

class qatgFault(abc.ABC):
	def __init__(self, gateType, description = "A qATG Fault."):
		if not issubclass(gateType, Gate):
			raise TypeError('gateType must be one of qiskit.circuit.library')
		self.gateType = gateType
		self.description = description

	def __str__(self):
		return self.description

	def getGateType(self):
		return self.gateType

	@abc.abstractmethod
	def getOriginalGateParameters(self, qubit):
		return NotImplemented

	def getOriginalGate(self, qubit):
		return self.gateType(*self.getOriginalGateParameters(qubit))

	@abc.abstractmethod
	def getFaulty(self, parameters, qubit):
		return NotImplemented
