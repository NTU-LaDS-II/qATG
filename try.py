import numpy as np
# import qiskit
import math
from Fault import *
from Gate import *
from scipy.stats import chi2, ncx2
import qiskit.circuit.library as Qgate
from qiskit.circuit import Parameter
from qiskit.circuit.quantumregister import Qubit
# from qiskit.quantum_info import process_fidelity
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError
from qiskit import Aer
from qiskit import execute, transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
# import statsmodels.stats.power as smp
from qiskit import transpile
from numpy import pi
import random

from util import *
class QuantumGate():
	def __init__(self, QiskitGate , pos , buff):
		self.gate = QiskitGate # what Qgate
		self.pos = pos # which qubit 型別為list
		self.buff = buff




q = QuantumCircuit(1)
a = QuantumGate(Qgate.U3Gate(1 , 1 , 1) , [Qubit(QuantumRegister(1) , 0)] , [])
print(type(a.gate))