import numpy as np
import math
import cmath
from Fault import *
from Gate import *
from scipy.stats import chi2, ncx2
import qiskit.circuit.library as Qgate
from qiskit.circuit import Parameter
from qiskit.circuit.quantumregister import Qubit
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError
# from qiskit import Aer
from qiskit import execute, transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
# from qiskit.extensions import *
# from qiskit import transpile
from numpy import pi
from copy import deepcopy
# import random
# from QuantumGate import *
# from util import *


# u = Qgate.U3Gate(1 , 1 , 1)
# print(u)
# print(u.params)

# u = [1]
# a = [[1, 2], [3, 4]]
# c = a + u
# print(c)

q = QuantumCircuit(2)
u = Qgate.U3Gate(1 , 1 , 1)
a = Qgate.U3Gate(2 , 2 , 2)
c = Qgate.U3Gate(3, 3, 3)
# q.append(u, qargs = [0])
# q.append(Qgate.Barrier(0))
# q.append(c, qargs = [0])
# q.append(a, qargs = [1])
# print(q)
# u_prime = deepcopy(u)
# print(u , u_prime)
# q._data.pop(0)
# q._data.pop(0)
d = Qgate.CXGate()
q.append(d, [0,1])
print(q)
# q.draw()
print(np.eye(4))