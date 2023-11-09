import numpy as np
import qiskit.circuit.library as qGate

from qatg import QATG
from qatg import QATGFault

def phi_lamToZXZ(lam, phi):
	#print(phi)
	theta1 = lam / 2 + np.pi / 2
	theta2 = phi
	theta3 = lam / 2 - np.pi / 2
	#
	return theta1, theta2, theta3
def wrap_to_2pi(x):
	return (x ) % (2*np.pi)
#class DepolarizingFault(QATGFault):
#    def __init__(self,qubits,prob):
class DepolarizingFault(QATGFault):
	def __init__(self,gateType,qubits,fault_size):
		super(RotationFault, self).__init__(gateType, qubits)
		self.fault_size = fault_size
		self.faultfreeGate = None
	def __str__(self):
		rm = self.description
		rm += '\n'
		rm += f"{self.fault_size=}"
		rm += '\n'
		return rm
	def createOriginalGate(self):
		G = self.getGateType()
		self.faultfreeGate = G()
		return G()
	def createFaultyGate(self):
		
		if len(self.qubits) == 1:
			p = self.fault_size/(4+2*self.fault_size)
			pauli_gate = random.choices([qGate.XGate,qGate.YGate,qGate.ZGate,qGate.IGate], weights=[p, p, p,1-3*p])[0]
			pauli_gate_M = pauli_gate.to_matrix()
			if self.faultfreeGate is not None:
				faultfree_M = self.faultfreeGate.to_matrix()
			else:
				G = self.createOriginalGate()
				faultfree_M = G.to_matrix()
			combined_faulty_M = pauli_gate_M@faultfree_M
		else:
			p = self.fault_size/(16+14*self.fault_size)
			pauli_gate = random.choices([[qGate.XGate,qGate.XGate],[qGate.YGate,qGate.XGate],[qGate.ZGate,qGate.XGate],[qGate.IGate,qGate.XGate], \
       										[qGate.XGate,qGate.YGate],[qGate.YGate,qGate.YGate],[qGate.ZGate,qGate.YGate],[qGate.IGate,qGate.YGate],\
                     						[qGate.XGate,qGate.ZGate],[qGate.YGate,qGate.ZGate],[qGate.ZGate,qGate.ZGate],[qGate.IGate,qGate.ZGate],\
                               				[qGate.XGate,qGate.IGate],[qGate.YGate,qGate.IGate],[qGate.ZGate,qGate.IGate],[qGate.IGate,qGate.IGate],\
                                       	], weights=[p, p, p,p,p,p,p,p,p,p,p,p,p,p,p,1-15*p])[0]
			pauli_gate_M = np.kron(pauli_gate[0].to_matrix(),pauli_gate[1].to_matrix())
			if self.faultfreeGate is not None:
				faultfree_M = self.faultfreeGate.to_matrix()
			else:
				G = self.createOriginalGate()
				faultfree_M = G.to_matrix()
			combined_faulty_M = pauli_gate_M@faultfree_M
		combined_faulty_gate = UnitaryGate(combined_faulty_M)
		return combined_faulty_gate
class RotationFault(QATGFault):
	def __init__(self, gateType,qubits,angleZ1,angleX,angleZ2,cnot_type='control'):
		super(RotationFault, self).__init__(gateType, qubits)
		self.angleZ1 = angleZ1
		self.angleX = angleX
		self.angleZ2 = angleZ2
		self.cnot_type = cnot_type
		self.combined_faulty_M = None
		self.faultfreeGate = None
		self.faultyOnlyGate = None
	def __str__(self):
		rm = self.description
		rm += '\n'
		rm += f"angleZ : {self.angleZ1+self.angleZ2} \n"
		rm += f"angleX : {self.angleX} \n"
		rm += self.cnot_type + "\n"
		return rm
	def createOriginalGate(self):
		G = self.getGateType()
		self.faultfreeGate = G()
		return G()
		#return qGate.UGate(*self.params)
	def createFaultyGate(self):
		if self.combined_faulty_M is not None:
			return UnitaryGate(self.combined_faulty_M)
		rz1_gate = qGate.RZGate(self.angleZ1).to_matrix()
		rx_gate = qGate.RXGate(self.angleX).to_matrix()
		rz2_gate = qGate.RZGate(self.angleZ2).to_matrix()
		ZXZ_gate = rz2_gate@rx_gate@rz1_gate
		self.faultyOnlyGate = ZXZ_gate
		if self.faultfreeGate is not None:
			faultfree_M = self.faultfreeGate.to_matrix()
		else:
			G = self.createOriginalGate()
			faultfree_M = G.to_matrix()
		#combined_faulty_gate = QuantumCircuit(1).compose(faultfreeGate).compose(rz1_gate).compose(rx_gate).compose(rz2_gate)
		
		if len(self.qubits) == 1:
			combined_faulty_M = ZXZ_gate@faultfree_M           
			#combined_faulty_gate.name = "Combined Single Faulty Gate"
			pass
		elif self.cnot_type == 'control':
			#combined_faulty_M = faultfree_M@rz1_gate@rx_gate@rz2_gate
			ZXZ_gate_I = np.linalg.inv(ZXZ_gate)
			combined_faulty_Mtemp =  np.matmul(np.kron(ZXZ_gate_I, np.eye(2)),faultfree_M )
			combined_faulty_M = np.matmul(combined_faulty_Mtemp,np.kron(ZXZ_gate, np.eye(2)))
		elif self.cnot_type == 'target':  
			ZXZ_gate_I = np.linalg.inv(ZXZ_gate)
			combined_faulty_Mtemp =  np.matmul(np.kron(np.eye(2),ZXZ_gate_I),faultfree_M )
			combined_faulty_M = np.matmul(combined_faulty_Mtemp,np.kron( np.eye(2),ZXZ_gate))
		else:
			ZXZ_gate_I = np.linalg.inv(ZXZ_gate)
			combined_faulty_Mtemp =  np.matmul(np.kron( ZXZ_gate_I,ZXZ_gate_I),faultfree_M)
			combined_faulty_M = np.matmul(combined_faulty_Mtemp,np.kron(ZXZ_gate,ZXZ_gate))
		#return qGate.UGate(wrap_to_2pi(faultfreeGate.params[0]) * ratio, faultfreeGate.params[1], faultfreeGate.params[2]) # ratio fault on theta
		combined_faulty_gate = UnitaryGate(combined_faulty_M)
		self.combined_faulty_M = combined_faulty_M
		#print(combined_faulty_gate)
		return combined_faulty_gate
	#def createFaultyMatrix(self, faultfreeGate

generator = QATG(maxTestTemplateSize=100,\
				  	circuitSize =1, basisGateSet = [qGate.SXGate,qGate.IGate,qGate.XGate,qGate.RZGate,qGate.UGate],\
					circuitInitializedStates = {1:[1,0],2:[1,0,0,0],3:[1,0,0,0,0,0,0,0]}, \
				  	minRequiredStateFidelity = 0.4,verbose=False)
configurationList = generator.createTestConfiguration([RotationFault(qGate.XGate,0,0.75-np.pi/2,1,0.75+2*np.pi/2)])#1.5,1

for configuration in configurationList:
    print(configuration)
    #configuration.circuit.draw('mpl')

