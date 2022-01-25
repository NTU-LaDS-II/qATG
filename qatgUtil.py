import numpy as np

INT_MIN = 1E-100
INT_MAX = 1E15

def U3(parameterList):
	# theta=0, phi=0, lam=0
	return np.array(
			[[
				np.cos(parameterList[0] / 2),
				-np.exp(1j * parameterList[2]) * np.sin(parameterList[0] / 2)
			],
			 [
				 np.exp(1j * parameterList[1]) * np.sin(parameterList[0] / 2),
				 np.exp(1j * (parameterList[1] + parameterList[2])) * np.cos(parameterList[0] / 2)
			 ]],
			dtype=complex)

def matrixOperation(matrixList, quantumState=[]):
	matrixList = [np.array(matrix) for matrix in matrixList]
	quantumState = np.array(quantumState)
	for matrix in matrixList:
		quantumState = np.dot(matrix, quantumState)
	return quantumState
	
def vectorDistance(vector1, vector2):
	return np.sum(np.square(np.abs(np.subtract(toProbability(vector1), toProbability(vector2)))))

def toProbability(probability):
	return np.array(probability*np.conj(probability), dtype=float)

def calEffectSize(faultyQuantumState, faultfreeQuantumState):
	deltaSquare = np.square(faultyQuantumState - faultfreeQuantumState)
	effectSize = np.sum(deltaSquare / (faultyQuantumState + INT_MIN))
	effectSize = np.sqrt(effectSize)
	if effectSize < 0.1:
		effectSize = 0.1
	return effectSize
