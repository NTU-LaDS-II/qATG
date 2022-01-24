import numpy as np
import math

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

def CNOT():
	return  np.array([[1,0,0,0],
					  [0,0,0,1],
					  [0,0,1,0],
					  [0,1,0,0]])

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
