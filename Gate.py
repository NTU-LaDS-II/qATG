import numpy as np
import math

def U3(parameter_list):
	# theda=0, phi=0, lam=0
	return np.array(
			[[
				np.cos(parameter_list[0] / 2),
				-np.exp(1j * parameter_list[2]) * np.sin(parameter_list[0] / 2)
			],
			 [
				 np.exp(1j * parameter_list[1]) * np.sin(parameter_list[0] / 2),
				 np.exp(1j * (parameter_list[1] + parameter_list[2])) * np.cos(parameter_list[0] / 2)
			 ]],
			dtype=complex)

def CNOT():
	return  np.array([[1,0,0,0],
					  [0,0,0,1],
					  [0,0,1,0],
					  [0,1,0,0]])
