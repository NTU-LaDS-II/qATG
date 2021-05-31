import numpy as np
import math

# def product_matrix(gate_list):
#     max_dim = 0
#     for gate in gate_list:
#         if gate.shape[0] > max_dim:
#             max_dim = gate.shape[0]


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

# print(
# np.dot(np.dot( np.kron(U3(1,2,3), np.eye(2)), CNOT()), np.kron(np.eye(2), U3(4,5,6))) 
# )


# print(
# np.dot( np.dot(np.dot( np.kron(U3(1,2,3), np.eye(2)), CNOT()), np.kron(np.eye(2), U3(4,5,6))), np.dot(np.dot( np.kron(U3(1,2,3), np.eye(2)), CNOT()), np.kron(np.eye(2), U3(4,5,6))) 
# )