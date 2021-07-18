import math
import random
def get_energy_for_single_qubit(self , parameter_list , faulty_matrix , faultfree_matrix , faulty_quantum_state , faultfree_quantum_state , fault):
	return vector_distance(
			matrix_operation([U3(parameter_list), faultfree_matrix], faultfree_quantum_state, max_size=2), 
			matrix_operation([U3(self.faulty_activation_gate(fault, parameter_list)), faulty_matrix], faulty_quantum_state, max_size=2))


def annealing_for_single_qubit(self , parameter_list , faulty_matrix , faultfree_matrix , faulty_quantum_state , faultfree_quantum_state , fault):
	best_parameter_list = [0] * len(parameter_list)
	curr_energy = self.get_energy_for_single_qubit(best_parameter_list , faulty_matrix , faultfree_matrix , faulty_quantum_state , faultfree_quantum_state , fault)
	accept = True
	r = 0.998
	# 考慮反向vector，所以diff最大為2
	T = 2.0
	T_min = 0.8
	# trial time上限取200
	trail_time = 0
	for i in range(len(parameter_list)):
		# 對每個parameter重複試驗200次
		# energy一直疊上去
		while trail_time < 200:
			best_parameter_list[i] += self.step
			up_energy = self.get_energy_for_single_qubit(best_parameter_list , faulty_matrix , faultfree_matrix , faulty_quantum_state , faultfree_quantum_state , fault)
			parameter_list[i] -= 2 * self.step
			down_energy = self.get_energy_for_single_qubit(best_parameter_list , faulty_matrix , faultfree_matrix , faulty_quantum_state , faultfree_quantum_state , fault)
			parameter_list[i] += self.step

			#判斷up_energy or down_energy 是否大於curr_energy
			if(up_energy > curr_energy || down_energy > curr_energy):
				accept = True
			else:
				# up_energy < curr_energy
				prob_accept = math.exp((up_energy - curr_energy) / T)
				if(prob_accept > random.random()):
					accept = True
				else:
					accept = False

			if accept == True:
				# 比原解更好的解
				if(up_energy > curr_energy || down_energy > curr_energy):
					if up_energy >= down_energy:
						best_parameter_list[i] += self.step
						curr_energy = up_energy
						self.step /= 0.99
					else:
						best_parameter_list[i] -= self.step
						curr_energy = down_energy
						self.step /= 0.99
				# 接受不好的解中相對不好的解
				else:
					
					if up_energy >= down_energy:
						best_parameter_list[i] -= self.step
						curr_energy = down_energy
					else:
						best_parameter_list[i] += self.step
						curr_energy = up_energy
			trail_time++
			T /= r
	parameter_list = best_parameter_list
	return



