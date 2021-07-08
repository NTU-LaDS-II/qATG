
def gat_params_list(i):
	ratio_list
	bias_list
	threshold_list
	if i == 0:
		ratio_list = [[1-r]]
		bias_list = [[r*np.pi]]
		threshold_list = [[1.5*np.pi]]
	elif i == 1:
		ratio_list = [[1-r, 1], [1, 1-r]]
		bias_list = [[r*np.pi, 0], [0, r*np.pi]]
		threshold_list = [[1.5*np.pi, 2*np.pi], [2*np.pi, 1.5*np.pi]]
	else:
		ratio_list = [[1-r, 1, 1], [1, 1-r, 1], [1, 1, 1-r]]
		bias_list = [[r*np.pi, 0, 0], [0, r*np.pi, 0], [0, 0, r*np.pi]]
		threshold_list = [[1.5*np.pi, 2*np.pi, 2*np.pi], [2*np.pi, 1.5*np.pi, 2*np.pi], [2*np.pi, 2*np.pi, 1.5*np.pi]]
	return ratio_list , bias_list , threshold_list

def get_fault_list(self , coupling_map):
	single_fault_list = []
 	two_fault_list = []

 	#first insert single_fault_list
 	for i in range(3):
 		ratio_list , bias_list , threshold_list = get_params_list(i)

 		
 	    for ratio in ratio_list:
            U_v_fault = []
            for qb in range(self.circuit_size):
                U_v_fault.append(U_variation_fault([qb], ratio=ratio))
            single_fault_list.append(U_v_fault)


        for bias in bias_list:
            U_v_fault = []
            for qb in range(self.circuit_size):
                U_v_fault.append(U_variation_fault([qb], bias=bias))
            single_fault_list.append(U_v_fault)

        for threshold in threshold_list:
            U_t_fault = []
            for qb in range(self.circuit_size):
                U_t_fault.append(U_threshold_lopa([qb], threshold=threshold))
            single_fault_list.append(U_t_fault)


        value = 0.05*np.pi
        f = [[value, value, value, value, value, value], [value, value, -value, value, value, -value], [value, -value, value, value, -value, value] , [value, -value, -value, value, -value, -value],
            [-value, value, value, -value, value, value], [-value, value, -value, -value, value, -value], [-value, -value, value, -value, -value, value] , [-value, -value, -value, -value, -value, -value]]
        # f = [[value, value, value, value, value, value]]
        for value in f:
        # for value in [[0.19, -0.17, 0.13, -0.22, 0.18, -0.15]]:
            one_type_fault = []
            drop_fault = [] 
            while len(drop_fault) != len(coupling_map):
                CNOT_v_fault = []
                for i in range(len(coupling_map)):
                    if coupling_map[i] in drop_fault:
                        continue
                    else:
                        push_fault = True
                        for fault in CNOT_v_fault:
                            if fault.index[0] in coupling_map[i] or fault.index[1] in coupling_map[i]:
                                push_fault = False
                                break
                        if push_fault:
                            CNOT_v_fault.append(CNOT_variation_fault(coupling_map[i], value=value))
                            drop_fault.append(coupling_map[i])
                one_type_fault.append(CNOT_v_fault)
            # for i in range(len(coupling_map)):
            #     one_type_fault.append([CNOT_variation_fault(coupling_map[i], value=value)])
            two_fault_list.append(deepcopy(one_type_fault))
        return (single_fault_list, two_fault_list)