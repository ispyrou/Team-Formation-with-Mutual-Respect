import numpy as np

def compute_score(solution_dict, w_weights):
	player_set = [x for x in solution_dict.values()]
	score = 0
	for skill, current_player in solution_dict.items():
		for other_player in player_set:
			if current_player != other_player:
				score += w_weights[skill][current_player] - w_weights[skill][other_player]
	return score

def individual_weight_calculation_rank(nba_dataset_ranking, included_list):
	w = {}
	for category in included_list:
		ranking = nba_dataset_ranking[category][:]
		num_of_nodes = len(ranking)
		for i in range(num_of_nodes):
			if category not in w:
				w[category] = {}
			w[category][ranking[i][0]] = len(ranking) - (i+1)
	return w

def individual_weight_calculation_graph(graphs, current_categories_list):
	w = {}
	for category in current_categories_list:
		graph = graphs[category]
		edges = list(graph.edges)
		data = list(graph.edges(data=True))
		nodes = list(graph.nodes)
		w[category] = {}
		for node in nodes:
			w[category][node] = 0
		for edge in data:
			w[category][edge[0]] += edge[2]['weight']
	return w
	
def whole_weight_calculation(categories, w):
	v = {}
	for category_1 in categories:
		for node in w[category_1]:
			if category_1 not in v:
				v[category_1] = {}
			v[category_1][node] = len(categories)*w[category_1][node]
			for category_2 in categories:
				v[category_1][node] -= w[category_2][node]
	return v

def greedy_algorithm(dict_of_weights, w_weights):
	greedy_list = []
	for skill, other in dict_of_weights.items():
		for worker, weight in other.items():
			greedy_list.append((weight,worker,skill))
	greedy_list.sort(key=lambda tup: tup[0], reverse=True)

	assigned = {}
	filled = []
	chosen = []
	for entry in greedy_list:
		if entry[1] not in chosen and entry[2] not in filled:
			assigned[entry[2]] = (entry[1],entry[0])
			filled.append(entry[2])
			chosen.append(entry[1])
		if len(assigned) == len(dict_of_weights):
			break
	for category in assigned:
		assigned[category] = assigned[category][0]
	score = compute_score(assigned, w_weights)
	return assigned, score

def min_zero_row(zero_mat, mark_zero):

	'''
	The function can be splitted into two steps:
	#1 The function is used to find the row which containing the fewest 0.
	#2 Select the zero number on the row, and then marked the element corresponding row and column as False
	'''

	#Find the row
	min_row = [99999, -1]

	for row_num in range(zero_mat.shape[0]): 
		if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
			min_row = [np.sum(zero_mat[row_num] == True), row_num]

	# Marked the specific row and column as False
	zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
	mark_zero.append((min_row[1], zero_index))
	zero_mat[min_row[1], :] = False
	zero_mat[:, zero_index] = False

def mark_matrix(mat):

	'''
	Finding the returning possible solutions for LAP problem.
	'''

	#Transform the matrix to boolean matrix(0 = True, others = False)
	cur_mat = mat
	zero_bool_mat = (cur_mat == 0)
	zero_bool_mat_copy = zero_bool_mat.copy()

	#Recording possible answer positions by marked_zero
	marked_zero = []
	while (True in zero_bool_mat_copy):
		min_zero_row(zero_bool_mat_copy, marked_zero)

	#Recording the row and column positions seperately.
	marked_zero_row = []
	marked_zero_col = []
	for i in range(len(marked_zero)):
		marked_zero_row.append(marked_zero[i][0])
		marked_zero_col.append(marked_zero[i][1])

	#Step 2-2-1
	non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))

	marked_cols = []
	check_switch = True
	while check_switch:
		check_switch = False
		for i in range(len(non_marked_row)):
			row_array = zero_bool_mat[non_marked_row[i], :]
			for j in range(row_array.shape[0]):
				#Step 2-2-2
				if row_array[j] == True and j not in marked_cols:
					#Step 2-2-3
					marked_cols.append(j)
					check_switch = True

		for row_num, col_num in marked_zero:
			#Step 2-2-4
			if row_num not in non_marked_row and col_num in marked_cols:
				#Step 2-2-5
				non_marked_row.append(row_num)
				check_switch = True
	#Step 2-2-6
	marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

	return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
	cur_mat = mat
	non_zero_element = []

	#Step 4-1
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					non_zero_element.append(cur_mat[row][i])
	min_num = min(non_zero_element)

	#Step 4-2
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					cur_mat[row, i] = cur_mat[row, i] - min_num
	#Step 4-3
	for row in range(len(cover_rows)):  
		for col in range(len(cover_cols)):
			cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
	return cur_mat

def ans_calculation(mat, pos, skills, workers, w_weights):
	ans_pos_final = []
	for i in range(len(pos)):
		if pos[i][1] < len(skills):
			ans_pos_final.append(pos[i])
#	total = 0
	ans_final = {}
	for i in range(len(ans_pos_final)):
		ans_final[skills[ans_pos_final[i][1]]] = workers[ans_pos_final[i][0]]
#		total += mat[ans_pos_final[i][0], ans_pos_final[i][1]]
	total = compute_score(ans_final, w_weights)
	return ans_final, total

def hungarian_algorithm(cost_mat, profit_mat, skills, workers, w_weights): 
	dim = cost_mat.shape[0]
	cur_mat = cost_mat

	#Step 1 - Every column and every row subtract its internal minimum
	for row_num in range(cost_mat.shape[0]): 
		cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])

	for col_num in range(cost_mat.shape[1]): 
		cur_mat[:,col_num] = cur_mat[:,col_num] - np.min(cur_mat[:,col_num])
	zero_count = 0
	while zero_count < dim:
		#Step 2 & 3
		ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
		zero_count = len(marked_rows) + len(marked_cols)

		if zero_count < dim:
			cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

	solution, score = ans_calculation(profit_mat, ans_pos, skills, workers, w_weights)

	return solution, score