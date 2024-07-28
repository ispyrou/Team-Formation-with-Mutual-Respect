import networkx as nx
import numpy as np
import cvxpy as cp
import pandas as pd
import json
import random
import time
import sys
import pickle
import os.path

def create_skill_author_index(category_rankings):
	skill_author_index_dict = {}
	for category, category_ranking in category_rankings.items():
		for ranking, author_score in enumerate(category_ranking):
			author = author_score[0]
			tup = (category,author)
			if tup not in skill_author_index_dict:
				skill_author_index_dict[tup] = ranking

	return skill_author_index_dict

def max_score_algo(category_rankings, weighted):
	F_category_author = {};
	F_author_category = {}; 
	D = {}
	solution_dict = {}; selected_authors = {}
	available_categories = {}
	num_of_skills = len(category_rankings.keys())
	num_of_workers = len(list(category_rankings.items())[0][1])
	solution_size = len(F_category_author)

	category_rankings_copy = dict(category_rankings)
	num_iterations = 0

	for category,category_ranking in category_rankings_copy.items():
		available_categories[category] = 1

	while solution_size < num_of_skills:
		category = random.choice(list(available_categories.keys()))

		category_ranking = category_rankings_copy[category]
		for ranking, author_score in enumerate(category_ranking):
			author = author_score[0]
			# print('Category:',category,'author:',author,'ranking:',ranking)
			if author in F_author_category:
				remove_category = F_author_category[author]
				# print('Removing category:',remove_category,'because of author:',author,'but also in category:',category)
				available_categories[remove_category] = 1
				D[author] = 1
				del F_category_author[remove_category]
				del F_author_category[author]

			elif (author not in F_author_category) and (author not in D):
				# print('Adding author:',author,'to category:',category)
				F_category_author[category] = (author,ranking)
				F_author_category[author] = category
				# print('Current solution is:',F_category_author)
				del available_categories[category]
				break

		solution_size = len(F_category_author)
		# print('solution size is:',solution_size,'length D:',len(D),'num workers:',num_of_workers)
		if len(D) >= num_of_workers-1:
			break

	# print('Solution is:',F_category_author)
	return F_category_author

def compute_ranking_score(solution_dict,category_ranking,skill_author_index_dict):
	score = 0
	indegree = {}
	outdegree = {}
	# print('Skill author index:',skill_author_index_dict)
	authors_set = [x[0] for x in solution_dict.values()]
	print(authors_set)
	for category, author_score in solution_dict.items():
		author_to = author_score[0];
		idx_author_to = skill_author_index_dict[(category,author_to)]
		for author_from in authors_set:
			idx_author_from = skill_author_index_dict[(category,author_from)]
			if author_to != author_from and idx_author_to < idx_author_from:
				if author_from not in outdegree:
					outdegree[author_from] = 0
				outdegree[author_from] += 1

				if author_to not in indegree:
					indegree[author_to] = 0
				indegree[author_to] += 1

				score += 1
	return score

def create_weighted_matrices(nba_dataset_ranking, included_list):
	weights = {}
	Weighted_m = {}
	for category in included_list:
		ranking = nba_dataset_ranking[category]
		nba_dataset_ranking_sorted_by_name = nba_dataset_ranking[category][:]
		nba_dataset_ranking_sorted_by_name.sort(key=lambda elem: elem[0])
		num_of_nodes = len(nba_dataset_ranking_sorted_by_name)
		for i in range(len(nba_dataset_ranking_sorted_by_name)):
			if category not in weights:
				weights[category] = {}
			weights[category][nba_dataset_ranking_sorted_by_name[i]] = len(nba_dataset_ranking_sorted_by_name) - nba_dataset_ranking_sorted_by_name[i][1]
		W = np.zeros((num_of_nodes, num_of_nodes))
		for i in range(num_of_nodes):
			for j in range(num_of_nodes):
				if i != j:
					if weights[category][nba_dataset_ranking_sorted_by_name[i]] < weights[category][nba_dataset_ranking_sorted_by_name[j]]:
						W[i][j] = weights[category][nba_dataset_ranking_sorted_by_name[i]] - weights[category][nba_dataset_ranking_sorted_by_name[j]]
		Weighted_m[category] = W
	return Weighted_m

def create_adjacency_matrices(nba_dataset_ranking, included_list):
	Adjacency_m = {}
	for category in included_list:
		ranking = nba_dataset_ranking[category]
		nba_dataset_ranking_sorted_by_name = nba_dataset_ranking[category][:]
		nba_dataset_ranking_sorted_by_name.sort(key=lambda elem: elem[0])
		num_of_nodes = len(nba_dataset_ranking_sorted_by_name)
		A = np.zeros((num_of_nodes, num_of_nodes))
		for i in range(num_of_nodes):
			for j in range(num_of_nodes):
				if i != j:
					if nba_dataset_ranking_sorted_by_name[i][1] > nba_dataset_ranking_sorted_by_name[j][1]:
						A[i][j] = 1
		Adjacency_m[category] = A
	return Adjacency_m

def create_matrix_m(Adjacency_m, included_list):
	col = []
	for category in included_list:
		matrix = Adjacency_m[category]
		num_of_nodes = len(matrix)
		for i in range(num_of_nodes):
			col.append([matrix[i,0]])

	M = np.array(col)

	for i in range(1, num_of_nodes*len(Adjacency_m)):
		if i >= num_of_nodes:
			j = i%num_of_nodes
		else:
			j = i
		col = []
		for category, matrix in Adjacency_m.items():
			for k in range(num_of_nodes):
				col.append([matrix[k,j]])
		M = np.append(M, col, 1)

	for k in range(len(Adjacency_m)):
		for i in range(num_of_nodes):
			for j in range(num_of_nodes):
				M[(k*num_of_nodes)+i][(k*num_of_nodes)+j] = 0

	return M

def create_constraint_matrices(Adjacency_m):
	num_of_nodes = len(list(Adjacency_m.items())[0][1])
	num_of_categories = len(Adjacency_m)
	B = np.zeros((num_of_categories, num_of_categories*num_of_nodes))
	for i in range(num_of_categories):
		for j in range(i*num_of_nodes, (i*num_of_nodes)+num_of_nodes):
			B[i,j] = 1

	C = np.zeros((num_of_nodes, num_of_categories*num_of_nodes))
	for i in range(num_of_nodes):
		for j in range(i,num_of_nodes*num_of_categories,num_of_nodes):
			C[i,j] = 1
	return B, C

def iqp_rank_algorithm(M, B, C, nba_dataset_ranking, included_list, skill_author_index_dict):
	x = cp.Variable(len(C)*len(B), boolean=True)
#	M = 0.5 * (M * np.transpose(M))
	M = M + np.transpose(M)
	
	objective = cp.Maximize(cp.quad_form(x, M))

	constraints = [np.all((B @ x) == 1), np.all((C @ x) <= 1)]

	prob = cp.Problem(objective, constraints)

#	prob.solve(solver=cp.CPLEX, verbose=True)
#	prob.solve(solver=cp.MOSEK, verbose=True)
	prob.solve(solver=cp.GUROBI, verbose=True, TimeLimit=1000)

	solution = x.value

	nodes = nba_dataset_ranking[included_list[0]][:]
	nodes.sort(key=lambda elem: elem[0])
	num_of_nodes = len(nodes)
	skill_sol = []
	solution_dict = {}
	for i in range(0, len(solution), num_of_nodes):
		skill_sol.append(solution[i:i + num_of_nodes])

	for i in range(len(included_list)):
		for j in range(len(skill_sol[i])):
			if skill_sol[i][j] == 1:
				solution_dict[included_list[i]] = nodes[j]

	score = compute_ranking_score(solution_dict, nba_dataset_ranking, skill_author_index_dict)

	return solution_dict, score

def main():
	included_dict = {'3P':1,'TRB':1,'AST':1,'STL':1,'BLK':1,'FG':1,'2P':1,'FT':1,'OBPM':1,'DBPM':1,'VORP':1}


	file_path = 'Seasons_Stats.csv'
	num_of_nodes_list = []; num_of_skills_list = []; algorithm_list = []; score_list = []; solution_list = []
	time_list = []; loop_time_list = []; year_list = []
	weighted = False
	filename = 'nba.csv'

	if weighted == True:
		file_name_results = 'results_method1_nba_weighted.pkl'
	else:
		file_name_results = 'results_method1_nba.pkl'

	random.seed(1)
	for year in range(2010,2018,1):
		year_list.append(year); year_list.append(year); year_list.append(year)
		total_start_time = time.time()
		print('year is:',year)
		nba_dataset_ranking = {}

		df = pd.read_csv(file_path)
		df = df.loc[df['Year'] == year]
		df = df.groupby(['Player','Age']).mean().reset_index()
		df = df.loc[df['G'] >= 27]
		df['MPG'] = df['MP']/df['G']
		df = df.loc[df['MPG'] >= 15]
		df = df.fillna(0)
		# df = df.head(60)
		print('For year:',year,'number of players is:',df.size,len(df))
		category_names = list(df.columns.values)
		names = df['Player'].tolist()

		for i,category_name in enumerate(category_names):
			if category_name in included_dict:
				category_list = df[category_name].tolist()
				category_ranking = []
				for j,value in enumerate(category_list):
					tup = (names[j],value)
					category_ranking.append(tup)

				# sort based on value from highest to lower
				category_ranking.sort(key=lambda elem: elem[1], reverse=True)
				nba_dataset_ranking[category_name] = category_ranking

		num_of_nodes = len(names)
		num_of_skills = len(nba_dataset_ranking.keys())

		skill_author_index_dict = create_skill_author_index(nba_dataset_ranking)

		max_score_solution = max_score_algo(nba_dataset_ranking,weighted)

		# print('Number of elements in max score solution is:',len(max_score_solution))
		if max_score_solution != None:
			max_score = compute_ranking_score(max_score_solution,nba_dataset_ranking,skill_author_index_dict)
			print('Max score is:',max_score)
			print('Max score solution:',max_score_solution)
		else:
			print('There is no optimal solution.')
			
		included_list = []
		for category, occ in included_dict.items():
			included_list.append(category)
		
		if weighted == True:
			# creating weighted matrices
			parts_of_m = create_weighted_matrices(nba_dataset_ranking, included_list)
		else:
			# creating adjacency matrices
			parts_of_m = create_adjacency_matrices(nba_dataset_ranking, included_list)

		# creating matrix M
		M = create_matrix_m(parts_of_m, included_list)
		print("created M")
		
		# creating constraint matrices
		B, C = create_constraint_matrices(parts_of_m)
		
		# solving
		loop_start_time = time.time()
		iqp_rank_solution, iqp_rank_score = iqp_rank_algorithm(M, B, C, nba_dataset_ranking, included_list, skill_author_index_dict)
		loop_elapsed_time_iqp_rank = time.time() - loop_start_time
		print('Solution is:', iqp_rank_solution)
		print('Solution score for iqp rank truth is:',iqp_rank_score)
		print('Running time is:',loop_elapsed_time_iqp_rank)
		'''
		dct = {'solution': iqp_rank_solution, 'score': iqp_rank_score, 'time': loop_elapsed_time_iqp_rank, 'edges': None}
		if os.path.isfile(file_name_results):
			with open(file_name_results, 'rb') as results:
				results_dict = pickle.load(results)
			with open(file_name_results, 'wb') as results:
				if year not in results_dict:
					results_dict[year] = dct
					pickle.dump(results_dict, results)
		else:
			results_dict = {}
			results_dict[year] = dct
			with open(file_name_results, 'wb') as results:
				pickle.dump(results_dict, results)
		'''


if __name__ == '__main__':
	main()
	print('Exiting main!')