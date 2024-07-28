from operator import itemgetter
import networkx as nx
import numpy as np
import cvxpy as cp
import json
import pickle
import csv
import time
import sys
import os.path

customized_categories_list = [
#{'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1},
#{'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1},
#{'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1,'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1},
#{'software engineering':1,'operating systems':1, 'high-performance computing': 1,'distributed and parallel computing':1},
{'signal processing':1,'wireless networks and mobile computing':1, 'computer networking':1, 'information retrieval': 1},
{'signal processing':1,'wireless networks and mobile computing':1, 'computer networking':1, 'information retrieval': 1,'software engineering':1,'operating systems':1, 'high-performance computing': 1,'distributed and parallel computing':1},
]

team_dictionairy = {'Team1': ['artificial intelligence','natural language processing','robotics', 'neural networks'], 'Team2': ['data mining','databases','algorithms and theory','algorithms'], 'Team3': ['data mining','databases','algorithms and theory','algorithms','artificial intelligence','natural language processing','robotics','neural networks'], 'Team4': ['software engineering','operating systems','high-performance computing','distributed and parallel computing'], 'Team5': ['signal processing','wireless networks and mobile computing','computer networking','information retrieval'], 'Team6': ['signal processing','wireless networks and mobile computing','computer networking','information retrieval','software engineering','operating systems','high-performance computing','distributed and parallel computing']}

def read_dict_to_list(file_name):
	print("in read_dict_to_list")

	# print('Reading txt file:',file_name)

	venue_list = json.load(open(file_name))

	# print('Reading completed.')
	print("out read_dict_to_list")
	return venue_list

def read_dict_to_pkl_file(file_name):
	print("in read_dict_to_pkl_file")
	# load hierarchical dictionary
	with open(file_name, 'rb') as f:
		name_dict = pickle.load(f)

	print("out read_dict_to_pkl_file")
	return name_dict

def read_venue_assignments_csv(file_name):
	print("in read_venue_assignments_csv")
	venue_cat_dict = {}
	with open(file_name, "rt", encoding="utf-8") as f:
		reader = csv.reader(f, delimiter="\t")
		for i, line in enumerate(reader):
			csv_line_split = line[0].strip().split(',')	
			venue_name = csv_line_split[0].strip(); cat_name = csv_line_split[2].strip();
			venue_cat_dict[venue_name] = cat_name 

	print("out read_venue_assignments_csv")
	return venue_cat_dict

def create_cat_venue_dictionary(venue_cat_dict):
	print("in create_cat_venue_dictionary")
	cat_venue_dict = {}

	for venue, cat in venue_cat_dict.items():
		if cat not in cat_venue_dict:
			cat_venue_dict[cat] = []
		cat_venue_dict[cat].append(venue)

	print("out create_cat_venue_dictionary")
	return cat_venue_dict

def get_authors(venue_author_dict,venue_cat_dict,customized_categories):
	print("in get_authors")
	'''
	Input: dictionary with venues and corresponding authors, dictionary with venues and corresponding categories
	Output: dictionary key: category value: authors who published in this category, set of authors that have published in the venues of the selected customized categoriess
	'''

	cat_authors_dict = {}
	authors_list = []
	for venue, category in venue_cat_dict.items():
		if category != 'category':
			if (category in customized_categories) and (venue in venue_author_dict):
				authors = venue_author_dict[venue].keys()
				if category not in cat_authors_dict:
					cat_authors_dict[category] = []
				cat_authors_dict[category] += authors
				cat_authors_dict[category] = list(set(cat_authors_dict[category]))

	for cat,authors in cat_authors_dict.items():
		authors_list += authors

	authors_set = list(set(authors_list))

	print("out get_authors")
	return cat_authors_dict, authors_set

def create_graph(category,cat_venue_dict,venue_author_citations_dict,venue_cat_dict,paper_id_authors_dict,paper_id_venue_dict,category_remaining_edges):
	print("in create_graph")
	'''
	Input:  name of graph category, 
			dictionary key: category val: list of venues, 
			dictionary key: venue key: author val: list of citations (paper ids),
			dictionary key: paper id and authors of the paper
			dictionary key: paper id and venue of the paper
	Output: a graph 
	'''
	# given the category of the graph
	# find the corresponding venues of this category from the cat, venues dictionary
	# for each of the venues, find authors that published in this venue - add the author as a node to the graph
	# 	for each author published in this venue find the citations the author has from this venue.
	# 		for each citation, i) see if paper that is being cited belongs to the same category, 1 i) if yes create an edge between author and authors of the paper, ii) if not continue


	DG=nx.DiGraph()
	edges = []
	category_venues_list = cat_venue_dict[category]
	for venue in category_venues_list:
		for from_author, citations in venue_author_citations_dict[venue].items():
			for citation in citations:
				if citation in paper_id_venue_dict:
					citation_venue = paper_id_venue_dict[citation]
					if citation_venue in venue_cat_dict:
						citation_category = venue_cat_dict[citation_venue]
						citation_authors = paper_id_authors_dict[citation]
						if citation_category == category:	
							for to_author in citation_authors:
								if DG.has_edge(from_author, to_author):
									DG[from_author.encode('ascii','ignore')][to_author.encode('ascii','ignore')]['weight'] += 1
								else:
									DG.add_edge(from_author.encode('ascii','ignore'), to_author.encode('ascii','ignore'), weight=1)
						else:
							if citation_category not in category_remaining_edges:
								category_remaining_edges[citation_category] = []
							for to_author in citation_authors:
								tup = (from_author.encode('ascii','ignore'),to_author.encode('ascii','ignore'))
								category_remaining_edges[citation_category].append(tup)

	print("out create_graph")
	return DG, category_remaining_edges

def add_cross_edges(DG,remaining_edges_list):

	for edge in remaining_edges_list:
		from_author = edge[0]; to_author = edge[1]

		if DG.has_edge(from_author, to_author):
			DG[from_author][to_author]['weight'] += 1
		else:
			DG.add_edge(from_author, to_author, weight=1)

	return DG

def create_graph_gml_files(file_name_pid_venue,file_name_venues_authors_citations,file_name_pid_authors,file_name_venues_assignment_csv,customized_categories):
	print("in create_graph_gml_files")
	# read paper id and venue .txt file
	paper_id_venue_dict = read_dict_to_list(file_name_pid_venue)

	# read hierarchical dictionary with venue, authors, citations to a file
	venue_author_citations_dict = read_dict_to_pkl_file(file_name_venues_authors_citations)

	# read paper id and corresponding authors from file
	paper_id_authors_dict = read_dict_to_pkl_file(file_name_pid_authors)

	# read venues with manually assigned category
	venue_cat_dict = read_venue_assignments_csv(file_name_venues_assignment_csv)

	# create a dictionary with categories and their corresponding venues
	cat_venue_dict = create_cat_venue_dictionary(venue_cat_dict)

	# get set of authors published in customized category venues
	cat_authors_dict, authors_set = get_authors(venue_author_citations_dict,venue_cat_dict,customized_categories)

	# dictionary that stores cross-edges that belong to different graphs
	category_remaining_edges = {}
	# dictionary that stores graph of each category
	category_graph = {}

	start_graph = time.time(); counter = 0
	print('Creating graphs.')
	# creating graph for each of the customized categories
	for category,category_selected in customized_categories.items():
		if customized_categories[category] == 1:
			DG,category_remaining_edges = create_graph(category,cat_venue_dict,venue_author_citations_dict,venue_cat_dict,paper_id_authors_dict,paper_id_venue_dict,category_remaining_edges)
			# storing graphs of each category in a dictionary
			category_graph[category] = DG
			counter+=1
			# if counter == 3:
			# 	break

	end_graph = time.time()
	# time duration of main function
	print('Time required to create graphhs is:',end_graph - start_graph,'sec')

	start_graph = time.time()
	print('Adding cross-edges to the graph.')
	for category, category_selected in customized_categories.items():
		print('Category is:',category)
		if customized_categories[category] == 1:
			DG = category_graph[category]; remaining_edges_list = category_remaining_edges[category]

			DG = add_cross_edges(DG,remaining_edges_list)
			category_graph[category] = DG
			print('New number of nodes of graph:',DG.number_of_nodes())
			print('New number of edges of graph:',DG.number_of_edges())

	end_graph = time.time()
	# time duration of main function
	print('Time required to add cross-edges is:',end_graph - start_graph,'sec')

	print("out create_graph_gml_files")
	
	return category_graph

def clean_graphs(category_graph,k_core,weighted):
	print("in clean_graphs")
	
	category_graph_clean = {}

	for category, graph in category_graph.items():
		ebunch = nx.selfloop_edges(graph)
		graph.remove_edges_from(ebunch)
		print('Creating core graph.')
		core_graph = nx.k_core(graph,k=k_core)
		if weighted == False:
			for u,v in core_graph.edges(data=False):
				core_graph[u][v]['weight'] = 1

		category_graph_clean[category] = core_graph
	print("out clean_graphs")
	return category_graph_clean

def get_overlapping_graph_nodes(category_graphs):
	print("in get_overlapping_graph_nodes")
	set_of_nodes = []
	category_num_nodes = []
	counter = 0
	for category,graph in category_graphs.items():
		graph_nodes = list(graph.nodes())
		if counter == 0:
			intersecting_nodes = set(graph_nodes)

		intersecting_nodes = intersecting_nodes.intersection(set(graph_nodes))
		counter += 1

	print('Number of intersecting nodes:',len(intersecting_nodes))
	print("out get_overlapping_graph_nodes")
	return intersecting_nodes

def reverse_graphs(category_graph_clean):
	reverse_category_graph_clean = {}
	for category, graph in category_graph_clean.items():
		category_node_list = list(graph.nodes())
		category_edge_list = list(graph.edges(data=True))
		reverse = nx.DiGraph()
		reverse.add_nodes_from(category_node_list)
		for edge in category_edge_list:
			reverse.add_edge(edge[1], edge[0], weight = edge[2]['weight'])
		reverse_category_graph_clean[category] = reverse
	return reverse_category_graph_clean

def compute_graph_score(solution_dict,category_graph):
	print("in compute graph score")
	score = 0
	existing_edges = []
	authors_set = [x for x in solution_dict.values()]

	for category, author_score in solution_dict.items():
		graph = category_graph[category]
		author_to = author_score
		for author_from in authors_set:
			if author_to != author_from and nx.has_path(graph, author_to, author_from):
					distance_ij = nx.shortest_path_length(graph, author_to, author_from)
			else:
				distance_ij = 0
			if author_to != author_from and nx.has_path(graph, author_from, author_to):
				distance_ji = nx.shortest_path_length(graph, author_from, author_to)
			else:
				distance_ji = 0
			score += distance_ij - distance_ji
			existing_edges.append((author_to,author_from))

	return score, existing_edges

def create_adjacency_matrices(reverse_category_graph_clean, current_categories_list):
	Adjacency_m = {}
	for category in current_categories_list:
		graph = reverse_category_graph_clean[category]
		nodes = list(graph.nodes)
		nodes.sort()
		num_of_nodes = graph.number_of_nodes()
		A = np.zeros((num_of_nodes, num_of_nodes))
		for i in range(num_of_nodes):
			for j in range(i, num_of_nodes):
				if nx.has_path(graph, nodes[i], nodes[j]):
					distance_ij = nx.shortest_path_length(graph, nodes[i], nodes[j])
				else:
					distance_ij = 0
				if nx.has_path(graph, nodes[j], nodes[i]):
					distance_ji = nx.shortest_path_length(graph, nodes[j], nodes[i])
				else:
					distance_ji = 0
				A[i][j] = distance_ij - distance_ji
				A[j][i] = distance_ji - distance_ij
		Adjacency_m[category] = A
	return Adjacency_m

def create_weighted_matrices(reverse_category_graph_clean, current_categories_list):
	Weighted_m = {}
	for category in current_categories_list:
		graph = reverse_category_graph_clean[category]
		edges = list(graph.edges)
		data = list(graph.edges(data=True))
		nodes = list(graph.nodes)
		nodes.sort()
		num_of_nodes = graph.number_of_nodes()
		weights = {}
		for i in range(len(edges)):
			weights[edges[i]] = data[i][2]['weight']
		W = np.zeros((num_of_nodes, num_of_nodes))
		for i in range(num_of_nodes):
			for j in range(i, num_of_nodes):
				edge = (nodes[i], nodes[j])
				reve = (nodes[j], nodes[i])
				if edge in weights and not reve in weights:
					W[i][j] = weights[edge]
					W[j][i] = -weights[edge]
				elif reve in weights and not edge in weights:
					W[i][j] = -weights[reve]
					W[j][i] = weights[reve]
				elif edge in weights and reve in weights:
					W[i][j] = weights[edge] - weights[reve]
					W[j][i] = weights[reve] - weights[edge]
		Weighted_m[category] = W
	return Weighted_m

def create_matrix_m(Adjacency_m, current_categories_list):
	col = []
	for category in current_categories_list:
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

def iqp_graph_algorithm(M, B, C, category_graphs, current_categories_list):
	x = cp.Variable(len(C)*len(B), boolean=True)
#	M = 0.5 * (M * np.transpose(M))
	M = M + np.transpose(M)
		
	objective = cp.Maximize(cp.quad_form(x, M))

	constraints = [np.all((B @ x) == 1), np.all((C @ x) <= 1)]

	prob = cp.Problem(objective, constraints)

#	prob.solve(solver=cp.CPLEX, verbose=True)
#	prob.solve(solver=cp.MOSEK, verbose=True)
	prob.solve(solver=cp.GUROBI, verbose=True, TimeLimit=43200)

	solution = x.value

	nodes = list(category_graphs[current_categories_list[0]].nodes)
	nodes.sort()
	num_of_nodes = len(nodes)
	skill_sol = []
	solution_dict = {}
	for i in range(0, len(solution), num_of_nodes):
		skill_sol.append(solution[i:i + num_of_nodes])

	for i in range(len(current_categories_list)):
		for j in range(len(skill_sol[i])):
			if skill_sol[i][j] == 1:
				solution_dict[current_categories_list[i]] = nodes[j]

	score, edges = compute_graph_score(solution_dict, category_graphs)

	return solution_dict, score, edges

def main():
	#file_path = args[0]; file_names = args[1:]; 

	file_name_venues_authors_citations = "venues_authors_citations.pkl"
	file_name_pid_authors = "pid_authors.pkl"
	file_name_pid_venue = "pid_venue.txt"
	file_name_venue_paper_counts = "venue_paper_counts.txt.txt"
	file_name_venues_assignment_csv = "venues_cat_assign_refined.csv"
	kcore = 5
	weighted = False
	
	if weighted == True:
		file_name_results = 'results_antisymmetric_weighted_dblp.pkl'
	else:
		file_name_results = 'results_antisymmetric_dblp_v2.pkl'


	# start time of main function
	start = time.time()

	num_of_skills_list = []; 
	score_graph_greedy_list = []; score_graph_ranking_list = []; score_graph_random_list = [];
	solution_graph_greedy_list = []; solution_graph_ranking_list = []; solution_graph_random_list = []
	time_list = []; loop_graph_greedy_time_list = []; loop_graph_ranking_time_list = []; loop_graph_random_time_list = [];
	max_score_list = []; categories_list = [] 

	for customized_categories in customized_categories_list:
		nodes = []
		edges = []
		max_endorsement = -float("inf")
		nodes_overlap = []
		total_start_time = time.time()
		max_score = (len(customized_categories))*(len(customized_categories)-1)
		num_of_skills = len(customized_categories)
		print('Considering the following customized categories:',customized_categories)
		# uncomment to create and store category graphs
		print('Creating graphs.')
		category_graph = create_graph_gml_files(file_name_pid_venue,file_name_venues_authors_citations,file_name_pid_authors,file_name_venues_assignment_csv,customized_categories)
		print('Cleaning graphs.')
		category_graph_clean = clean_graphs(category_graph,kcore,weighted)

		for category, graph in category_graph_clean.items():
			# print('Category:',category,sorted(graph.in_degree(), key=lambda x: x[1], reverse=True)[:1])
			nodes_list = list(graph.nodes())
			edge_list = list(graph.edges())
			print('Length of nodes list:',len(nodes_list),'length of nodes',len(nodes))
			nodes = nodes + nodes_list
			nodes_overlap = list(set(nodes) & set(nodes_list))
			# nodes_overlap = set(nodes_overlap).intersection(nodes_list)
			edges = edges + edge_list
			nd = sorted(list(graph.degree),key=itemgetter(1),reverse=True)[0]
			in_degree = list(dict(graph.in_degree(nd)).values())[0]
			if in_degree > max_endorsement:
				max_endorsement = in_degree

		print('Number of unique nodes',len(set(nodes)),'number of all nodes:',len(nodes))
		print('Avg End./Role:',len(edges)/len(category_graph_clean))
		print('Avg End./Expert:',len(edges)/len(set(nodes)))
		print('Max End./Expert',max_endorsement)
		print('Number of overlapping nodes:',len(set(nodes_overlap)))

		# remove non-overlapping nodes from graphs
		overlapping_nodes = get_overlapping_graph_nodes(category_graph_clean)
		for category, graph in category_graph_clean.items():
			nodes_list = list(graph.nodes())
			for node in nodes_list:
				if node not in overlapping_nodes:
					graph.remove_node(node)

		current_categories_list = []
		for category, occ in customized_categories.items():
			current_categories_list.append(category)

		# reversing graphs to fit new problem preliminaries
		reverse_category_graph_clean = reverse_graphs(category_graph_clean)
				
		if weighted == True:
			# creating weighted matrices
			parts_of_m = create_weighted_matrices(reverse_category_graph_clean, current_categories_list)
		else:
			# creating adjacency matrices
			parts_of_m = create_adjacency_matrices(reverse_category_graph_clean, current_categories_list)

		# creating matrix M
		M = create_matrix_m(parts_of_m, current_categories_list)

		# creating constraint matrices
		B, C = create_constraint_matrices(parts_of_m)
		
		# solving
		loop_start_time = time.time()
		iqp_graph_solution, iqp_graph_score, iqp_graph_edges = iqp_graph_algorithm(M, B, C, reverse_category_graph_clean, current_categories_list)
		loop_elapsed_time_iqp_graph = time.time() - loop_start_time
		print('Solution is:', iqp_graph_solution)
		print('Solution score for iqp graph truth is:',iqp_graph_score)
		print('Edges are:', iqp_graph_edges)
		print('Running time is:',loop_elapsed_time_iqp_graph)

		for teams, lst in team_dictionairy.items():
			if lst == current_categories_list:
				team = teams
		
		dct = {'solution': iqp_graph_solution, 'score': iqp_graph_score, 'time': loop_elapsed_time_iqp_graph, 'edges': iqp_graph_edges}
		if os.path.isfile(file_name_results):
			with open(file_name_results, 'rb') as results:
				results_dict = pickle.load(results)
			with open(file_name_results, 'wb') as results:
				if team not in results_dict:
					results_dict[team] = dct
					pickle.dump(results_dict, results)
		else:
			results_dict = {}
			results_dict[team] = dct
			with open(file_name_results, 'wb') as results:
				pickle.dump(results_dict, results)
		
#		break


if __name__ == '__main__':
	main()
	print('Exiting main!')