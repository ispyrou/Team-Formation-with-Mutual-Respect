from operator import itemgetter
import networkx as nx
import numpy as np
import json
import pickle
import csv
import time
import copy
import os
import random

customized_categories_list = [
{'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1},
{'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1},
{'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1,'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1},
{'software engineering':1,'operating systems':1, 'high-performance computing': 1,'distributed and parallel computing':1},
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
    with open(file_name, "rt") as f:
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

def whole_weight_calculation(workers, skills, w):
    v = {}
    for skill in skills:
        for worker in workers[skill]:
            if skill not in v:
                v[skill] = {}
            v[skill][worker] = len(skills)*w[skill][worker]
            for skill_2 in skills:
                v[skill][worker] -= w[skill_2][worker]
    return v  

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

def hungarian_algorithm(cost_mat, profit_mat, skills, workers): 
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

    solution, score = ans_calculation(profit_mat, ans_pos, skills, workers)

    return solution, score

def ans_calculation(mat, pos, skills, workers):
    ans_pos_final = []
    for i in range(len(pos)):
        if pos[i][1] < len(skills):
            ans_pos_final.append(pos[i])
    total = 0
    ans_final = {}
    for i in range(len(ans_pos_final)):
        ans_final[skills[ans_pos_final[i][1]]] = workers[ans_pos_final[i][0]]
        total += mat[ans_pos_final[i][0], ans_pos_final[i][1]]
    return ans_final, total

def compute_score(solution_dict):
    score = 0
    for skill, player in solution_dict.items():
        score += player[1]
    return score

def compute_greedy_score(solution_dict, graphs_dict, landmarks):
    score = 0
    for skill, graph in graphs_dict.items():
        landmark = landmarks[skill]
        player_out = solution_dict[skill][0]
        for key, playerin in solution_dict.items():
            player_in = playerin[0]
            if player_out != player_in:
                if nx.has_path(graph, player_out, player_in):
                    distance_ij = nx.shortest_path_length(graph, player_out, player_in)
                else:
                    distance_ij = 0
                if nx.has_path(graph, player_in, player_out):
                    distance_ji = nx.shortest_path_length(graph, player_in, player_out)
                else:
                    distance_ji = 0
                score += distance_ij - distance_ji
    return score

def compute_hungarian_score(solution_dict, graphs_dict, landmarks):
    score = 0
    for skill, graph in graphs_dict.items():
        landmark = landmarks[skill]
        player_out = solution_dict[skill]
        for key, player_in in solution_dict.items():
            if player_out != player_in:
                if nx.has_path(graph, player_out, player_in):
                    distance_ij = nx.shortest_path_length(graph, player_out, player_in)
                else:
                    distance_ij = 0
                if nx.has_path(graph, player_in, player_out):
                    distance_ji = nx.shortest_path_length(graph, player_in, player_out)
                else:
                    distance_ji = 0
                score += distance_ij - distance_ji
    return score
        

def greedy_algorithm(dict_of_weights):
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

    score = compute_score(assigned)
    return assigned, score

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
        file_name_results = 'results_landmarks_weighted_dblp.pkl'
    else:
        file_name_results = 'results_landmarks_dblp.pkl'


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

        reverse_category_graph_clean = reverse_graphs(category_graph_clean)

        copy_of_full_graphs = copy.deepcopy(reverse_category_graph_clean)
        count = 0
        max_greedy_score = -float('inf')
        best_greedy_solution = {}
        best_greedy_time = float('inf')
        max_hungarian_score = -float('inf')
        best_hungarian_solution = {}
        best_hungarian_time = float('inf')
        while count < 100:
            count +=1
            gens = {}
            new_graphs = {}
            remaining_nodes = {}
            landmarks = {}
            w_weights = {}
            for category, graph in copy_of_full_graphs.items():
    #            outdegrees = list(graph.out_degree())
    #            outdegrees.sort(key=lambda tup: tup[1])#, reverse=True)
    #            landmark = outdegrees[0][0]
                nodes = list(graph.nodes())
                landmark = random.choice(nodes)
#                print(landmark)
                landmarks[category] = landmark
                w_weights[category] = {}
                nodes = set(graph.nodes())
                for node in nodes:
                    if nx.has_path(graph, node, landmark):
                        distance_il = nx.shortest_path_length(graph, node, landmark)
                    else:
                        distance_il = 0
                    if nx.has_path(graph, landmark, node):
                        distance_li = nx.shortest_path_length(graph, landmark, node)
                    else:
                        distance_li = 0
                    w_weights[category][node] = distance_il - distance_li

            current_categories_list = []
            for category, occ in customized_categories.items():
                current_categories_list.append(category)
            nodes = {}    
            for category in current_categories_list:
                nodes[category] = list(reverse_category_graph_clean[category].nodes()) 

            v_weights = whole_weight_calculation(nodes, current_categories_list, w_weights)

            nodes = list(reverse_category_graph_clean[current_categories_list[0]].nodes())
            nodes.sort()

            profit_matrix = np.zeros((len(nodes), len(current_categories_list)))
            counter = 0
            for category in v_weights:
                c = []
                for node in nodes:
                    c.append(v_weights[category][node])
                profit_matrix[:, counter] = np.array(c)
                counter += 1

            if np.shape(profit_matrix)[0] > np.shape(profit_matrix)[1]:
                difference = np.shape(profit_matrix)[0] - np.shape(profit_matrix)[1]
                c = np.array(len(nodes)*[difference*[profit_matrix.min()]])
                profit_matrix = np.append(profit_matrix, c, axis=1)

            cost_matrix = np.max(profit_matrix) - profit_matrix

            # solving

            loop_start_time = time.time()
            greedy_solution, greedy_score = greedy_algorithm(v_weights)
            loop_elapsed_time_greedy = time.time() - loop_start_time
            greedy_score = compute_greedy_score(greedy_solution, copy_of_full_graphs, landmarks)
            
            if greedy_score > max_greedy_score:
                max_greedy_score = greedy_score
                best_greedy_solution = greedy_solution
                best_greedy_time = loop_elapsed_time_greedy
            
#            print('Greedy solution is:', greedy_solution)
#            print('Solution score for greedy graph truth is:', greedy_score)
#            print('Running time is:', loop_elapsed_time_greedy)
#            continue

            loop_start_time = time.time()
            hungarian_solution, hungarian_score = hungarian_algorithm(cost_matrix, profit_matrix, current_categories_list, nodes)
            loop_elapsed_time_hungarian = time.time() - loop_start_time
            hungarian_score = compute_hungarian_score(hungarian_solution, copy_of_full_graphs, landmarks)
            
            if hungarian_score > max_hungarian_score:
                max_hungarian_score = hungarian_score
                best_hungarian_solution = hungarian_solution
                best_hungarian_time = loop_elapsed_time_hungarian
            
#            print('Hungarian solution is:', hungarian_solution)
#            print('Solution score for hungarian graph truth is:', hungarian_score)
#            print('Running time is:', loop_elapsed_time_hungarian)
#        break
        
        for teams, lst in team_dictionairy.items():
            if lst == current_categories_list:
                team = teams

        dct = {'solution': greedy_solution, 'score': greedy_alt_score, 'time': loop_elapsed_time_greedy, 'edges': None}
        if os.path.isfile('results_random_landmarks_greedy.pkl'):
            with open('results_random_landmarks_greedy.pkl', 'rb') as results:
                results_dict = pickle.load(results)
            with open('results_random_landmarks_greedy.pkl', 'wb') as results:
                if team not in results_dict:
                    results_dict[team] = dct
                pickle.dump(results_dict, results)
        else:
            results_dict = {}
            results_dict[team] = dct
            with open('results_random_landmarks_greedy.pkl', 'wb') as results:
                pickle.dump(results_dict, results)


        dct = {'solution': hungarian_solution, 'score': hungarian_alt_score, 'time': loop_elapsed_time_hungarian, 'edges': None}
        if os.path.isfile('results_random_landmarks_hungarian.pkl'):
            with open('results_random_landmarks_hungarian.pkl', 'rb') as results:
                results_dict = pickle.load(results)
            with open('results_random_landmarks_hungarian.pkl', 'wb') as results:
                if team not in results_dict:
                    results_dict[team] = dct
                pickle.dump(results_dict, results)
        else:
            results_dict = {}
            results_dict[team] = dct
            with open('results_random_landmarks_hungarian.pkl', 'wb') as results:
                pickle.dump(results_dict, results)
        
        print('Best greedy solution is:', best_greedy_solution)
        print('Best greedy score is:', max_greedy_score)
        print('Best greedy time is:', best_greedy_time)
        
        print('Best hungarian solution is:', best_hungarian_solution)
        print('Best hungarian score is:', max_hungarian_score)
        print('Best hungarian time is:', best_hungarian_time)
#        break

if __name__ == '__main__':
    main()
    print('Exiting main!')  