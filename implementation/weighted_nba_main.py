import weighted_algs as algs
import pandas as pd
import numpy as np
import random 
import time
import os.path
import pickle
import sys

def create_skill_author_index(category_rankings):
	skill_author_index_dict = {}
	for category, category_ranking in category_rankings.items():
		for ranking, author_score in enumerate(category_ranking):
			author = author_score[0]
			tup = (category,author)
			if tup not in skill_author_index_dict:
				skill_author_index_dict[tup] = ranking

	return skill_author_index_dict

def main():
	included_dict = {'3P':1,'TRB':1,'AST':1,'STL':1,'BLK':1,'FG':1,'2P':1,'FT':1,'OBPM':1,'DBPM':1,'VORP':1}


	file_path = 'Seasons_Stats.csv'
	num_of_nodes_list = []; num_of_skills_list = []; algorithm_list = []; score_list = []; solution_list = []
	time_list = []; loop_time_list = []; year_list = []
	weighted = True
	filename = 'nba.csv'

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

		included_list = []
		for category, occ in included_dict.items():
			included_list.append(category)

		w_weights = algs.individual_weight_calculation_rank(nba_dataset_ranking, included_list)
		v_weights = algs.whole_weight_calculation(included_list, w_weights)

		players = []
		for player in v_weights[included_list[0]]:
			players.append(player)
		players.sort()
		
		profit_matrix = np.zeros((len(players), len(included_list)))
		counter = 0
		for skill in v_weights:
			c = []
			for player in players:
				c.append(v_weights[skill][player])
			profit_matrix[:, counter] = np.array(c)
			counter += 1
			
		if np.shape(profit_matrix)[0] > np.shape(profit_matrix)[1]:
			difference = np.shape(profit_matrix)[0] - np.shape(profit_matrix)[1]
			c = np.array(len(players)*[difference*[profit_matrix.min()]])
			profit_matrix = np.append(profit_matrix, c, axis=1)

		cost_matrix = np.max(profit_matrix) - profit_matrix

#		break

		loop_start_time = time.time()
		greedy_solution, greedy_score = algs.greedy_algorithm(v_weights, w_weights)
		loop_elapsed_time_greedy = time.time() - loop_start_time
		print('Greedy solution is:', greedy_solution)
		print('Solution score for greedy rank truth is:', greedy_score)
		print('Running time is:', loop_elapsed_time_greedy)
#		break

		loop_start_time = time.time()
		hungarian_solution, hungarian_score = algs.hungarian_algorithm(cost_matrix, profit_matrix, included_list, players, w_weights)
		loop_elapsed_time_hungarian = time.time() - loop_start_time
		print('Hungarian solution is:', hungarian_solution)
		print('Solution score for hungarian rank truth is:', hungarian_score)
		print('Running time is:', loop_elapsed_time_hungarian)

		break

		dct = {'solution': greedy_solution, 'score': greedy_score, 'time': loop_elapsed_time_greedy, 'edges': None}
		if os.path.isfile('results_weighted_nba_greedy.pkl'):
			with open('results_weighted_nba_greedy.pkl', 'rb') as results:
				results_dict = pickle.load(results)
			with open('results_weighted_nba_greedy.pkl', 'wb') as results:
				if year not in results_dict:
					results_dict[year] = dct
					pickle.dump(results_dict, results)
		else:
			results_dict = {}
			results_dict[year] = dct
			with open('results_weighted_nba_greedy.pkl', 'wb') as results:
				pickle.dump(results_dict, results)


		dct = {'solution': hungarian_solution, 'score': hungarian_score, 'time': loop_elapsed_time_hungarian, 'edges': None}
		if os.path.isfile('results_weighted_nba_hungarian.pkl'):
			with open('results_weighted_nba_hungarian.pkl', 'rb') as results:
				results_dict = pickle.load(results)
			with open('results_weighted_nba_hungarian.pkl', 'wb') as results:
				if year not in results_dict:
					results_dict[year] = dct
					pickle.dump(results_dict, results)
		else:
			results_dict = {}
			results_dict[year] = dct
			with open('results_weighted_nba_hungarian.pkl', 'wb') as results:
				pickle.dump(results_dict, results)

#		break

if __name__ == '__main__':
	main()
	print('Exiting main!')