# -*- coding: utf-8 -*-
"""
This module contains basic simulation functions
"""

import numpy as np
import pandas as pd
import random
import csv
import timeit

from ewa import EWA

#names of columns in simulation records
cols= ['set_numb', 'id','round','pl_1_move','pl_2_move', 'model','probs_pl1',
'probs_pl2', 'pl1_move_wide', 'pl2_move_wide','pl_1_payoff', 'pl_2_payoff']


def simulate_record (ages, set_numbs, switcher, couple_ides, game_matrices, model_name):
    """function simulate sequences of games
    
    return pandas.DataFrame according to cols
    """
    matrix_len = game_matrices[0].shape[0]
    result_records = []
    for set_numb in range(set_numbs):
        for couple_id in range(couple_ides):
                pl1_num_actions = len(game_matrices[0].columns.values)
                #for game_matrix in game_matrices:
                # player_n = 
                # set the initial values of parameters of players
                player_1 = EWA(prob_s1 = [1/(pl1_num_actions) for attrs in range(pl1_num_actions)], 
                    attr_s1 = np.resize([1, 1], game_matrices[0].shape[0]), 
                    pl_num =1,strat_space=[0],  game_matrix = game_matrices[0], **switcher ) #ro = swicher[0],  delta = swicher[1],  phi = swicher[2],  N_in = swicher[3], lambd = swicher[4], )
                pl2_num_actions = len(game_matrices[0].columns.values)
                player_2 = EWA(
                    prob_s1 = [1/(pl2_num_actions) for attrs in range(pl2_num_actions)],  
                    attr_s1 =np.resize([1, 1], game_matrices[1].shape[0]), 
                     pl_num =2, strat_space=[0],game_matrix = game_matrices[1], **switcher) # ro = swicher[0],  delta = swicher[1],  phi = swicher[2], N_in = swicher[3], lambd = swicher[4])

                for age in range(ages):
                # choose the moves
                    pl_1_mov = np.random.choice(player_1.strat_space, p=player_1.prob_s1)
#                     pl_2_mov = np.random.choice(player_2.strat_space, p=[0.70, 0.3])
#                   pl_2_mov = np.random.choice(player_2.strat_space, p=[0.70, 0.10,0.1,0.1])
                    pl_2_mov = np.random.choice(player_2.strat_space, p=player_2.prob_s1)
                    pl_1_payoff = player_1.get_payoff(alter_move = pl_2_mov, index = pl_1_mov)
                    pl_2_payoff = player_2.get_payoff(alter_move = pl_1_mov, index = pl_2_mov)
                    pl_1_mov_wide = [0] * matrix_len
                    pl_2_mov_wide = [0] * matrix_len
                    pl_1_mov_wide[pl_1_mov]=1
                    pl_2_mov_wide[pl_2_mov]=1
                    predict_probs_pl1 = player_1.prob_s1
                    predict_probs_pl2 = player_2.prob_s1
                    
#                 record results
                    fields = [set_numb, couple_id, age, pl_1_mov, pl_2_mov, model_name, 
                              predict_probs_pl1, predict_probs_pl2, pl_1_mov_wide, pl_2_mov_wide,pl_1_payoff,pl_2_payoff]
                    result_records.append(fields)
                

        
                #update parameters
                    player_1.EWA_compute_new_prmtrs(u=player_1.attr_s1, pl_1_mov = pl_1_mov, pl_2_mov = pl_2_mov)
                    player_2.EWA_compute_new_prmtrs(u=player_2.attr_s1,pl_1_mov = pl_1_mov, pl_2_mov = pl_2_mov)


                    

# #                 write results
#                     with open(r'{}_data.csv'.format(model_name), 'a') as f:
#                         writer = csv.writer(f)
#                         writer.writerow(fields)

    return pd.DataFrame(result_records, columns=cols)


def alt_simulate_record(ages, set_numbers,simulation,  switcher, couple_ides, game_matrices, model_name):
    cols= ['set_numb', 'id','round', 'model','probs_pl1']
    matrix_len = game_matrices[0].shape[0]
    result_records = []
    for set_numb in set_numbers:
        data2 = simulation[np.where(simulation[:,0] == set_numb)]
        for couple_id in range(couple_ides):
            pl1_num_actions = len(game_matrices[0].columns.values)
            #for game_matrix in game_matrices:
            # player_n = 
            # set the initial values of parameters of players

            player_3 = EWA(prob_s1 = [1/(pl1_num_actions) for attrs in range(pl1_num_actions)],  
                attr_s1 =np.resize([1, 1], game_matrices[0].shape[0]), 
                 pl_num =1, strat_space=[0],game_matrix = game_matrices[0], **switcher)

            player1_moves = data2[np.where(data2[:,1] == couple_id)][:,3]
            player2_moves = data2[np.where(data2[:,1] == couple_id)][:,4]

            for age in range(ages):
                pl_1_mov = player1_moves[age]
                pl_2_mov = player2_moves[age]
                predict_probs_pl1 = player_3.prob_s1
                #data2[np.where(data2[:,1] == couple_id)][:,6][age]
                fields = [set_numb, couple_id, age, model_name, predict_probs_pl1]
                result_records.append(fields)
                player_3.EWA_compute_new_prmtrs(u=player_3.attr_s1, pl_1_mov = pl_1_mov, pl_2_mov = pl_2_mov)
    return pd.DataFrame(result_records, columns=cols)