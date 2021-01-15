# -*- coding: utf-8 -*-
"""
This module contains basic simulation functions
"""

import numpy as np
import pandas as pd
import random
import csv
import timeit

from ewa import EWA, SEWA


#names of columns in simulation records
cols= ['set_numb', 'id','round','pl_1_move','pl_2_move', 'model','probs_pl1',
'probs_pl2', 'pl1_move_wide', 'pl2_move_wide','pl_1_payoff', 'pl_2_payoff']


def choose_class(name):
    if name == 0:
        class localEWA(EWA):
            pass
        return EWA # возвращает класс, а не экземпляр
    elif name == 1:
        class localSEWA(SEWA):
            pass
        return SEWA

def PlayerDefiner(switcher, game_matrices, SwitchClass, pl_num):
    space = game_matrices[pl_num-1].columns.values
    return SwitchClass(probs = [1/(len(space)) for attrs in range(len(space))],  
                       attr_s1 =np.resize([1, 1], game_matrices[pl_num-1].shape[0]), pl_num = pl_num, 
                       action_space = space, game_matrix = game_matrices[pl_num-1], **switcher)

def simulate_record (ages, set_numbs, switcher, couple_ides, game_matrices, model_name):
    """function simulate sequences of games
    
    return pandas.DataFrame according to cols
    """
    matrix_len = game_matrices[0].shape[0]
    result_records = []
    for set_numb in range(set_numbs):
        for couple_id in range(couple_ides):
#                 pl1_num_actions = len(game_matrices[0].columns.values)
                #for game_matrix in game_matrices:
                # player_n = 
                # set the initial values of parameters of players

                SwitchClass = choose_class(switcher['if_strategic']) 
#                 player_1 = SwitchClass(prob_s1 = [1/(len(game_matrices[1-1].columns.values)) for attrs in range(len(game_matrices[1-1].columns.values))], attr_s1 =np.resize([1, 1], game_matrices[1-1].shape[0]), pl_num = 1,strat_space=[0], game_matrix = game_matrices[1-1], **switcher)
                player_1 =PlayerDefiner(switcher, game_matrices, SwitchClass, 1)
#                 pl2_num_actions = len(game_matrices[0].columns.values)
                player_2 =PlayerDefiner(switcher, game_matrices, SwitchClass, 2)
                for age in range(ages):
                # choose the moves
                    print(player_1.attr_s1)
                    pl_1_mov = np.random.choice(player_1.action_space, p=player_1.probs)
#                     pl_2_mov = np.random.choice(player_2.action_space, p=[0.70, 0.3])
#                   pl_2_mov = np.random.choice(player_2.action_space, p=[0.70, 0.10,0.1,0.1])
                    pl_2_mov = np.random.choice(player_2.action_space, p=player_2.probs)
                    pl_1_payoff = player_1.get_payoff(alter_move = pl_2_mov, index = pl_1_mov)
                    pl_2_payoff = player_2.get_payoff(alter_move = pl_1_mov, index = pl_2_mov)
                    pl_1_mov_wide = [0] * matrix_len
                    pl_2_mov_wide = [0] * matrix_len
                    pl_1_mov_wide[pl_1_mov]=1
                    pl_2_mov_wide[pl_2_mov]=1
                    predict_probs_pl1 = player_1.probs
                    predict_probs_pl2 = player_2.probs
                    
#                 record results
                    fields = [set_numb, couple_id, age, pl_1_mov, pl_2_mov, model_name, 
                              predict_probs_pl1, predict_probs_pl2, pl_1_mov_wide, pl_2_mov_wide,pl_1_payoff,pl_2_payoff]
                    result_records.append(fields)
                

        
                #update parameters
                    player_1.EWA_compute_new_prmtrs(pl_1_mov = pl_1_mov, pl_2_mov = pl_2_mov)
                    player_2.EWA_compute_new_prmtrs(pl_1_mov = pl_1_mov, pl_2_mov = pl_2_mov)


                    

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