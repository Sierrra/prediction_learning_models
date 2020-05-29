# -*- coding: utf-8 -*-
"""
Main module with optimization tasks
"""
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import minimize
from numba import njit

sns.set(context='notebook')

@njit
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@njit
def new_N(N1, ro):
    """
    compute new N
    """
    return ro * N1 + 1


@njit
def get_prob(paction, paction_set, plambda):
    """
    get new probability for action
    """
    soft = softmax(paction_set * np.ones(paction_set.shape) * plambda)
    return soft[np.where(paction_set == paction)][0]
    # return np.exp(plambda * paction) / (1e-4 + np.sum([np.exp(plambda * action) for action in paction_set]))


@njit
def get_payoff(move, game_matrix, index, strat_space):
    """
    payoff matrix view
    """
    return game_matrix[move, strat_space[index]]

@njit
def get_payoff_like_random_man(move, game_matrix):
    """
    payoff matrix view via all actions
    """
    return game_matrix[move, :]

@njit
def get_A(phi, delta, N1, a, ro, index, ego_move, alter_move, game_matrix, strat_space):
    I = int(strat_space[index] == ego_move)
    new_a = (phi * N1 * a + (delta + (1 - delta) * I) * get_payoff(alter_move, game_matrix, strat_space[index],
                                                                   strat_space)) / new_N(N1, ro)
    return new_a

@njit
def get_white_A(phi, delta, N1, a, ro, index, ego_move, alter_move, game_matrix, strat_space):
#     I = int(strat_space[index] == ego_move)
    I = np.zeros(a.size)
    I[ego_move] = 1
    new_a = (phi * N1 * a + (delta + (1 - delta) * I) * get_payoff_like_white_man(alter_move, game_matrix)) / new_N(N1, ro)
    return new_a

@njit
def likelihood(x, data, ids, rounds, set_numb, game_matrices):
    
    # take initial the parametrs
    ro = x[1 - 1]
    delta = x[2 - 1]
    phi = x[3 - 1]
    N0 = x[4 - 1]
    plambda = x[5 - 1]
    loglikelihood = 0.
#     Attr_start_1_0 = x[6 - 1]
#     Attr_start_2_0 = x[7 - 1]
#     Attr_start_1_1 = x[8 - 1]
#     Attr_start_2_1 = x[9 - 1]
    mgame1=game_matrices[0]
    mgame2=game_matrices[1]
    data1 = data[np.where(data[:,0] == set_numb)]
    strat_space = np.arange(0, mgame1.shape[0])
    for it in ids:
        A01 = np.ones(len(strat_space))
        A02 = np.ones(len(strat_space))
        player1_moves = data1[np.where(data1[:,1] == it)][:,3]
        player2_moves = data1[np.where(data1[:,1] == it)][:,4]
        for rount, _ in enumerate(rounds):
            if rount == 0:
                N1 = N0
            loglikelihood -= np.log(1e-4 + get_prob(A01[strat_space[player1_moves[rount]]], A01, plambda)) + np.log(
                1e-4 + get_prob(A02[strat_space[player2_moves[rount]]], A02, plambda))
            A01 = np.array([get_A(phi, delta, N1, A01[idx], ro, idx, player1_moves[rount],
                                  player2_moves[rount], mgame1,strat_space) for idx in range(mgame1.shape[0])])
            A02 = np.array([get_A(phi, delta, N1, A02[idx], ro, idx, player2_moves[rount],
                                  player1_moves[rount], mgame2,strat_space) for idx in range(mgame2.shape[0])])
#             A01 = get_white_A(phi, delta, N1, A01, ro, 0, player1_moves[rount], player2_moves[rount], mgame1,strat_space)
#             A02 = get_white_A(phi, delta, N1, A02, ro, 0, player2_moves[rount], player1_moves[rount], mgame2,strat_space)
        
            if rount == 0:
                N1 = new_N(N0, ro)
            else:
                N1 = new_N(N1, ro)
             
    return loglikelihood + 10 * max(0, plambda - 1.9)


def minimize_likelihood(likelihood, x0, data, ids, rounds, set_numb, bounds, game_matrices):
    """
    Run computation of loglikelihood maximization(-loglikelihood minimization)
    """
    res = minimize(likelihood, x0=x0, args=(data, ids, rounds, set_numb, game_matrices), method='L-BFGS-B',
                    bounds=bounds)
    return res
    

def set_evaluation (set_numbs, data, ids, rounds, game_matrices):
    """
    Run parallel computation of optimization tasks(loglikelihood maximization) over set_numbs
    """
    ll_results =  Parallel(n_jobs=8)(delayed(minimize_likelihood)(likelihood, tuple(np.random.rand(1,5)), 
                           data, ids, rounds, set_numb,((0, 1), (0, 1), (0, None), (0, 1), (0, None),), game_matrices) for set_numb in set_numbs)
        
    return [res.x for res in ll_results]

# def set_evaluation (set_numbs, data, ids, rounds, game_matrices):  
#     ll_results =  [minimize_likelihood(likelihood, tuple(np.random.rand(1,5)), 
#                            data, ids, rounds, set_numb,((0, 1), (0, 1), (0, None), (0, 1), (0, None),), game_matrices) for set_numb in set_numbs]
        
#     return [res.x for res in ll_results]

def set_evaluation_with_graphs(set_numbs, data, ids, rounds, game_matrices, real_params, model_name, save_graphs=False, save_evaluations=False):
    """
    make evaluations for model over data with the payoff matrices
    Params:
    data: simulations
    game_matrices: payoff matrices
    real_params: dict of real params
    save_graphs: flag for plot saving
    save_evalutions: save optimal params to csv
    """
    dt_string = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M')
    set_evaluations = np.array(set_evaluation(set_numbs, data, ids, rounds, game_matrices))
    atmp = np.array(set_evaluations)
    
    # loglikelihood may not converge, filter this cases
    if np.any(atmp >= 2):
        where_hovno_is = np.sum(atmp >= 2, axis=1)
        who_made_hovno = np.sum(atmp >= 2, axis=0)
        
        print("we have {amounts} bad computations".format(amounts=np.sum(where_hovno_is)))
        print("bad computations in sets {hovno_places}".format(
            hovno_places=', '.join([str(i) for i in np.where(where_hovno_is)[0]])))
        
        hovno_blames=['{t} times in parameter {p}'.format(t=t, p=p) for p, t in enumerate(who_made_hovno) if t != 0]
        print("bad computations {hovno_blames}".format(
            hovno_blames=', '.join(hovno_blames)) )
        
        print("average bad computation magnitude  IS {mgn}".format(mgn=np.mean(atmp[np.where(atmp >= 2)])))
        
        atmp = atmp[np.nonzero(1 - where_hovno_is), :]
        if np.sum(where_hovno_is)>0:
            atmp = atmp[0,:,:]
    
    f, axes = plt.subplots(figsize=(14, 8), sharex=True)
    print(real_params)
        
    for idx,key in enumerate(real_params):
        sns.distplot(atmp[:,idx],kde=False,label="-".join([str(key), str(real_params[key])]), kde_kws={'clip': (-5, 5.)})
        
    pct = plt.legend()
    #saving graphs
    if save_graphs:
        filename = '_'.join((model_name, dt_string)) + '.png'
        plt.savefig(filename)
    #saving optimal params
    if save_evaluations:
        filename = '_'.join((model_name, dt_string)) + '.csv'
        evaluations = pd.DataFrame(atmp, columns=real_params.keys())
        evaluations.to_csv(filename, index=False)
    plt.xlabel('parameters distribution for {model_name}'.format(model_name=model_name))
        