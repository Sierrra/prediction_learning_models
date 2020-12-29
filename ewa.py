# -*- coding: utf-8 -*-
"""
This module contains EWA computation algo
"""

import numpy as np
import pandas as pd
import random
import csv
import itertools

from scipy.special import softmax

class SEWA:
    def __init__(self, probs, attractions, pl_num,action_space,ro,delta,phi,N_in,lambd, game_matrix):
        self.strat_space = np.array(list(itertools.product(*[action_space]*3)))
        self.action_space = np.array(action_space)
        self.attractions = attractions
        self.lambd = lambd
        self.N_in = N_in
        self.ro = ro
        self.delta = delta
        self.phi = phi
        self.probs = self.get_initial_prob(self.action_space)
        self.history = []
        self.game_matrix = game_matrix
        self.step_attractions = [1.]*self.action_space
        self.pl_num = pl_num


    def get_initial_prob(self, paction_set):
        """
        finding initial probabilities over paction_set
        return probability distribution: numpy.array(float)
        """
        soft = softmax(paction_set * np.ones(paction_set.shape)*self.lambd)
        return soft

    def get_new_N(self, N_in):
        """
        iterating N for non-initial round (N is EWA parameter that normalizes previous experience)
        return new N
        """
        return self.ro * N_in + 1
    
    def get_payoff(self, alter_move, index):
        """
        payof matrix view
        return payoff for current move: float
        """
        z=self.game_matrix.loc[alter_move, self.strat_space[index]]
        return z

    def p(self, int_a, set_a):
        """
        computing probabilities for non-initial rounds 
        """
        return np.exp(self.lambd * int_a) / sum([np.exp(self.lambd * j) for j in set_a])

    def get_A(self, N_in, a, index, ego_move, alter_move):
        """
        function for computing round's attractions 
        return new attraction for round: float
        """
        I = int(self.action_space[index] == ego_move)
        new_a = (self.phi * N_in * a + (self.delta + (1 - self.delta) * I) 
                * self.get_payoff(alter_move, self.action_space[index])) / self.get_new_N(N_in)
        return new_a

    def strategy_met_history(self):
        """
        computing probabilities for non-initial rounds 
        """
        for i in range(self.strat_space.shape[0]):
            for g in range(len(self.action_space)):
                if self.strat_space[i] == history + [g]:
                   self.step_attractions[g] = i
        # for g in range(3):
        #     self.probs[g] = self.probs[g]/summa 

    def EWA_compute_new_prmtrs (self, u, pl_1_mov, pl_2_mov):
        """
        recalculate new attractions from old attractions and recent game history 
        """
        new_as = self.attractions
        N_in = self.N_in
        bin_v = int(self.pl_num == 1)
        ego_move = (pl_1_mov * bin_v + (1 - bin_v) * pl_2_mov)
        alter_move = (pl_2_mov * bin_v + (1 - bin_v) * pl_1_mov)
        self.history = [ego_move, alter_move]
        self.strategy_met_history()
        for  idx in self.step_attractions:
            new_as[idx] = self.get_A(N_in, a, i, ego_move, alter_move)        
        self.attractions = new_as
        new_ps = []
        for a in self.step_attractions:
            new_ps.append(self.p(a, self.step_attractions))        
        self.probs = new_ps
        self.N_in = self.get_new_N(N_in)

    

#  class str_FP_27:
#     def __init__ (self,propensity, frequency, predict_probs_pl1):
# #       initial parameters
#         self.frequency = [1]*27
#         self.propensity = [1/27]*27
#         self.strat_space = list(itertools.product([0,1,2], [0,1,2], [0,1,2]))
#         self.predict_probs_pl1 = np.array([1/3]*3)
        
#     def counter_prop(self, last_move, pl_2_move_last, move):
#         for i in range(len(self.strat_space)):
#             if self.strat_space[i] == (last_move, pl_2_move_last, move):
#                 self.frequency[i] += 1

#         for i in range(len(self.propensity)):
#             self.propensity[i] = self.frequency[i]/(sum(self.frequency))

        
#     def prediction (self, last_move, pl_2_move_last):
#         self.predict_probs_pl1 = [0,0,0]
#         for i in range(len(self.strat_space)):
#             for g in range(3):
#                 if self.strat_space [i] ==(last_move, pl_2_move_last, g):
#                     self.predict_probs_pl1[g] = self.propensity[i]
#         summa =  np.sum(self.predict_probs_pl1)
#         for g in range(3):
#             self.predict_probs_pl1[g] = self.predict_probs_pl1[g]/summa   
    

class EWA:
    """Basic implementation of EWA(experience weighted attraction)


    Attributes:
        ro,delta,phi, N_in, lambd - EWA parameters
        game_matrix - payoff matrix

    """
    def __init__ (self,prob_s1,attr_s1,pl_num,strat_space,ro,delta,phi,N_in,lambd, game_matrix):
#       initial parameters
        self.attr_s1 = attr_s1
        self.lambd = lambd
        self.prob_s1 = self.get_initial_prob(self.attr_s1)
        self.pl_num = pl_num
        self.strat_space = strat_space
        self.ro = ro 
        self.delta = delta 
        self.phi = phi 
        self.N_in = N_in 
        # extra steps to accomodate possibility of non-quadratic game matrices
        self.strat_space = game_matrix.columns.values
        self.game_matrix = game_matrix


    def get_initial_prob(self, paction_set):
        """
        finding initial probabilities over paction_set
        return probability distribution: numpy.array(float)
        """
        soft = softmax(paction_set * np.ones(paction_set.shape)*self.lambd)
        return soft

   
    def p(self, int_a, set_a):
        """
        computing probabilities for non-initial rounds 
        """
        return np.exp(self.lambd * int_a) / sum([np.exp(self.lambd * j) for j in set_a])

    def get_new_N(self, N_in):
        """
        iterating N for non-initial round (N is EWA parameter that normalizes previous experience)
        return new N
        """
        return self.ro * N_in + 1


    def get_payoff(self, alter_move, index):
        """
        payof matrix view
        return payoff for current move: float
        """
        z=self.game_matrix.loc[alter_move, self.strat_space[index]]
        return z


    def get_A(self, N_in, a, index, ego_move, alter_move):
        """
        function for computing round's attractions 
        return new attraction for round: float
        """
        I = int(self.strat_space[index] == ego_move)
        new_a = (self.phi * N_in * a + (self.delta + (1 - self.delta) * I) *self.get_payoff(alter_move, self.strat_space[index])) / self.get_new_N(N_in)
        return new_a
    
    # 
    def EWA_compute_new_prmtrs (self,  pl_1_mov, pl_2_mov):
        """
        recalculate new attractions from old attractions and recent game history 
        """
        new_as = []
        N_in=self.N_in
        u = self.attr_s1
        bin_v = int(self.pl_num == 1)
        ego_move = (pl_1_mov*bin_v+(1-bin_v)*pl_2_mov)
        alter_move= (pl_2_mov*bin_v+(1-bin_v)*pl_1_mov)
        for i,a in enumerate(u):
            new_as.append(self.get_A(N_in, a, i, ego_move, alter_move))        
        self.attr_s1 = new_as
        new_ps = []
        for a in self.attr_s1:
            new_ps.append(self.p(a, self.attr_s1))        
        self.prob_s1 = new_ps
        self.N_in = self.get_new_N(N_in)
