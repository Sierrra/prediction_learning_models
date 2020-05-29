# -*- coding: utf-8 -*-
"""
This module contains EWA computation algo
"""

import numpy as np
import pandas as pd
import random
import csv

from scipy.special import softmax

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
    def EWA_compute_new_prmtrs (self, u, pl_1_mov, pl_2_mov):
        """
        recalculate new attractions from old attractions and recent game history 
        """
        new_as = []
        N_in=self.N_in
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
