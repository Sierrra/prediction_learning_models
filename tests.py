# -*- coding: utf-8 -*-
"""
Module with stat tests
"""

import numpy as np

from scipy.stats import chi2
from likelihood import likelihood, minimize_likelihood
#likelihood ratio test

def transformation (set_of_models):
    model_set = dict()
    for model in set_of_models:
        model_set[model] =  dict(zip(range(len(set_of_models[model])-1), set_of_models[model]))
    return model_set

def likelihood_ratio_test(likelihood_function, params_for_likelihood, start_params, params_restrictions, set_numb, game_matrices, _bounds):
    """
    Basic hypothesis test
    
    """
    data, ids, rounds = params_for_likelihood
    if params_restrictions:
        if len(params_restrictions) >= len(start_params):
            raise ValueError ( "Alternate features must have more features than null features")
        bounds = _bounds
        r_bounds = []
        for idx, item in enumerate(_bounds):
            if params_restrictions.get(idx) is not None:
                r_bounds.append([params_restrictions[idx]]*2)
            else:
                r_bounds.append(_bounds[idx])
        results = minimize_likelihood(likelihood_function, start_params, 
                           data, ids, rounds, set_numb,r_bounds, game_matrices)
        
        null_ll = results.fun
        df = len(start_params) - len(params_restrictions)
    else:
        df = len(start_params)
        results = minimize_likelihood(likelihood_function, start_params, 
                           data, ids, rounds, set_numb,zip(start_params, start_params), game_matrices)
        
        null_ll = results.fun
    results_2 = minimize_likelihood(likelihood_function, start_params, 
                           data, ids, rounds, set_numb ,bounds, game_matrices)
    alt_ll = results_2.fun
#     print(alt_ll, null_ll,df)
    G = 2 * np.abs(alt_ll - null_ll)
    p_value = chi2.sf(G, df)
#     print(G, p_value)

    return p_value

def salmon_set_test(start_params, params_for_likelihood, params_restrictions, test_results, set_numbs, game_matrices, _bounds):
    test_results = np.array([])
    for set_numb in set_numbs:
        test_results = test_results
        test = likelihood_ratio_test(likelihood, params_for_likelihood, tuple(start_params), params_restrictions, set_numb, game_matrices, _bounds)
        test_results = np.append(test_results, test)
    return test_results
