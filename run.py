import numpy as np
import pandas as pd

from pandas import DataFrame, Series

from ewa import EWA
from simulations import simulate_record, alt_simulate_record
from likelihood import set_evaluation, set_evaluation_with_graphs, likelihood
from tests import transformation, likelihood_ratio_test, salmon_set_test
import datetime
from matrices import matrix_set
from metrics import brier_multi, log_score_multi, wide_brier_calc

dt_string = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M')

matrixSet = [[np.array(item[0].T, order='F'), item[1]] for item in matrix_set]

matrixSetTransposed = [[item[0].T, item[1]] for item in matrix_set]

for item in matrixSet:
    item[0] = DataFrame(np.array(item[0]))
    item[1] = DataFrame(np.asarray(item[1]))
# matrixSet
print(matrixSet[0][0])


population_set = {
    'EWA_FP': [1, 1, 1, 0, 1],
    'EWA_CB': [0, 1, 0, 1, 1],
    'EWA_RL': [0, 0, 0.75,1,1], 
    'EWA_mdl':[0.5,0.5,0.5,0.5,1],
#     'test': [0.97, 0.6,  0.94,  1, 0.8, ],
            }
params_name = ['ro','delta','phi','N_in','lambd']
set_nubmbs = 50
couple_ides = 40
ages = 40


#Run evaluations over simulated data
for model in population_set:
    for idx, game_matrices in enumerate(matrixSet):
        print("MODEL: {model_name}".format(model_name=model))
        print("Matrix_set index : " + str(idx))
        simulation_df = simulate_record(ages, set_nubmbs,  dict(zip(params_name, population_set[model])),couple_ides, game_matrices, model_name=model)
        simulation =  simulation_df[simulation_df.columns[[0,1,2,3,4,]]]
        ids = sorted(simulation_df.id.unique())
        rounds = sorted(simulation_df['round'].unique())
        set_numbs = sorted(simulation_df['set_numb'].unique())
        simulation = simulation.to_numpy(dtype=None, copy=False)
        game_matrices=[item.values for item in game_matrices]
        set_evaluation_with_graphs(set_numbs, simulation, ids, rounds, game_matrices, dict(zip(params_name,population_set[model])), model, True, True)
        break
        
is_tested_set  = transformation(population_set)
hypothesis_set = {
#     'rho_1':{0:1},
#     'rho_0':{0:0},
    'delta_1':{1:1},
    'delta_0':{1:0},
     'phi_1':{1:1},
     'phi_0':{1:0}
    }

#Run hypothesis tests for restricted params
res = []
for model in population_set:
    for idx, game_matrices in enumerate(matrixSet):
        print("MODEL: {model_name}".format(model_name=model))
        print("Matrix: " + str(len(game_matrices[0]))+"x"+ str(len(game_matrices[0])))
        simulation_df = simulate_record(ages, set_nubmbs,  dict(zip(params_name, population_set[model])),
                                        couple_ides, game_matrices, model_name=model)
        simulation =  simulation_df[simulation_df.columns[[0,1,2,3,4,]]]
        ids = sorted(simulation_df.id.unique())
        rounds = sorted(simulation_df['round'].unique())
        set_numbs = sorted(simulation_df['set_numb'].unique())
        simulation = simulation.to_numpy(dtype=None, copy=False)
        game_matrices=[item.values for item in game_matrices]
        params_for_likelihood = [simulation, ids, rounds]
        y = salmon_set_test(start_params=[np.random.rand(1,1)[0][0] for i in range(5)], params_for_likelihood=params_for_likelihood, 
                        params_restrictions= is_tested_set[model], test_results =  np.array([]), set_numbs = set_numbs, 
                        game_matrices=tuple([game_matrices[0], game_matrices[1]]), 
                        _bounds=((0, 1),(0, 1),(0, None),(0, 1),(0, None)) )
        percent_of_set_H0_not_false = np.mean((y>=0.95).astype(int))
        hyp_set = []
        for hyp in hypothesis_set.keys():
            z = salmon_set_test(start_params=[np.random.rand(1,1)[0][0] for i in range(5)], params_for_likelihood=params_for_likelihood, 
                        params_restrictions= hypothesis_set[hyp], test_results =  np.array([]), set_numbs = set_numbs, 
                        game_matrices=tuple([game_matrices[0], game_matrices[1]]), 
                        _bounds=((0, 1),(0, 1),(0, None),(0, 1),(0, None)) )
            percent_of_z_set_H0_not_false = np.mean((z>=0.95).astype(int))
            mean_p_val = round(np.mean(z),2)
            hyp_set.append([hyp,'percent this one param hold', percent_of_z_set_H0_not_false,'mean_p_val', mean_p_val])

        for hypo in hyp_set:
            dct = {'model_name':model, 'matrix':(str(len(game_matrices[0]))+"x"+ str(len(game_matrices[0])))}
            dct['hyp'] = hypo[0]
            dct[hypo[1]] = hypo[2]
            dct[hypo[3]] = hypo[4]
            res.append(dct)
        print('percent, 4 params hold', percent_of_set_H0_not_false, 'mean p value', round(np.mean(y),2), hyp_set)
#             break
tests = pd.DataFrame(res)

tests.to_csv("tests_{dt}.csv".format(dt=dt_string))


fun = [0]
res = []
for model in population_set:
    for idx, game_matrices in enumerate(matrixSet):
        print("MODEL: {model_name}".format(model_name=model))
        print("Matrix_set index : " + str(idx))
        simulation_df = simulate_record(ages, set_nubmbs,  dict(zip(params_name, population_set[model])),
                                        couple_ides, game_matrices, model_name=model)
        ids = sorted(simulation_df.id.unique())
        rounds = sorted(simulation_df['round'].unique())
        set_numbers = sorted(simulation_df['set_numb'].unique())
        simulation = simulation_df.to_numpy(dtype=None, copy=False)
#             simulation =  simulation_df[simulation_df.columns[[0,1,2,3,4,]]]

        for alt_model in population_set:

            alt_simulation_df = alt_simulate_record(ages, set_numbers,  simulation,
                                                dict(zip(params_name, population_set[alt_model])),
                                                couple_ides, game_matrices, model_name=alt_model)
            pl1_move_wide_data = simulation_df['pl1_move_wide']


            y_true_pl1 = np.asarray([np.asarray(x)  for x in  pl1_move_wide_data], dtype=np.float32)



            probs_pl1_data = alt_simulation_df['probs_pl1']
            y_prob_pl1 = np.asarray([np.asarray(x)   for x in  probs_pl1_data], dtype=np.float32)
#                 [::-1]
#                 print(probs_pl1_data.head, pl1_move_wide_data.head)
            brier_pl1 = brier_multi(y_true_pl1, y_prob_pl1)
            container = []
            wide_brier = wide_brier_calc(simulation_df, alt_simulation_df, container, model,alt_model )
#                 print(np.round(brier_pl1,3)-np.round(np.mean(wide_brier["brier_score"]),3) )
            for one_set in range(set_nubmbs):
                dct = dict((('brier', (wide_brier["brier_score"][one_set])),( "sd_brier", np.std(wide_brier["brier_score"])),
                  ("log_score", np.mean(wide_brier["log_score"])), ("sd_log_score", np.std(wide_brier["log_score"])), 
                  ('player','pl_1'),('true_model', model), ('alt_model',  alt_model), ('set', wide_brier["set_numb"][one_set])))
                dct['model_name'] = model
                dct['matrix'] = str(len(game_matrices[0]))+"x"+ str(len(game_matrices[0]))
                res.append(dct)
#                 print('brier:', brier_pl1, "sd_brier", np.std(wide_brier["brier_score"]),
#                       "log_score", np.mean(wide_brier["log_score"]), "sd_log_score", 
#                       'pl_1','true_model', model, 'alt_model',  alt_model)

#             break
briers = pd.DataFrame(res)

briers.to_csv('measure_of_accuracy_{dt}.csv'.format(dt=dt_string))
