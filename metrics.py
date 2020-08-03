import pandas as pd
import numpy as np
import ast
from sklearn.metrics import log_loss

def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets)**2, axis=1))


def log_score_multi(targets, probs):
    return np.mean(np.asarray([log_loss(x[0],x[1]) for x in [[targets[y], probs[y]] for y in range(len(targets)) ]]))

def wide_brier_calc (df_true, df_hypotetic, new_container, true_model, alt_model, set_nubmbs):
    for one_set in set_nubmbs:
        data = np.asarray([np.asarray(x)  for x in  df_true[df_true["set_numb"]==one_set]['pl1_move_wide']], dtype=np.float32),np.asarray([np.asarray(x)  for x in  df_hypotetic[df_hypotetic["set_numb"]==one_set]['probs_pl1']], dtype=np.float32)
        new_container.append([one_set, brier_multi(data[0],data[1]), log_score_multi(data[0],data[1]), true_model, alt_model])
    return pd.DataFrame(new_container, columns=["set_numb", "brier_score","log_score", "true_model", "alt_model"])
