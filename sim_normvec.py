import numpy as np
import pandas as pd
import simulation
import utils

STD_ERR = 0.001

def euclidist(v1, v2):
    return np.linalg.norm(v1 - v2)

def create_user_data(uid, df, pct_items, u_err, difficulty_dict=None, extraarg=None):
    items = df.topic_item.unique()
    n_items_labeled = int(np.round(pct_items * len(items)))
    items_labeled = sorted(np.random.choice(items, n_items_labeled, replace=False))
    labels = []
    for item in items_labeled:
        idf = df[df.topic_item == item]
        idflen = len(idf.gold.values[0])
        err = np.random.normal(0, u_err, idflen)
        if difficulty_dict is not None:
            i_difficulty = difficulty_dict.get(item)
            err += np.random.normal(0, i_difficulty, idflen)
        label = idf.gold.values[0] + err
        labels.append(label)
    dfdict = {
        "uid": [uid] * len(items_labeled),
        "topic_item": items_labeled,
        "label": labels,
    }
    return pd.DataFrame(dfdict)

class VectorSimulator(simulation.Simulator):
    def __init__(self, num_items, vectorlength=6):
        itemM = np.random.normal(0, 1, (num_items, vectorlength))
        self.df = pd.DataFrame({"topic_item":np.arange(len(itemM)), "gold":list(itemM)})

    def create_stan_data(self, n_users, pct_items, err_rates, difficulty_dict):
        self.err_rates = err_rates
        self.difficulty_dict = difficulty_dict
        self.sim_df = simulation.create_sim_df(create_user_data, self.df, n_users, pct_items, err_rates, difficulty_dict)
        stan_data = utils.calc_distances(self.sim_df, (lambda x,y: 1 - self.eval_fn(x, y)), label_colname="label", item_colname="topic_item")
        return stan_data

    def sim_uerr_fn(self, uerr_a, uerr_b, n_users):
        return np.random.lognormal(uerr_a, uerr_b, n_users) / 2
    
    def sim_diff_fn(self, difficulty_a, difficulty_b):
        _, difficulty_dict = simulation.create_item_param_dicts(self.df.topic_item, 1, 1, difficulty_a, difficulty_b)
        return difficulty_dict