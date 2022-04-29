import numpy as np
import pandas as pd
import simulation
import utils

def binary_match(x, y):
    return int(x == y)

def create_user_data(uid, df, pct_items, u_err, difficulty_dict=None, extraarg=None):
    ''' each worker assigned items '''
    items = df.item.unique()
    n_items_labeled = int(np.round(pct_items * len(items)))
    items_labeled = sorted(np.random.choice(items, n_items_labeled, replace=False))
    labels = []
    for item in items_labeled:
        idf = df[df.item == item]
        err = u_err
        if difficulty_dict is not None:
            i_difficulty = difficulty_dict.get(item)
            err += np.random.normal(0, i_difficulty)
        flipped = np.random.binomial(1, min(max(err, 0), 1))
        label = np.abs(idf.gold.values[0] - flipped)
        labels.append(label)
    dfdict = {
        "uid": [uid] * len(items_labeled),
        "item": items_labeled,
        "label": labels,
    }
    return pd.DataFrame(dfdict)

def create_item_data(item_i, df, n_users, pct_users, err_rates, difficulty_dict=None, extraarg=None):
    ''' each item assigned workers '''
    item = df.item.unique()[item_i]
    n_users_assigned = int(np.round(pct_users * n_users))
    users_assigned = sorted(np.random.choice(range(n_users), n_users_assigned, replace=False))
    labels = []
    for user in users_assigned:
        idf = df[df.item == item]
        err = err_rates[user]
        if difficulty_dict is not None:
            i_difficulty = difficulty_dict.get(item)
            err += np.random.normal(0, i_difficulty)
        flipped = np.random.binomial(1, min(max(err, 0), 1))
        label = np.abs(idf.gold.values[0] - flipped)
        labels.append(label)
    dfdict = {
        "uid": users_assigned,
        "item": [item] * len(users_assigned),
        "label": labels,
    }
    return pd.DataFrame(dfdict)

class BinarySimulator(simulation.Simulator):
    def __init__(self, n_items, p_true=0.5):
        self.gold = list(np.random.binomial(1, p_true, n_items))
        self.df = pd.DataFrame({"item":np.arange(len(self.gold)), "gold":self.gold})

    def create_stan_data(self, n_users, pct_items, err_rates, difficulty_dict):
        self.err_rates = err_rates
        self.difficulty_dict = difficulty_dict
        # self.sim_df = simulation.create_sim_df(create_user_data, self.df, n_users, pct_items, err_rates, difficulty_dict)
        self.sim_df = simulation.create_sim_df_by_item(create_item_data, self.df, n_users, len(self.gold), pct_items, err_rates, difficulty_dict)
        stan_data = utils.calc_distances(self.sim_df, (lambda x,y: 1 - self.eval_fn(x, y)), label_colname="label", item_colname="item")
        return stan_data

    def sim_uerr_fn(self, uerr_a, uerr_b, n_users):
        z = 0.5 * np.random.beta(uerr_a, uerr_b, 10000)
        result = np.quantile(z, np.linspace(0,1,n_users+2)[1:-1])
        return result

    def sim_diff_fn(self, difficulty_a, difficulty_b):
        z = 1 * np.random.beta(difficulty_a, difficulty_b, 10000)
        n_items = len(self.df["item"].unique())
        return dict(zip(np.arange(n_items), np.quantile(z, np.linspace(0,1,n_items+2)[1:-1])))
