from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_item_param_dicts(itemIds, beta_a, beta_b, lognorm_loc, lognorm_scale):
    beta_dict = {}
    lognorm_dict = {}
    for i in itemIds:
        beta_dict[i] = np.random.beta(beta_a, beta_b)
        lognorm_dict[i] = np.random.lognormal(lognorm_loc, lognorm_scale)
    return beta_dict, lognorm_dict

def create_sim_df_parallel(create_user_data_fn, df, n_users, pct_items, err_rates, difficulty_dict=None, extraarg=None):
    args = tuple([(i, df, pct_items, err_rates[i], difficulty_dict, extraarg) for i in range(n_users)])
    with Pool() as p:
        r = list(p.starmap(create_user_data_fn, args))
        return pd.concat([pd.DataFrame(d) for d in r]).to_dict(orient="list")

def create_sim_df(create_user_data_fn, df, n_users, pct_items, err_rates, difficulty_dict=None, extraarg=None):
    udatas = []
    for i in tqdm(range(n_users)):
        pct_items_u = pct_items if isinstance(pct_items, float) else pct_items.roll()
        udatas.append(create_user_data_fn(i, df, pct_items_u, err_rates[i], difficulty_dict=difficulty_dict, extraarg=extraarg))
    return pd.concat(udatas)

def create_sim_df_by_item(create_item_data_fn, df, n_users, n_items, pct_users, err_rates, difficulty_dict=None, extraarg=None):
    idatas = []
    for i in tqdm(range(n_items)):
        pct_users_i = pct_users if isinstance(pct_users, float) else pct_users.roll()
        idatas.append(create_item_data_fn(i, df, n_users, pct_users_i, err_rates, difficulty_dict=difficulty_dict, extraarg=extraarg))
    return pd.concat(idatas)

class BetaDist():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def roll(self):
        return np.random.beta(self.alpha, self.beta)

class Simulator():
    def __init__(self):
        pass
    
    def sim_uerr_fn(self, uerr_loc, uerr_scale, n_users):
        raise ValueError("Unimplemented")
        return []
    
    def sim_diff_fn(self, difficulty_a, difficulty_b):
        raise ValueError("Unimplemented")
        return []

    def create_stan_data(self, n_users, pct_items, err_rates, difficulty_dict):
        raise ValueError("Unimplemented")
        return {}

    def create_stan_data_scenario(self, n_users=10, pct_items=0.5,
                                    uerr_a=3, uerr_b=5,
                                    difficulty_a=5, difficulty_b=20,
                                    n_gold_users=0,
                                    sim_gold_user_err=0.01,
                                    model_gold_user_err=-4):
        err_rates = self.sim_uerr_fn(uerr_a, uerr_b, n_users)
        err_rates[:n_gold_users] = sim_gold_user_err # first n_gold assumed to be gold
        difficulty_dict = self.sim_diff_fn(difficulty_a, difficulty_b)
        stan_data = self.create_stan_data(n_users=n_users, pct_items=pct_items, err_rates=err_rates, difficulty_dict=difficulty_dict)
        stan_data["NDATA"] = len(stan_data["distances"])
        stan_data["NITEMS"] = np.max(np.unique(stan_data["items"]))
        stan_data["NUSERS"] = len(err_rates)
        stan_data["n_gold_users"] = n_gold_users
        stan_data["gold_user_err"] = model_gold_user_err
        return stan_data