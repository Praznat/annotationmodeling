import numpy as np
import pandas as pd

def sumprod_distance(data):
    s1 = data.groupby("u1s").sum()["wgtdistances1"]
    s2 = data.groupby("u2s").sum()["wgtdistances2"]
    n1 = data.groupby("u1s").sum()["u2score"]
    n2 = data.groupby("u2s").sum()["u1score"]
    return s1.add(s2, fill_value=0) / n1.add(n2, fill_value=0)

# calc annotation scores
def calc_anno_scores(data):
    data["wgtdistances1"] = data["distances"] * data["u2score"]
    data["wgtdistances2"] = data["distances"] * data["u1score"]

# calc user scores
def calc_user_scores(data):
    user_distances = sumprod_distance(data)
    data["u1score"] = 1 - user_distances[data["u1s"]].values
    data["u2score"] = 1 - user_distances[data["u2s"]].values

def score_all(stan_data, num_rounds=3, init_userscores=None):
    distance_df = pd.DataFrame(stan_data)
    distance_df["u1s"] = distance_df["u1s"] - 1
    distance_df["u2s"] = distance_df["u2s"] - 1
    distance_df["items"] = distance_df["items"] - 1
    distance_df["u1score"] = 1
    distance_df["u2score"] = 1
    distance_df["wgtdistances1"] = distance_df["distances"]
    distance_df["wgtdistances2"] = distance_df["distances"]
    
    for round in range(num_rounds):
        calc_user_scores(distance_df)
        calc_anno_scores(distance_df)

    return distance_df.groupby("items").apply(sumprod_distance)

def pick_best_user(data):
    users = np.array([x[1] for x in data.index])
    return users[np.argsort(data).values]

def per_item_user_rankings(scoreall_scores):
    return scoreall_scores.groupby("items").apply(pick_best_user)