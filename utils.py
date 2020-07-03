from itertools import combinations
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from matplotlib import pyplot as plt
import pystan
from sklearn.decomposition import PCA

def stanmodel(modelname, overwrite):
    picklefile = modelname + ".pkl"
    try:
        if overwrite:
            raise IOError("Overwriting picked files")
        stan_model = pickle.load(open(picklefile, 'rb'))
        print("Pickled model loaded")
    except (OSError, IOError):
        print("Pickled model not found")
        print("Compiling model")
        stan_model = pystan.StanModel(file=modelname + ".stan")
        with open(picklefile, 'wb') as f:
            print("Pickling model")
            pickle.dump(stan_model, f)
    return stan_model

def make_categorical(df, colname, overwrite=True):
    orig = df[colname].values
    if overwrite:
        df[colname] = pd.Categorical(df[colname]).codes
    return dict(zip(orig, df[colname]))

def translate_categorical(df, colname, coldict, drop_missing=True):
    df[colname] = np.array([coldict.get(i) for i in df[colname].dropna().values])
    result = df[df[colname] >= 0].copy()
    result[colname] = result[colname].astype(int)
    return result

# def calc_distances_foritem(idf, compare_fn, label_colname, item_colname, uid_colname):
#     users = idf[uid_colname].unique()
#     items = []
#     u1s = []
#     u2s = []
#     distances = []
#     for u1, u2 in combinations(users, 2):
#         p1 = idf[idf[uid_colname]==u1][label_colname].values[0]
#         p2 = idf[idf[uid_colname]==u2][label_colname].values[0]
#         distance = compare_fn(p1, p2)
#         items.append(idf[item_colname].values[0])
#         u1s.append(u1)
#         u2s.append(u2)
#         distances.append(distance)
#     distances /= 2
#     distances = np.array(distances) + (.1 - np.min(distances))
#     return {
#         "items":np.array(items) + 1,
#         "u1s":np.array(u1s) + 1,
#         "u2s":np.array(u2s) + 1,
#         "distances":distances
#     }
    
# def calc_distances_parallel(df, compare_fn, label_colname, item_colname, uid_colname="uid"):
#     items = df[item_colname].unique()
#     args = tuple([(df[df[item_colname]==i], compare_fn, label_colname, item_colname, uid_colname) for i in items])
#     with Pool() as p:
#         r = list(p.starmap(calc_distances_foritem, args))
#         return pd.concat([pd.DataFrame(d) for d in r]).to_dict(orient="list")

def calc_distances(df, compare_fn, label_colname="label", item_colname="item", uid_colname="uid"):
    items = []
    u1s = []
    u2s = []
    distances = []
    for item in tqdm(df[item_colname].unique()):
        idf = df[df[item_colname] == item]
        users = idf[uid_colname].unique()
        for u1, u2 in combinations(users, 2):
            p1 = idf[idf[uid_colname]==u1][label_colname].values[0]
            p2 = idf[idf[uid_colname]==u2][label_colname].values[0]
            distance = compare_fn(p1, p2)
            if np.isnan(distance):
                print(f"WARNING: NAN DISTANCE BETWEEN {p1} and {p2}")
            items.append(item)
            u1s.append(u1)
            u2s.append(u2)
            distances.append(distance)
    distances = np.array(distances) + (.1 - np.min(distances))
    numnans = np.sum(np.isnan(distances))
    if numnans > 0:
        print(f"WARNING! FOUND {numnans} NAN DISTANCES!!")
    stan_data = {
        "items":np.array(items) + 1,
        "u1s":np.array(u1s) + 1,
        "u2s":np.array(u2s) + 1,
        "distances":distances
    }
    stan_data["NDATA"] = len(stan_data["distances"])
    stan_data["NITEMS"] = np.max(np.unique(stan_data["items"]))
    stan_data["NUSERS"] = len(df[uid_colname].unique())
    stan_data["n_gold_users"] = 0
    stan_data["gold_user_err"] = 0
    return stan_data

def proper_score(model_scores, gold_scores, score_fn=np.square):
    map_ps = model_scores / np.sum(model_scores)
    max_i = np.argmax(gold_scores)
    map_r = 1 - map_ps
    map_r[max_i] = map_ps[max_i]
    return np.mean(score_fn(map_r))

def visualize_embeddings(stan_data, opt, sim_df=None, preds={}):
    from sklearn.decomposition import PCA
    def userset(data):
        result = set(data["u1s"]).union(set(data["u2s"]))
        return result
    sddf = pd.DataFrame(stan_data)
    item_userset = sddf.groupby("items").apply(userset)
    for i, iue in enumerate(opt["item_user_errors"]):
        users = item_userset.get(i+1)
        if users is None or len(users) < 3:
            continue
        dist_from_truth = opt["dist_from_truth"][i]
        print("item", str(i+1))
        print(sddf[sddf["items"]==i+1][["u1s", "u2s", "distances"]])
        if len(iue[0]) > 2:
            embeddings = PCA(n_components=2).fit_transform(iue)
        embeddings = np.array([embeddings[u-1] for u in users])
        dists = [dist_from_truth[u-1] for u in users]
        skills = [opt["uerr"][u-1] for u in users]
        scale = np.max(np.abs(embeddings)) * 1.05
        plt.scatter(embeddings[:,0], embeddings[:,1])
        for ui, emb in enumerate(embeddings):
            plt.plot([0,emb[0]], [0,emb[1]], "b")
            plt.annotate(str(list(users)[ui]) + ":" + str(np.round(dists[ui],2)) + ":" + str(np.round(skills[ui],2)), emb)
        if sim_df is not None:
            preds.get(i)
            sim_df[sim_df.topic_item==0]
        plt.xlim(-scale, scale)
        plt.ylim(-scale, scale)
        plt.show()

def plot_vectorrange(vr, color="b", alpha=0.3):
    plt.plot([vr.start_vector[0], vr.end_vector[0]], [vr.start_vector[1], vr.start_vector[1]], color, alpha=alpha)
    plt.plot([vr.start_vector[0], vr.start_vector[0]], [vr.start_vector[1], vr.end_vector[1]], color, alpha=alpha)
    plt.plot([vr.start_vector[0], vr.end_vector[0]], [vr.end_vector[1], vr.end_vector[1]], color, alpha=alpha)
    plt.plot([vr.end_vector[0], vr.end_vector[0]], [vr.start_vector[1], vr.end_vector[1]], color, alpha=alpha)

def plot_annos(data, expmnt):
    vrs = [vr for annotation in data["annotation"] for vr in annotation]
    for vr in vrs:
        plot_vectorrange(vr)
    item = data["item"].values[0]
    try:
        for gvr in expmnt.golddict.get(item):
            plot_vectorrange(gvr, "yo--", alpha=1)
    except:
        pass
    plt.title(data["item"].values[0])
    plt.show()