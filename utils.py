from itertools import combinations
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import pickle

def stanmodel(modelname, overwrite):
    import pystan
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
    orig = list(df[colname].values)
    if overwrite:
        df[colname] = pd.Categorical(df[colname]).codes
    return dict(zip(orig, df[colname]))

def flatten(listoflists):
    return [item for sublist in listoflists for item in sublist]

def translate_categorical(df, colname, coldict, drop_missing=True):
    df[colname] = np.array([coldict.get(i) for i in df[colname].dropna().values])
    result = df[df[colname] >= 0].copy()
    result[colname] = result[colname].astype(int)
    return result

def groups_of(df, colname, colvals=None):
    if colvals is None:
        colvals = df[colname].unique()
    gdf = df.groupby(colname)
    for colval in colvals:
        yield colval, gdf.get_group(colval)

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

def calc_distances(df, compare_fn, label_colname, item_colname, uid_colname="uid", bound=True):
    items = []
    u1s = []
    u2s = []
    a1s = []
    a2s = []
    distances = []
    n_labels = 0
    for item in tqdm(sorted(df[item_colname].unique())):
        idf = df[df[item_colname] == item]
        users = idf[uid_colname].unique()
        u_i = range(len(users))
        user_i_lookup = dict(zip(users, u_i))
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
            a1s.append(user_i_lookup.get(u1) + n_labels)
            a2s.append(user_i_lookup.get(u2) + n_labels)
        if len(users) > 1: # labels not used when no redundancy
            n_labels += len(users)
    if bound:
        distances = np.array(distances) + (.1 - np.min(distances))
        # distances = (np.array(distances) + 0.01 - np.min(distances)) / (np.max(distances) + 0.02 - np.min(distances))
    numnans = np.sum(np.isnan(distances))
    if numnans > 0:
        print(f"WARNING! FOUND {numnans} NAN DISTANCES!!")
    stan_data = {
        "items":np.array(items) + 1,
        "u1s":np.array(u1s) + 1,
        "u2s":np.array(u2s) + 1,
        "a1s":np.array(a1s) + 1,
        "a2s":np.array(a2s) + 1,
        "distances":distances,
    }
    stan_data["NDATA"] = len(stan_data["distances"])
    stan_data["NITEMS"] = np.max(np.unique(stan_data["items"]))
    stan_data["NUSERS"] = len(df[uid_colname].unique())
    stan_data["NLABELS"] = n_labels
    stan_data["n_gold_users"] = 0
    stan_data["gold_user_err"] = 0

    sdf = pd.DataFrame(stan_data)
    all_as = set(sdf["a1s"]).union(set(sdf["a2s"]))
    expected = set(range(1, sdf["NLABELS"].unique()[0]))
    empty = expected - all_as
    assert len(empty) == 0, F"we found some missing labels: {sorted(list(empty))}"

    return stan_data

def nancorr(v1, v2):
    return pd.DataFrame({"v1":v1, "v2":v2}).corr().values[0,1]

def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return np.array(m.T)

def bounded_cauchy(scale, shape, abs_bound):
    return np.maximum(np.minimum(np.random.standard_cauchy(shape) * scale, abs_bound), -abs_bound)

def proper_score(model_scores, gold_scores, score_fn=np.square):
    map_ps = model_scores / np.sum(model_scores)
    max_i = np.argmax(gold_scores)
    map_r = 1 - map_ps
    map_r[max_i] = map_ps[max_i]
    return np.mean(score_fn(map_r))
