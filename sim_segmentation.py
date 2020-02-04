import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import simulation
import utils

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def create_user_data(uid, df, pct_items, u_err, difficulty_dict=None, extraarg=None):
    items = df["img"].unique()
    n_items_labeled = int(np.round(pct_items * len(items)))
    items_labeled = sorted(np.random.choice(items, n_items_labeled, replace=False))
    labels = []
    for item in items_labeled:
        idf = df[df["img"] == item]
        idflen = len(idf["goldcoords"].values[0])
        err_scale = u_err
        if difficulty_dict is not None:
            i_difficulty = difficulty_dict.get(item)
            err_scale *= np.exp(i_difficulty)
        err = np.random.normal(0, err_scale, idflen)
        label = idf["goldcoords"].values[0] + err
        labels.append(label)
    dfdict = {
        "uid": [uid] * len(items_labeled),
        "item": items_labeled,
        "label": labels,
    }
    return pd.DataFrame(dfdict)

class SegmentationSimulator(simulation.Simulator):
    def __init__(self, rawdata_dir, max_items=10000):
        self.df = pd.read_csv(rawdata_dir, error_bad_lines=False, header=None, sep=" ", names=["img", "x", "y", "w", "h"])
        self.df = self.df[:max_items]
        self.df["goldcoords"] = self.df.apply(lambda row: [row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]], axis=1)
        self.img_lookup = utils.make_categorical(self.df, "img")
        self.gold = self.df.set_index("img")["goldcoords"]

    def create_stan_data(self, n_users, pct_items, err_rates, difficulty_dict):
        self.err_rates = err_rates
        self.difficulty_dict = difficulty_dict
        self.sim_df = simulation.create_sim_df(create_user_data, self.df, n_users, pct_items, err_rates, difficulty_dict)
        stan_data = utils.calc_distances(self.sim_df, (lambda x,y: 1 - bb_intersection_over_union(x, y)))
        return stan_data

    def sim_uerr_fn(self, uerr_a, uerr_b, n_users):
        z = np.abs(np.random.normal(uerr_a, uerr_b, 10000))
        return np.quantile(z, np.linspace(0,1,n_users+2)[1:-1])
    
    def sim_diff_fn(self, difficulty_a, difficulty_b):
        z = 1 * np.random.beta(difficulty_a, difficulty_b, 10000)
        n_items = len(self.df["img"].unique())
        return dict(zip(np.arange(n_items), np.quantile(z, np.linspace(0,1,n_items+2)[1:-1])))