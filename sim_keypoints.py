import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import simulation
from eval_functions import oks_score_multi
import utils

def alter_location(points, x_offset, y_offset):
    x, y = points.T
    return np.array([x + x_offset, y + y_offset]).T

def alter_rotation(points, radians):
    centroid = np.mean(points, axis=0)
    return utils.rotate_via_numpy((points - centroid).T, radians) + centroid

def alter_magnitude(points, percent_diff):
    centroid = np.mean(points, axis=0)
    return (points - centroid) * np.exp(percent_diff) + centroid

def alter_normal_jump(points, scale):
    return points + np.random.normal(0, scale, points.shape)

def alter_cauchy_jump(points, scale, abs_bound):
    return points + utils.bounded_cauchy(scale, points.shape, abs_bound)

def disappear(points, p_disappear):
    return None if np.random.uniform() < p_disappear else points

def shift_by_uerr(annotation, uerr):
    shifts = [
        alter_rotation(annotation, np.random.normal(0, 0.5 * uerr) * np.pi / 8),
        alter_magnitude(annotation, np.random.normal(0, 0.3 * uerr)),
        alter_normal_jump(annotation, 30 * uerr),
        alter_cauchy_jump(annotation, 30 * uerr, 100),
    ]
    return np.mean(shifts, axis=0) * np.abs(np.sign(annotation))

def create_user_data(uid, df, pct_items, u_err, difficulty_dict=None, extraarg=None):
    items = df["item"].unique()
    n_items_labeled = int(np.round(pct_items * len(items)))
    items_labeled = sorted(np.random.choice(items, n_items_labeled, replace=False))
    labels = []
    for item in items_labeled:
        gold = df[df["item"] == item]["gold"].values[0]
        shifted_kpobjs = [shift_by_uerr(kpobj, u_err) for kpobj in gold]
        kpobjs = [shifted_kpobjs[0]] + [disappear(kp, u_err / 2) for kp in shifted_kpobjs[1:]]
        kpobjs = [kp for kp in kpobjs if kp is not None]
        labels.append(kpobjs)
    dfdict = {
        "uid": [uid] * len(items_labeled),
        "item": items_labeled,
        "annotation": labels,
    }
    return pd.DataFrame(dfdict)

class KeypointSimulator(simulation.Simulator):
    def __init__(self, rawdata_dir='data/coco/person_keypoints_train2017.json', max_items=500, minlabelsperitem=4):
        with open(rawdata_dir) as f:
            dataset = json.load(f)
        self.category_id_skeletons = {c["id"]: np.array(c["skeleton"])-1 for c in iter(dataset["categories"])}
        
        img_label = {}
        for dataset_annotation in iter(dataset["annotations"]):
            v = img_label.setdefault(dataset_annotation["image_id"], [])
            v.append(dataset_annotation)
        img_label_minlen = {k: v for k, v in img_label.items() if len(v) >= minlabelsperitem}  
        
        i = 0
        rows = []
        item = []
        annotation = []
        category = []
        for dataset_annotations in iter(img_label_minlen.values()):
            for dataset_annotation in dataset_annotations:
                kp = np.reshape(dataset_annotation["keypoints"], (-1,3))
                kp = kp[kp[:,2]>-90][:,:2]
                if len(kp) == 0:
                    continue
                item.append(dataset_annotation["image_id"])
                annotation.append(kp)
                category.append(dataset_annotation["category_id"])
            i += 1
            if i > max_items:
                break
        kp_df = pd.DataFrame({"item":item, "gold":annotation, "category":category})
        self.df = kp_df.groupby("item")["gold"].apply(list).reset_index()
        self.itemdict = utils.make_categorical(self.df, "item")

    def create_stan_data(self, n_users, pct_items, err_rates, difficulty_dict):
        self.err_rates = err_rates
        self.difficulty_dict = difficulty_dict
        self.sim_df = simulation.create_sim_df(create_user_data, self.df, n_users, pct_items, err_rates, difficulty_dict)
        stan_data = utils.calc_distances(self.sim_df, (lambda x,y: 1 - self.eval_fn(x, y)), label_colname="annotation", item_colname="item")
        return stan_data

    def sim_uerr_fn(self, uerr_a, uerr_b, n_users):
        z = np.abs(np.random.normal(uerr_a, uerr_b, 10000))
        return np.quantile(z, np.linspace(0,1,n_users+2)[1:-1])
    
    def sim_diff_fn(self, difficulty_a, difficulty_b):
        z = 1 * np.random.beta(difficulty_a, difficulty_b, 10000)
        n_items = len(self.df["item"].unique())
        return dict(zip(np.arange(n_items), np.quantile(z, np.linspace(0,1,n_items+2)[1:-1])))