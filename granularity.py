import numpy as np
import pandas as pd
import copy
from collections import defaultdict
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.metrics.pairwise import euclidean_distances
from eval_functions import strdistance, _iou_score
from utils import flatten

def merge_strset(strings):
    sets = [set(s.split(" ")) for s in strings]
    unionset = set().union(*sets)
    return " ".join(list(unionset))

class TaggedString():
    def __init__(self, string, tag=None):
        self.string = string
        self.tag = tag
        self.numdims = 1
    def intersects(self, other):
        s1 = set(self.string.split(" "))
        s2 = set(other.string.split(" "))
        return len(s1.intersection(s2)) > 0
    def __getitem__(self, item):
        return self.string[item]
    def __lt__(self, other):
        return self.string < other.string
    def __eq__(self, other):
        if self.intersects(other):
            setstring = merge_strset([self.string, other.string])
            self.string = setstring
            other.string = setstring
            return True
        else:
            return False
    def __hash__(self):
        return 1
    def __repr__(self):
        return self.string

class VectorRange():
    def __init__(self, start_vector, end_vector, tag=None):
        assert len(start_vector) == len(end_vector)
        self.numdims = len(start_vector)
        for i in range(len(start_vector)):
            assert start_vector[i] <= end_vector[i]
        self.start_vector = start_vector
        self.end_vector = end_vector
        self.tag = tag
    def intersects(self, other):
        for i in range(self.numdims):
            if self.start_vector[i] > other.end_vector[i] or self.end_vector[i] < other.start_vector[i]:
                return False
        return True
    def centroid(self):
        return (np.array(self.start_vector) + np.array(self.end_vector)) / 2
    def __getitem__(self, item):
        return self.start_vector[item]
    def __lt__(self, other):
        return self.start_vector[0] < other.start_vector[0]
    def __eq__(self, other):
        assert self.numdims == other.numdims
        if self.intersects(other):
            for i in range(self.numdims):
                self.start_vector[i] = min(self.start_vector[i], other.start_vector[i])
                self.end_vector[i] = max(self.end_vector[i], other.end_vector[i])
                other.start_vector[i] = self.start_vector[i]
                other.end_vector[i] = self.end_vector[i]
            return True
        else:
            return False
    def __hash__(self):
        return 1
    def __repr__(self):
        return str((list(self.start_vector), list(self.end_vector)))

class SeqRange(VectorRange):
    def __init__(self, startend, tag=None):
        super().__init__([startend[0]], [startend[1]], tag)
        self.startend = startend
    def __getitem__(self, item):
        return self.startend[item]

def unionize_vectorrange_sequence(vectorranges, **kwargs):
    vectorranges = copy.deepcopy(vectorranges)
    for dim in range(vectorranges[0].numdims):
        sortedvectorranges = sorted(vectorranges, key=lambda x:x[dim])
        vectorranges = sorted(list(set(sortedvectorranges)))
    return vectorranges

def cluster_decomp(minilabels, dist_fn=None, n_clusters=None, plot_fn=None):
    if len(minilabels) < 2:
        if dist_fn is None:
            print(minilabels)
        return [np.array([0])]
        # return minilabels if dist_fn is None else [np.array([0])]
    if dist_fn is None:
        centroids = [vr.centroid() for vr in minilabels]
        dists = euclidean_distances(centroids)
        mean_dist = np.std(dists)
        clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                             distance_threshold=mean_dist if n_clusters is None else None)
        clustering.fit(centroids)
    else:
        dists = np.array([[dist_fn([a], [b]) for a in minilabels] for b in minilabels])
        # mean_dist = np.std(dists)
        mean_dist = np.mean(np.median(dists, axis=0))
        # print(mean_dist)
        clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                             distance_threshold=mean_dist if n_clusters is None else None,
                                             affinity="precomputed",
                                             linkage="average") # single
        clustering.fit(dists)
        # print(clustering.labels_)
        
    labels = clustering.labels_
    labeldict = defaultdict(list)
    for i, label in enumerate(labels):
        labeldict[label].append(i)
    result = []
    for indices in labeldict.values():
        # uv = unionize_vectorrange_sequence(np.array(minilabels)[np.array(indices)])
        # result += uv
        # uv = np.array(minilabels)[np.array(indices)]
        # result.append(uv)
        result.append(np.array(indices))
    
    if plot_fn is not None:
        colors = ["r", "b", "g", "y", "m", "c", "k"]
        for i, vr in enumerate(minilabels):
            plot_fn(vr, color=colors[labels[i] % len(colors)], alpha=0.5, text=labels[i])
        # for vr in result:
        #     plot_fn(vr, color="ko--", alpha=1)
        plt.gca().invert_yaxis()
        plt.show()
    return result

def fragment_by_overlaps(experiment, use_oracle=False, decomp_fn=unionize_vectorrange_sequence, dist_fn=None):
    return _fragment_by_overlaps(experiment.annodf,
                                 experiment.uid_colname,
                                 experiment.item_colname,
                                 experiment.label_colname,
                                 decomp_fn,
                                 dist_fn,
                                 experiment.golddict if use_oracle else None)

def _fragment_by_overlaps(annodf, uid_colname, item_colname, label_colname, decomp_fn, dist_fn=None, oracle_golddict=None):
    resultdfs = []
    for item_id in annodf[item_colname].unique():
        idf = annodf[annodf[item_colname] == item_id]
        vectorranges = [vr for annotation in idf[label_colname].values for vr in annotation]

        if oracle_golddict is not None:
            or_dist_fn = dist_center
            use_strings = False
            if hasattr(vectorranges[0], "string"):
                or_dist_fn = lambda x,y: strdistance(x.string, y.string)
                use_strings = True
            regions = []
            try:
                gold_vrs = [vr for vr in oracle_golddict.get(item_id)]
                orbuckets = {}
                for vr in vectorranges:
                    vr_orbucket = np.argmin([or_dist_fn(gold_vr, vr) for gold_vr in gold_vrs])
                    orbuckets.setdefault(vr_orbucket, []).append(vr)
                for orbucket in orbuckets.values():
                    if use_strings:
                        setstring = merge_strset([vr.string for vr in orbucket])
                        regions.append(TaggedString(setstring))
                    else:
                        minstart = np.min([vr.start_vector for vr in orbucket], axis=0)
                        maxend = np.max([vr.end_vector for vr in orbucket], axis=0)
                        regions.append(VectorRange(minstart, maxend))
            except Exception as e:
                print(e)
                pass
        else:
            regions = decomp_fn(vectorranges, dist_fn=dist_fn)
        origItemID = []
        newItemID = []
        newItemVR = []
        uid = []
        label = []
        gold = []
        for region in regions:
            for i, row in idf.iterrows():
                origItemID.append(item_id)
                newItemID.append(F"{item_id}-{region}")
                newItemVR.append(region)
                uid.append(row[uid_colname])
                label.append([vr for vr in row[label_colname] if region.intersects(vr)])
                gold.append(None)
        resultdfs.append(pd.DataFrame({"origItemID":origItemID, "newItemID":newItemID, "newItemVR":newItemVR, uid_colname:uid, label_colname:label, "gold":gold}))
    return pd.concat(resultdfs)

def decomposition(experiment, decomp_fn, plot_fn=None):
    resultdfs = []
    annodf = experiment.annodf
    uid_colname = experiment.uid_colname
    item_colname = experiment.item_colname
    label_colname = experiment.label_colname
    dist_fn = experiment.distance_fn
    resultdfs = []
    for item_id in annodf[item_colname].unique():
        idf = annodf[annodf[item_colname] == item_id]
        uids = []
        labels = []
        num_latent = []
        for _, row in idf[[uid_colname, label_colname]].iterrows():
            uid = row[uid_colname]
            uid_labels = row[label_colname]
            num_latent.append(len(uid_labels))
            for label in uid_labels:
                uids.append(uid)
                labels.append(label)
        est_num_latent = int(np.ceil(np.median(sorted(num_latent)[1:-1])) if len(num_latent) > 2 else np.max(num_latent))
        
        origItemID = []
        newItemID = []
        newItemVR = []
        region_label_indices_list = decomp_fn(labels, dist_fn, est_num_latent, plot_fn)
        uid_set = set(uids)
        for region_i, region_label_indices in enumerate(region_label_indices_list):
            region_uids = list(np.array(uids)[region_label_indices])
            region_labels = [[l] for l in np.array(labels)[region_label_indices]]
            remaining_uids = list(uid_set - set(region_uids))
            remaining_labels = [[]] * len(remaining_uids)
            region_uids += remaining_uids
            region_labels += remaining_labels
            dfdict = {
                        uid_colname: region_uids,
                        label_colname: region_labels,
                        }
            df = pd.DataFrame(dfdict)
            df = df.groupby(uid_colname).agg(flatten).reset_index()
            df["newItemID"] = F"{item_id}-{region_i}"
            df["origItemID"] = item_id
            resultdfs.append(df)
    return pd.concat(resultdfs)

def dist_center(vr1, vr2):
    vr1c = (np.array(vr1.start_vector) + np.array(vr1.end_vector)) / 2
    vr2c = (np.array(vr2.start_vector) + np.array(vr2.end_vector)) / 2
    return np.sum(np.abs(vr1c - vr2c))
