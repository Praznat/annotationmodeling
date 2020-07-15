import numpy as np
import pandas as pd
import copy
from collections import defaultdict
import sklearn
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.metrics.pairwise import euclidean_distances
from eval_functions import strdistance, _iou_score
from utils import plot_vectorrange

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

def unionize_vectorrange_sequence(vectorranges):
    vectorranges = copy.deepcopy(vectorranges)
    for dim in range(vectorranges[0].numdims):
        sortedvectorranges = sorted(vectorranges, key=lambda x:x[dim])
        vectorranges = sorted(list(set(sortedvectorranges)))
    return vectorranges

def cluster_decomp(vectorranges, use_centroids=True, do_plot=False):
    if len(vectorranges) < 2:
        return vectorranges
    if use_centroids:
        centroids = [vr.centroid() for vr in vectorranges]
        dists = euclidean_distances(centroids)
        mean_dist = np.std(dists)
        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=mean_dist)
        clustering.fit(centroids)
    else:
        dists = np.array([[1 - _iou_score(a, b) for a in vectorranges] for b in vectorranges])
        mean_dist = np.std(dists)
        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=mean_dist,
                                             affinity="precomputed",
                                             linkage="average")
        clustering.fit(dists)
        
    labels = clustering.labels_
    labeldict = defaultdict(list)
    for i, label in enumerate(labels):
        labeldict[label].append(i)
    result = []
    for indices in labeldict.values():
        uv = unionize_vectorrange_sequence(np.array(vectorranges)[np.array(indices)])
        result += uv
    
    if do_plot:
        colors = ["r", "b", "g", "y", "m", "c", "k", "k--"]
        try:
            for i, vr in enumerate(vectorranges):
                plot_vectorrange(vr, color=colors[labels[i]])
            for vr in result:
                plot_vectorrange(vr, color="ko--", alpha=1)
            plt.show()
        except Exception as e:
            print(labels)
            print(e)
            pass
    return result

def fragment_by_overlaps(experiment, use_oracle=False, decomp_fn=unionize_vectorrange_sequence):
    return _fragment_by_overlaps(experiment.annodf,
                                 experiment.uid_colname,
                                 experiment.item_colname,
                                 experiment.label_colname,
                                 decomp_fn,
                                 experiment.golddict if use_oracle else None)

def _fragment_by_overlaps(annodf, uid_colname, item_colname, label_colname, decomp_fn, oracle_golddict=None):
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
            regions = decomp_fn(vectorranges)
        origItemID = []
        newItemID = []
        newItemVR = []
        uid = []
        label = []
        gold = []
        for region in regions:
            for i, row in idf.iterrows():
                origItemID.append(idf[item_colname].values[0])
                newItemID.append(F"{idf[item_colname].values[0]}-{region}")
                newItemVR.append(region)
                uid.append(row[uid_colname])
                label.append([vr for vr in row[label_colname] if region.intersects(vr)])
                gold.append(None)
        resultdfs.append(pd.DataFrame({"origItemID":origItemID, "newItemID":newItemID, "newItemVR":newItemVR, uid_colname:uid, label_colname:label, "gold":gold}))
    return pd.concat(resultdfs)

def dist_center(vr1, vr2):
    vr1c = (np.array(vr1.start_vector) + np.array(vr1.end_vector)) / 2
    vr2c = (np.array(vr2.start_vector) + np.array(vr2.end_vector)) / 2
    return np.sum(np.abs(vr1c - vr2c))
