import numpy as np
import pandas as pd
import copy

class TaggedString():
    def __init__(self, string, tag=None):
        self.string = string
        self.tag = tag
    def intersects(self, other):
        return self.string == other.string
        # return self.string in other.string or other.string in self.string
    def __getitem__(self, item):
        raise Exception("no get item")
    def __lt__(self, other):
        return self.string < other.string
    def __eq__(self, other):
        return self.string == other.string
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

def unionize_range_sequence(ranges):
    return sorted(list(set([SeqRange(x) for x in sorted(ranges)])))

def merge_vectorranges(annotations):
    non_empty = [anno for anno in annotations if len(anno) > 0]
    frac_non_empty = len(non_empty) / len(annotations)
    if frac_non_empty < 0.5:
        return []
    else:
        vrs = [vr for annotation in non_empty for vr in annotation]
        start = [int(x) for x in np.round(np.array([vr.start_vector for vr in vrs]).mean(axis=0))]
        end = [int(x) for x in np.round(np.array([vr.end_vector for vr in vrs]).mean(axis=0))]
        if isinstance(vrs[0], SeqRange):
            return [SeqRange((start, end))]
        else:
            return [VectorRange(start, end)]

def fragment_by_overlaps(experiment, use_oracle=False):
    return _fragment_by_overlaps(experiment.annodf,
                                 experiment.uid_colname,
                                 experiment.item_colname,
                                 experiment.label_colname,
                                 experiment.golddict if use_oracle else None)

def _fragment_by_overlaps(annodf, uid_colname, item_colname, label_colname, oracle_golddict=None):
    resultdfs = []
    for item_id in annodf[item_colname].unique():
        idf = annodf[annodf[item_colname] == item_id]
        vectorranges = [vr for annotation in idf[label_colname].values for vr in annotation]
        if oracle_golddict is not None:
            unranges = []
            try:
                gold_vrs = [vr for vr in oracle_golddict.get(item_id)]
                orbuckets = {}
                for vr in vectorranges:
                    vr_orbucket = np.argmin([dist_center(vr, gold_vr) for gold_vr in gold_vrs])
                    orbuckets.setdefault(vr_orbucket, []).append(vr)
                for orbucket in orbuckets.values():
                    minstart = np.min([vr.start_vector for vr in orbucket], axis=0)
                    maxend = np.max([vr.end_vector for vr in orbucket], axis=0)
                    unranges.append(VectorRange(minstart, maxend))
            except:
                pass
        else:
            unranges = unionize_vectorrange_sequence(vectorranges)
        origItemID = []
        newItemID = []
        newItemVR = []
        uid = []
        label = []
        gold = []
        for unrange in unranges:
            for i, row in idf.iterrows():
                origItemID.append(idf[item_colname].values[0])
                newItemID.append(F"{idf[item_colname].values[0]}-{unrange}")
                newItemVR.append(unrange)
                uid.append(row[uid_colname])
                label.append([vr for vr in row[label_colname] if unrange.intersects(vr)])
                gold.append(None)
        resultdfs.append(pd.DataFrame({"origItemID":origItemID, "newItemID":newItemID, "newItemVR":newItemVR, uid_colname:uid, label_colname:label, "gold":gold}))
    return pd.concat(resultdfs)

def dist_center(vr1, vr2):
    vr1c = (np.array(vr1.start_vector) + np.array(vr1.end_vector)) / 2
    vr2c = (np.array(vr2.start_vector) + np.array(vr2.end_vector)) / 2
    return np.sum(np.abs(vr1c - vr2c))
