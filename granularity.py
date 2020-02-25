import numpy as np
import pandas as pd
import copy

class VectorRange():
    def __init__(self, start_vector, end_vector, tag=None):
        assert len(start_vector) == len(end_vector)
        self.numdims = len(start_vector)
        for i in range(len(start_vector)):
            assert start_vector[i] <= end_vector[i]
        self.start_vector = start_vector
        self.end_vector = end_vector
    def __getitem__(self, item):
        return self.start_vector[item]
    def __lt__(self, other):
        return self.start_vector[0] < other.start_vector[0]
    def intersects(self, other):
        for i in range(self.numdims):
            if self.start_vector[i] > other.end_vector[i] or self.end_vector[i] < other.start_vector[i]:
                return False
        return True
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

def fragment_by_overlaps(annodf, uid_colname="uid", item_colname="item", label_colname="annotation", gold_colname="gold", include_gold=False):
    resultdfs = []
    for item_id in annodf[item_colname].unique():
        idf = annodf[annodf[item_colname] == item_id]
        vectorranges = [vr for annotation in idf[label_colname].values for vr in annotation]
        if include_gold:
            try:
                vectorranges += [vr for vr in idf[gold_colname].values[0]]
            except:
                continue
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
                if include_gold:
                    gold.append([vr for vr in row[gold_colname] if unrange.intersects(vr)])
                else:
                    gold.append(None)
        resultdfs.append(pd.DataFrame({"origItemID":origItemID, "newItemID":newItemID, "newItemVR":newItemVR, "uid":uid, "annotation":label, "gold":gold}))
    return pd.concat(resultdfs)
