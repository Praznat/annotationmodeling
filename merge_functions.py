import pandas as pd
import numpy as np
from granularity import SeqRange, VectorRange
from utils import flatten

def numerical_mean(values, weights):
    return np.array([np.array(values[i]) * weights[i] for i in range(len(values))]).sum(axis=0) / np.sum(weights)

def borda_count(values, weights):
    # input values are ranked lists of integers
    n = len(values[0])
    points = np.zeros(np.max([np.max(v) for v in values])+1)
    for rankedlist, weight in zip(values, weights):
        for i, element in enumerate(rankedlist):
            points[element] += (n - i) * weight
    return np.argsort(-points)

def keypoint_merge(annotations, weights=None):
    non_empty = []
    non_empty_wgts = []
    for i in range(len(annotations)):
        anno = annotations[i]
        if len(anno) > 0:
            non_empty.append(anno)
            non_empty_wgts.append(weights[i])
    # if only one non-empty annotation, return empty
    if len(non_empty_wgts) < 2 and sum(non_empty_wgts) / sum(weights) < 0.25:
        return []
    else:
        sum_kp = 0
        sum_wgt = 0
        for annotation, weight in zip(non_empty, non_empty_wgts):
            sum_kp += annotation[0] * weight # randomly pick first one
            sum_wgt += weight
        return [sum_kp / sum_wgt]

def _wgt_median(values, weights):
    df = pd.DataFrame({"values":values, "weights":weights}).sort_values('values')
    cumsum = df["weights"].cumsum()
    cutoff = df["weights"].sum() / 2.0
    median = df["values"].values[cumsum >= cutoff][0]
    return median

def vectorrange_merge(annotations, weights=None):
    non_empty = []
    non_empty_wgts = []
    for i in range(len(annotations)):
        anno = annotations[i]
        if len(anno) > 0:
            non_empty.append(anno)
            non_empty_wgts.append(weights[i])
    # if only one non-empty annotation, return empty
    if len(non_empty_wgts) < 2 and sum(non_empty_wgts) / sum(weights) < 0.5:
        return []
    # if majority of annotations are empty, return empty
    # if sum(non_empty_wgts) / sum(weights) < 0.5:
    #     return []
    else:
        # WEIGHTED MEDIAN
        all_weights = []
        for annotation, weight in zip(non_empty, non_empty_wgts):
            all_weights += [weight for vr in annotation]
        all_annotations = flatten(non_empty)
        starts = np.array([vr.start_vector for vr in all_annotations])
        ends = np.array([vr.end_vector for vr in all_annotations])
        dims = starts.shape[1]
        start = [_wgt_median(starts[:,dim], all_weights) for dim in range(dims)]
        end = [_wgt_median(ends[:,dim], all_weights) for dim in range(dims)]

        # WEIGHTED MEAN
        # sum_start = 0
        # sum_end = 0
        # sum_wgt = 0
        # for annotation, weight in zip(non_empty, non_empty_wgts):
        #     umeanstart = np.array([vr.start_vector for vr in annotation]).mean(axis=0)
        #     umeanend = np.array([vr.end_vector for vr in annotation]).mean(axis=0)
        #     sum_start += umeanstart * weight
        #     sum_end += umeanend * weight
        #     sum_wgt += weight
        # start = np.round(sum_start / sum_wgt).astype(int)
        # end = np.round(sum_end / sum_wgt).astype(int)
        
        if len(start) == 1:
            return [SeqRange((start[0], end[0]))]
        else:
            return [VectorRange(start, end)]