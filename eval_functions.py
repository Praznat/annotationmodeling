import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from sklearn import metrics

def _tag(x, strict):
    return x or "" if strict else ""

def _spanrange(span, str_spans=True):
    return span.string.split(" ") if str_spans else range(*span.startend)

def _labels2tokenset(spans, strict_tag, str_spans):
    ranges = [[str(t) + "_" + _tag(s.tag, strict_tag) for t in _spanrange(s, str_spans)] for s in spans]
    return set([y for x in ranges for y in x])

def _labels2rangeset(spans, strict_tag):
    return set([str(s.startend) + "_" + _tag(s.tag, strict_tag) for s in spans])

def _tokenintersects_per_span(denom_spans, nom_spans, strict_tag, str_spans):
    denom_sets = [_labels2tokenset([a], strict_tag, str_spans) for a in denom_spans]
    nom_set = _labels2tokenset(nom_spans, strict_tag, str_spans)
    scores = [len(denom_set.intersection(nom_set)) / len(denom_set) for denom_set in denom_sets]
    return np.mean(scores)

def _exact_intersects_per_ranges(denom_spans, nom_spans, strict_tag):
    denom_set = _labels2rangeset(denom_spans, strict_tag)
    nom_set = _labels2rangeset(nom_spans, strict_tag)
    return len(nom_set.intersection(denom_set)) / len(nom_set)

def _eval_pred_per_gold(pred_spans, gold_spans, strict_range, strict_tag, str_spans):
    if strict_range:
        return _exact_intersects_per_ranges(pred_spans, gold_spans, strict_tag)
    else:
        return _tokenintersects_per_span(pred_spans, gold_spans, strict_tag, str_spans)

def eval_f1(a_spans, b_spans, strict_range, strict_tag, str_spans):
    if len(a_spans) * len(b_spans) == 0:
        return 0
    p = _eval_pred_per_gold(a_spans, b_spans, strict_range, strict_tag, str_spans)
    r = _eval_pred_per_gold(b_spans, a_spans, strict_range, strict_tag, str_spans)
    denom = (p + r)
    return 2 * p * r / denom if denom > 0 else 0

def _iou_score(vrA, vrB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(vrA.start_vector[0], vrB.start_vector[0])
    yA = max(vrA.start_vector[1], vrB.start_vector[1])
    xB = min(vrA.end_vector[0], vrB.end_vector[0])
    yB = min(vrA.end_vector[1], vrB.end_vector[1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (vrA.end_vector[0] - vrA.start_vector[0] + 1) * (vrA.end_vector[1] - vrA.start_vector[1] + 1)
    boxBArea = (vrB.end_vector[0] - vrB.start_vector[0] + 1) * (vrB.end_vector[1] - vrB.start_vector[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def iou_score_multi(vrAs, vrBs):
    scoresA = [np.max([_iou_score(vrA, vrB) for vrB in vrBs] + [0]) for vrA in vrAs]
    scoresB = [np.max([_iou_score(vrA, vrB) for vrA in vrAs] + [0]) for vrB in vrBs]
    score = np.mean(scoresA + scoresB)
    return 0 if np.isnan(score) else score


def rmse(x, y):
    return np.sqrt(np.mean(np.square(np.array(x) / 100 - np.array(y) / 100)))


smoother = SmoothingFunction()

def _bleu(x, y):
    return sentence_bleu([x.split(" ")], y.split(" "), smoothing_function=smoother.method4)

def bleu2way(x, y):
    return (_bleu(x, y) + _bleu(y, x)) / 2

def bleu_multi(x, y):
    return sentence_bleu([xx.split(" ") for xx in x], y.split(" "), smoothing_function=smoother.method4)

def _gleu(x, y):
    return sentence_gleu([x.split(" ")], y.split(" "))

def gleu2way(x, y):
    return (_gleu(x, y) + _gleu(y, x)) / 2

def gleu_multi(x, y):
    return sentence_gleu([xx.split(" ") for xx in x], y.split(" "))

def strdistance(a, b):
    return SequenceMatcher(None, a, b).ratio()

#eval functions that operate on dictionaries
def apply_metric(gold_dict, pred_dict, metric, **kwargs):
    gold_list = []
    preds_list = []
    for k, v in gold_dict.items():
        gold_list.append(v)
        preds_list.append(pred_dict[k])
    return metric(gold_list, preds_list, **kwargs)

accuracy    = lambda gold, preds: apply_metric(gold, preds, metrics.accuracy_score)
f1_weighted = lambda gold, preds: apply_metric(gold, preds, metrics.f1_score, average='weighted')
f1_macro    = lambda gold, preds: apply_metric(gold, preds, metrics.f1_score, average='macro')
mae         = lambda gold, preds: apply_metric(gold, preds, metrics.mean_absolute_error)
mse         = lambda gold, preds: apply_metric(gold, preds, metrics.mean_squared_error)

def score_predictions(gold_dict, pred_dict, eval_fn):
    return {k: eval_fn(gold_dict[k], pred_dict[k]) for k, v in gold_dict.items()}
