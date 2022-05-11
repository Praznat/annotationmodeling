import pandas as pd
import numpy as np
from difflib import SequenceMatcher, ndiff
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from sklearn import metrics
from zss.compare import simple_distance as tree_edit_distance

def _get_children_nltktree(node):
    if isinstance(node[0], str):
        return []
    else:
        return [t for t in node]

def _get_label_nltktree(node):
    result = node.label()
    return result

def _label_distance(a, b):
    return 1 if a != b else 0

def _test_nltktree(node):
    for kid in _(node):
        _test_nltktree(kid)

def ted_distance(a, b):
    return tree_edit_distance(a, b,
                            get_children=_get_children_nltktree,
                            get_label=_get_label_nltktree,
                            label_dist=_label_distance) / 100

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
    # print(a_spans, "::::", b_spans)
    # print("____")
    if len(a_spans) * len(b_spans) == 0:
        return 0
    p = _eval_pred_per_gold(a_spans, b_spans, strict_range, strict_tag, str_spans)
    r = _eval_pred_per_gold(b_spans, a_spans, strict_range, strict_tag, str_spans)
    denom = (p + r)
    return 2 * p * r / denom if denom > 0 else 0

def _score_multi(thingsA, thingsB, score_fn, combine_fn=None, **kwargs):
    scoresA = [np.max([score_fn(thingA, thingB, **kwargs) for thingB in thingsB] + [0]) for thingA in thingsA]
    scoresB = [np.max([score_fn(thingA, thingB, **kwargs) for thingA in thingsA] + [0]) for thingB in thingsB]
    score = np.mean(scoresA + scoresB) if combine_fn is None else combine_fn(scoresA, scoresB)
    return 0 if np.isnan(score) else score

def _iou_score(vrA, vrB, use_G=False):
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
    U = float(boxAArea + boxBArea - interArea)
    iou = interArea / U

    if use_G:
        # determine the (x, y)-coordinates of the enclosing rectangle
        xA_C = min(vrA.start_vector[0], vrB.start_vector[0])
        yA_C = min(vrA.start_vector[1], vrB.start_vector[1])
        xB_C = max(vrA.end_vector[0], vrB.end_vector[0])
        yB_C = max(vrA.end_vector[1], vrB.end_vector[1])
        # compute the area of enclosing rectangle
        area_C = max(0, xB_C - xA_C + 1) * max(0, yB_C - yA_C + 1)
        iou -= (area_C - U) / area_C

    # return the intersection over union value
    return iou

def iou_score_multi(vrAs, vrBs, use_G=False):
    combine_fn = lambda x, y: (np.mean(x) + np.mean(y)) / 2
    return _score_multi(vrAs,
                        vrBs,
                        _iou_score,
                        combine_fn=combine_fn,
                        use_G=use_G)

def _iou_f1(vrA, vrB):
    return np.round(_iou_score(vrA, vrB))

def _f1_combine(a_bools, b_bools):
    p = np.sum(a_bools) / len(a_bools)
    r = np.sum(b_bools) / len(b_bools)
    return 2 * p * r / (p + r)

def iou_f1_multi(vrAs, vrBs):
    return _score_multi(vrAs, vrBs, _iou_f1, combine_fn=_f1_combine)

def _kp2vr(kp):
    from granularity import VectorRange
    if kp.sum() == 0:
        return None
    start, end = np.min(kp[kp.sum(axis=1)>0], axis=0), np.max(kp, axis=0)
    return VectorRange(start, end)

def iou_KP_multi(points1, points2, use_G=False):
    vrAs = [v for v in [_kp2vr(kp) for kp in points1] if v is not None]
    vrBs = [v for v in [_kp2vr(kp) for kp in points2] if v is not None]
    return _score_multi(vrAs, vrBs, _iou_score, use_G=use_G)

def _oks_score(points1, points2, area_scale=None, per_keypoint_constant=10):
    if area_scale is None:
        p1scale = np.sqrt(np.mean(np.std(points1, axis=0)**2))
        p2scale = np.sqrt(np.mean(np.std(points2, axis=0)**2))
        area_scale = (p1scale + p2scale) / 2
        if area_scale == 0:
            return 1
    denom = (area_scale * (per_keypoint_constant * 2)**2)
    if denom == 0:
        print("ZERO TIME", area_scale, per_keypoint_constant, points1, points2)
    e = (points1 - points2)**2 / denom
    return np.mean(np.exp(-np.mean(e, axis=1)))

def oks_score_multi(points1, points2, per_keypoint_constant=1):
    return _score_multi(points1, points2, _oks_score, per_keypoint_constant=per_keypoint_constant)

def rmse(x, y):
    return np.sqrt(np.mean(np.square(np.array(x) - np.array(y))))

def rmse_score_multi(points1, points2):
    return _score_multi(points1, points2, rmse)

smoother = SmoothingFunction()

def bleu(x, y):
    return sentence_bleu([x.split(" ")], y.split(" "), smoothing_function=smoother.method4)

def bleu2way(x, y):
    return (bleu(x, y) + bleu(y, x)) / 2

def bleu_multi(x, y):
    return sentence_bleu([xx.split(" ") for xx in x], y.split(" "), smoothing_function=smoother.method4)

def gleu(x, y):
    return sentence_gleu([x.split(" ")], y.split(" "))

def gleu2way(x, y):
    return (gleu(x, y) + gleu(y, x)) / 2

def gleu_multi(x, y):
    return sentence_gleu([xx.split(" ") for xx in x], y.split(" "))

def strdistance(a, b):
    return SequenceMatcher(None, a, b).ratio()

def levenshtein_distance(a, b):
    a = a.split(" ")
    b = b.split(" ")
    distance = 0
    buffer_removed = buffer_added = 0
    for x in ndiff(a, b):
        code = x[0]
        # Code ? is ignored as it does not translate to any modification
        if code == ' ':
            distance += max(buffer_removed, buffer_added)
            buffer_removed = buffer_added = 0
        elif code == '-':
            buffer_removed += 1
        elif code == '+':
            buffer_added += 1
    distance += max(buffer_removed, buffer_added)
    return distance

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
