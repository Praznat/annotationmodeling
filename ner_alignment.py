import numpy as np

def align(anno_df, gold_df, debug = False):
    merged_df = anno_df.merge(gold_df, on="item")
    new_annos = []
    new_golds = []
    for _, row in merged_df.iterrows():
        anno, gold = row[["clean_label_x", "clean_label_y"]]
        anno = [t for t in anno]
        gold = [t for t in gold]
        anno_t = 0
        gold_t = 0
        try:
            new_anno = []
            new_gold = []
            anno_t = anno.pop(0)
            gold_t = gold.pop(0)
            hit_partial = None
            while anno or gold:
                if anno_t[0] == gold_t[0]: # words match perfectly
                    if debug:
                        print(anno_t[0], "==", gold_t[0])
                    new_anno.append(anno_t)
                    new_gold.append(gold_t)
                    anno_t = anno.pop(0)
                    gold_t = gold.pop(0)
                elif anno_t[0] in gold_t[0] and not hit_partial:
                    if debug:
                        print(anno_t[0], "<", gold_t[0])
                    new_anno.append(anno_t)
                    new_gold.append(gold_t)
                    anno_t = anno.pop(0)
                    hit_partial = anno
                elif gold_t[0] in anno_t[0] and not hit_partial:
                    if debug:
                        print(anno_t[0], ">", gold_t[0])
                    new_anno.append(anno_t)
                    new_gold.append(gold_t)
                    gold_t = gold.pop(0)
                    hit_partial = gold
                elif anno_t[0] in gold_t[0] and hit_partial is anno:
                    if debug:
                        print(anno_t[0], "<<", gold_t[0])
                    anno_t = anno.pop(0)
                elif gold_t[0] in anno_t[0] and hit_partial is gold:
                    if debug:
                        print(anno_t[0], ">>", gold_t[0])
                    gold_t = gold.pop(0)
                elif anno_t[0] not in gold_t[0] and hit_partial is anno:
                    if debug:
                        print(anno_t[0], "<!", gold_t[0])
                    gold_t = gold.pop(0)
                elif gold_t[0] not in anno_t[0] and hit_partial is gold:
                    if debug:
                        print(anno_t[0], ">!", gold_t[0])
                    anno_t = anno.pop(0)
                else: # no match at all
                    if debug:
                        print(anno_t[0], "!=", gold_t[0])
                    anno_t = anno.pop(0)
                    gold_t = gold.pop(0)
            if debug:
                print([t[0] for t in new_anno])
                print("----")
                print([t[0] for t in new_gold])
        except Exception as e:
            print(e)
            new_anno = np.nan
            new_gold = np.nan
        new_annos.append(new_anno)
        new_golds.append(new_gold)
    merged_df["annotation"] = new_annos
    merged_df["gold"] = new_golds
    merged_df["worker"] = merged_df["worker_x"]
    return merged_df.dropna()