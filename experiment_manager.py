import os
import json
import random
import pandas as pd
import numpy as np
import experiments
import utils
import granularity
from granularity import SeqRange, VectorRange, TaggedString, cluster_decomp
from eval_functions import eval_f1, iou_score_multi, rmse
import merge_functions


def label2tvr(label, default=None):
    return default if label is None else [SeqRange(l) for l in label]
    # return default if label is None else [{"range":SeqRange(l), "tag":None} for l in label]


def convert2vr(annotations):
    result = []
    for annotation in annotations:
        left = annotation["left"]
        top = annotation["top"]
        if annotation["width"] > 0 and annotation["height"] > 0:
            result.append(VectorRange([left, top], [left + annotation["width"], top + annotation["height"]]))
        else:
            return np.nan
    return result


class ExperimentResult(pd.DataFrame):
    '''
    Output experiment results together as csv/dataframe with following columns:
    * Dataset name
    * Evaluation function name
    * Distance function name
    * Aggregation method name
    * Predicted annotations for all items
    * Total average evaluation score
    * Extra arguments
    '''
    def __init__(self, dataset_name, eval_fn_name, dist_fn_name, agg_method_name, preds, score, ss, extra=None):
        result = {
            "Dataset name": [dataset_name],
            "Eval Fn name": [eval_fn_name],
            "Dist Fn name": [dist_fn_name],
            "Agg method name": [agg_method_name],
            "Predicted": [preds],
            "Eval score": [score],
            "Statistical significance": [ss],
            "Misc.": [extra]
        }
        super().__init__(result)

class SimpleExperiment(experiments.RealExperiment):
    def __init__(self, **kwargs):
        eval_fn = kwargs['eval_fn']
        dist_fn = kwargs['dist_fn']
        super().__init__(eval_fn=eval_fn,
                                label_colname='answer',
                                item_colname='question',
                                uid_colname='worker',
                                distance_fn=dist_fn)

    def set_merge_fn(self, merge_fn):
        self.merge_fn = merge_fn

class AffectExperiment(experiments.RealExperiment):
    
    def __init__(self, **kwargs):
        super().__init__(lambda x,y: 1 / rmse(x,y) , "annotation", "item", "uid", rmse)
        self.data_dir = "data/snow_affect/"
        self.merge_fn = merge_functions.numerical_mean

    def setup(self):
        emotions = ["surprise", "disgust", "sadness", "fear", "valence", "joy", "anger"]
        def load_snow(relfilepath):
            return pd.read_csv(relfilepath, sep="\t").set_index("!amt_annotation_ids")
        dfs = [load_snow(self.data_dir + f + ".standardized.tsv") for f in emotions]
        full_df = pd.concat(dfs, join="inner", axis=1)

        full_df["annotation"] = full_df["response"].values.tolist()
        full_df["groundtruth"] = full_df["gold"].values.tolist()
        full_df["uid"] = full_df["!amt_worker_ids"].values[:,0]
        full_df["item"] = full_df["orig_id"].values[:,0]
        full_df = full_df[["item", "uid", "annotation", "groundtruth"]]

        super().setup(full_df, full_df[["item", "groundtruth"]], c_gold_label="groundtruth")


class DecompositionExperiment(experiments.RealExperiment):

    def __init__(self, eval_fn, label_colname, item_colname, uid_colname, distance_fn=None, **kwargs):
        super().__init__(eval_fn, label_colname, item_colname, uid_colname, distance_fn, **kwargs)
        # self.gran_exp = experiments.RealExperiment(self.eval_fn, self.label_colname, "newItemID", self.uid_colname)
        # self.granno_df_cluster = experiments.RealExperiment(self.eval_fn, self.label_colname, "newItemID", self.uid_colname)
        # self.gran_exp_orc = experiments.RealExperiment(self.eval_fn, self.label_colname, "newItemID", self.uid_colname)
        self.gran_experiments = {}
    
    def register_gran_exp(self, name, granno_df):
        gran_exp = experiments.RealExperiment(self.eval_fn, self.label_colname, "newItemID", self.uid_colname)
        gran_exp.setup(granno_df, merge_index="origItemID")
        self.gran_experiments[name] = gran_exp
        
    def setup(self, annodf, golddf, c_gold_item=None, c_gold_label=None, skip_gran=False):
        super().setup(annodf=annodf, golddf=golddf, c_gold_item=c_gold_item, c_gold_label=c_gold_label)
        if not skip_gran:
            self.register_gran_exp("intersect", granularity.fragment_by_overlaps(self))
            self.register_gran_exp("cluster", granularity.fragment_by_overlaps(self, decomp_fn=cluster_decomp))
            self.register_gran_exp("oracle", granularity.fragment_by_overlaps(self, use_oracle=True))

    def register_weighted_merge(self):
        if hasattr(self, "merge_fn"):
            for gran_experiment in self.gran_experiments.values():
                gran_experiment.merge_fn = self.merge_fn
            super().register_weighted_merge()

    def train(self, dem_iter, mas_iter):
        super().train(dem_iter=dem_iter, mas_iter=mas_iter)
        for gran_experiment in self.gran_experiments.values():
            gran_experiment.train(dem_iter=dem_iter, mas_iter=mas_iter)
    
    def test(self, debug):
        super().test(debug=debug)
        for name, gran_experiment in self.gran_experiments.items():
            gran_experiment.test_merged_granular(orig_golddict=self.golddict, debug=debug)
            gran_sb = {F"GRANULAR {name} {k}": v for k, v in gran_experiment.scoreboard.items()}
            gran_sb_scores = {F"GRANULAR {name} {k}": v for k, v in gran_experiment.scoreboard_scores.items()}
            self.scoreboard.update(gran_sb)
            self.scoreboard_scores.update(gran_sb_scores)

        # self.gran_exp.test_merged_granular(orig_golddict=self.golddict, debug=debug)
        # self.gran_exp_orc.test_merged_granular(orig_golddict=self.golddict, debug=debug)
        # gran_sb = {F"GRANULAR {k}": v for k, v in self.gran_exp.scoreboard.items()}
        # gran_orc_sb = {F"GRANULAR ORACLE {k}": v for k, v in self.gran_exp_orc.scoreboard.items()}
        # gran_sb_scores = {F"GRANULAR {k}": v for k, v in self.gran_exp.scoreboard_scores.items()}
        # gran_orc_sb_scores = {F"GRANULAR ORACLE {k}": v for k, v in self.gran_exp_orc.scoreboard_scores.items()}
        # self.scoreboard = {**self.scoreboard, **gran_sb, **gran_orc_sb}
        # self.scoreboard_scores = {**self.scoreboard_scores, **gran_sb_scores, **gran_orc_sb_scores}


class PICOExperiment(DecompositionExperiment):

    def __init__(self, **kwargs):
        super().__init__(lambda x,y: eval_f1(x, y, strict_range=False, strict_tag=False, str_spans=False),
                        "label", "itemID", "uid")
        self.rawdf = pd.read_json("data/PICO/PICO-annos-crowdsourcing.json", lines=True)
        self.aggdf = pd.read_json("data/PICO/PICO-annos-crowdsourcing-agg.json", lines=True)
        self.golddf = pd.read_json("data/PICO/PICO-annos-professional.json", lines=True)
        self.merge_fn = merge_functions.vectorrange_merge

    def setup(self):
        userIDs = []
        itemIDs = []
        labels = []
        golds = []
        hmmcrowds = []
        majorityvotes = []
        for row in self.rawdf.iterrows():
            itemID = row[1]["docid"]
            data = row[1]["Participants"]
            gold = self.golddf[self.golddf["docid"] == itemID]["Participants"].values[0]
            gold = gold.get("MedicalStudent")
            agg = self.aggdf[self.aggdf["docid"] == itemID]["Participants"].values[0]
            for userID, label in data.items():
                userIDs.append(userID)
                itemIDs.append(itemID)
                labels.append(label2tvr(label, default=[]))
                golds.append(label2tvr(gold))
                hmmcrowds.append(agg["HMMCrowd"])
                majorityvotes.append(agg["MajorityVote"])
        df = pd.DataFrame({"uid":userIDs, "itemID":itemIDs, "label":labels, "gold":golds,
                        "HMMCrowd":hmmcrowds, "MajorityVote":majorityvotes})
        df = df.sort_values("itemID")
        userIdDict = utils.make_categorical(df, "uid")
        itemIdDict = utils.make_categorical(df, "itemID")
        anno_df = df.copy()
        super().setup(anno_df, anno_df, c_gold_label="gold")
        mv_labels = {k:label2tvr(v) for k, v in dict(df.groupby("itemID").first()["MajorityVote"].dropna()).items()}
        hmm_labels = {k:label2tvr(v) for k, v in dict(df.groupby("itemID").first()["HMMCrowd"].dropna()).items()}
        self.register_baseline("Tokenwise MV", mv_labels)
        self.register_baseline("Crowd-HMM", hmm_labels)


class BBExperiment(DecompositionExperiment):

    def __init__(self, eval_fn=iou_score_multi, dist_fn=None, **kwargs):
        super().__init__(eval_fn=eval_fn, label_colname="annotation", item_colname="item", uid_colname="uid")
        self.merge_fn = merge_functions.vectorrange_merge
        # np.random.seed(42)
        with open('data/gt_canary_od_pretty_02042020.json') as f:
            self.dataset = json.load(f)
    
    def setup(self, **kwargs):
        cols = ["item", "uid", "annotation", "groundtruth"]
        NUM_ITEMS = kwargs.pop("n_items", None) or 200
        MAX_WORKERS_PER_ITEM = kwargs.pop("max_workers_per_item", None) or 100
        i = 0
        rows = []
        for image_key, all_data in self.dataset.items():
            i+=1
            if i > NUM_ITEMS:
                break
            raw_annotations_dict = all_data['worker_answers']
            gt = all_data['ground_truth']['annotations']
            keys =  list(raw_annotations_dict.keys())
            random.shuffle(keys)
            
            for worker_id in keys[:MAX_WORKERS_PER_ITEM]:
                anno = raw_annotations_dict[worker_id]
                stripped_anno = anno['answerContent']['boundingBox']['boundingBoxes']                                                      
                row = [image_key, worker_id, stripped_anno, gt]
                rows.append(row)
        df = pd.DataFrame(rows, columns=cols)
        df["annotation"] = df["annotation"].apply(convert2vr).dropna()
        df["groundtruth"] = df["groundtruth"].apply(convert2vr).dropna()
        super().setup(annodf=df, golddf=df, c_gold_item="item", c_gold_label="groundtruth", **kwargs)
    

class NERExperiment(DecompositionExperiment):
    def __init__(self, eval_fn=None, dist_fn=None, **kwargs):
        if eval_fn is None:
            eval_fn = lambda x,y: eval_f1(x, y, strict_range=False, strict_tag=False, str_spans=True)
        if dist_fn is None:
            dist_fn = lambda x,y: 1 - eval_f1(x, y, strict_range=False, strict_tag=False, str_spans=True)
        super().__init__(eval_fn, "annotation", "item", "worker", dist_fn)
        self.data_dir = "../seqcrowd-acl17/task1/val/mturk_train_data/"
        self.gold_dir = "../seqcrowd-acl17/task1/val/ground_truth/"

    def setup(self):

        def load(data_dir, isgold=False):
            workers = []
            items = []
            raw_labels = []
            worker_dirs = [""] if isgold else os.listdir(data_dir)
            for worker_dir in [d for d in worker_dirs if "." not in d]:
                item_files = [f for f in os.listdir(data_dir + worker_dir) if ".txt" in f]
                for item_file in item_files:
                    item_dir = data_dir + worker_dir + "/" + item_file
                    with open(item_dir) as openedfile:
                        raw_label = []
                        for line in openedfile:
                            token_labels = [t for t in line.replace("\n", "").split(" ")]
                            if len(token_labels) == 2:
                                raw_label.append(token_labels)
                        workers.append("gold" if isgold else worker_dir)
                        items.append(item_file.replace(".txt", ""))
                        raw_labels.append(raw_label)
            return pd.DataFrame({"worker":workers, "item":items, "raw_label":raw_labels})

        def clean(df):
            df["clean_label"] = [[seq for seq in label if len(seq[0]) > 0] for label in df["raw_label"]]
            return df
        
        def raw2ranges(raw_label):
            ''' represent annotations as sequences of token index ranges (with tags) '''
            result = []
            start_i = None
            curr_tag = 'O'
            for i, token_label in enumerate(raw_label):
                if token_label[1] != curr_tag and token_label[1][0] != "I": # new labeled span
                    newspan = {"range":[start_i, i], "tag":curr_tag}
                    if start_i is not None and curr_tag != "O":
                        newspan["tag"] = newspan["tag"][2:]
                        result.append(newspan)
                    start_i = i
                    curr_tag = token_label[1]
            return result

        def raw2NEs(raw_label):
            ''' represent annotations as named entity strings (with tags) '''
            result = []
            start_i = None
            curr_tag = 'O'
            for i, token_label in enumerate(raw_label):
                if token_label[1] != curr_tag and token_label[1][0] != "I": # new labeled span
                    newspan = {"range":[start_i, i], "tag":curr_tag}
                    if start_i is not None and curr_tag != "O":
                        start, end = tuple(newspan["range"])
                        newspan["range"] = (" ".join([x[0] for x in raw_label[start:end]])).lower()
                        newspan["tag"] = newspan["tag"][2:]
                        result.append(TaggedString(newspan["range"], newspan["tag"]))
                    start_i = i
                    curr_tag = token_label[1]
            return result

        anno_df = clean(load(self.data_dir).sort_values("item"))
        gold_df = clean(load(self.gold_dir, isgold=True))

        anno_df["annotation"] = anno_df["clean_label"].apply(raw2NEs)
        gold_df["annotation"] = gold_df["clean_label"].apply(raw2NEs)

        super().setup(anno_df, gold_df, c_gold_label="annotation")

class RationalesExperiment(experiments.CategoricalExperiment):
    def __init__(self, eval_fn=None, dist_fn=None, **kwargs):
        if eval_fn is None:
            eval_fn = lambda x,y: (1 if x["cat"] == y["cat"] else 0)
        super().__init__(eval_fn=eval_fn, distance_fn=dist_fn)
        self.data_dir = "data/webcrowd25k/crowd_judgements.csv"
        self.gold_dir = "data/webcrowd25k/gold_judgements.txt"
    def setup(self):
        allannodf = pd.read_csv(self.data_dir, sep=",", error_bad_lines=False, header=0,
                    names=["uid","_1","url","_2","relevance_label","_3","tid","_4","rationale","duration","item","_5"])
        allgolddf = pd.read_csv(self.gold_dir, sep=" ", error_bad_lines=False, header=0,
                    names=["tid", "_", "item", "relevance_gold"])
        TOPIC_ID = [259,267] # find ones where AGG underperforms RU
        annodf = allannodf[allannodf["tid"].isin(TOPIC_ID)].copy()
        golddf = allgolddf[allgolddf["tid"].isin(TOPIC_ID)].copy()
        annodf["annotation"] = [{"cat":cat, "rat":rat} for cat, rat in zip(annodf["relevance_label"], annodf["rationale"])]
        golddf["gold"] = [{"cat":cat, "rat":None} for cat in golddf["relevance_gold"]]
        super().setup(annodf, golddf, c_anno_label="annotation", c_gold_label="gold")

class SimRankingExperiment(experiments.RankerExperiment):
    def __init__(self, eval_fn=None, dist_fn=None, **kwargs):
        super().__init__(base_dir="data/qrels.all.txt")
        self.distance_fn = lambda x,y: 1 - self.eval_fn(x, y)
        self.merge_fn = merge_functions.numerical_mean = merge_functions.borda_count
    
    def setup(self):
        super().setup(n_items=100, n_users=20, pct_items=0.2, uerr_a=1, uerr_b=1, difficulty_a=1, difficulty_b=1, ngoldu=0)

def test_NER(debug=True):
    eval_fns = {}
    dist_fns = {}
    eval_fns["strict range strict tag"] = lambda x, y: eval_f1(x, y, strict_range=True, strict_tag=True, str_spans=True)
    eval_fns["strict range lenient tag"] = lambda x, y: eval_f1(x, y, strict_range=True, strict_tag=False, str_spans=True)
    eval_fns["lenient range strict tag"] = lambda x, y: eval_f1(x, y, strict_range=False, strict_tag=True, str_spans=True)
    eval_fns["lenient range lenient tag"] = lambda x, y: eval_f1(x, y, strict_range=False, strict_tag=False, str_spans=True)
    dist_fns["strict range strict tag"] = lambda x, y: 1 - eval_f1(x, y, strict_range=True, strict_tag=True, str_spans=True)
    dist_fns["strict range lenient tag"] = lambda x, y: 1 - eval_f1(x, y, strict_range=True, strict_tag=False, str_spans=True)
    dist_fns["lenient range strict tag"] = lambda x, y: 1 - eval_f1(x, y, strict_range=False, strict_tag=True, str_spans=True)
    dist_fns["lenient range lenient tag"] = lambda x, y: 1 - eval_f1(x, y, strict_range=False, strict_tag=False, str_spans=True)

    return test_experiment("NER", NERExperiment, eval_fns, dist_fns, debug=debug)


def test_experiment(experiment_name,
                    experiment_factory,
                    eval_fn_dict={"default":None},
                    dist_fn_dict={"default":None},
                    dem_iter=500,
                    mas_iter=500,
                    prune_ratio=0,
                    debug=False):
    results = []
    for eval_name, eval_fn in eval_fn_dict.items():
        for dist_name, dist_fn in dist_fn_dict.items():
            if debug:
                print("\n", eval_name, dist_name)
            inputs = {"eval_fn": eval_fn, "dist_fn": dist_fn}
            the_experiment = experiment_factory(**{k: v for k, v in inputs.items() if v is not None})
            the_experiment.setup()
            print(the_experiment.describe_data())
            the_experiment.train(dem_iter=dem_iter, mas_iter=mas_iter)
            the_experiment.register_weighted_merge()
            the_experiment.test(debug=False)
            for method_name, score in the_experiment.scoreboard.items():
                if debug:
                    print(method_name, score)
                ss = the_experiment.statistical_significance(method_name)
                results.append(ExperimentResult(experiment_name, eval_name, dist_name, method_name, None, score, ss))

    results_df = pd.concat([r.reset_index() for r in results])
    results_df.to_csv(F"results/{experiment_name}_results.csv")
    return results_df

if __name__ == '__main__':
    test_NER()