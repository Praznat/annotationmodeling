import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel as ttest
from sim_normvec import VectorSimulator, euclidist
from sim_parser import ParserSimulator, ParsableStr, evalb
from sim_ranker import RankerSimulator, kendaltauscore
from simulation import BetaDist
import utils

def params_model(sim, opt, stan_data, skip_gold=False):
    ''' visually compare MAS parameters against known simulator parameters '''
    start = stan_data["n_gold_users"] if skip_gold else 0
    scatter_corr(opt["uerr"][start:], sim.err_rates[start:], title="User baseline")
    scatter_corr(opt.get("diff"), list(sim.difficulty_dict.values()), title="Item baseline")

def user_avg_dist(stan_data):
    ''' BAU scores for each user = user's average distance across whole dataset '''
    sddf = pd.DataFrame(stan_data)
    s1 = sddf.groupby("u1s").sum()["distances"]
    s2 = sddf.groupby("u2s").sum()["distances"]
    n1 = sddf.groupby("u1s").count()["distances"]
    n2 = sddf.groupby("u2s").count()["distances"]
    avg_distances = s1.add(s2, fill_value=0) / n1.add(n2, fill_value=0)
    return avg_distances

def item_avg_dist(stan_data):
    ''' items' average distance '''
    sddf = pd.DataFrame(stan_data)
    avg_distances = sddf.groupby("items").mean()["distances"]
    return avg_distances

def scatter_corr(pred_vals, true_vals, jitter=False, title=None, log=False):
    ''' nice scatter plot '''
    if len(pred_vals) == 0:
        return
    if len(pred_vals) < len(true_vals):
        true_vals = np.array(true_vals)[pred_vals.index - 1]
    noise = lambda: np.random.uniform(-.5, .5, len(pred_vals)) if jitter else 0
    if len(pred_vals) < 1000:
        plt.scatter(np.array(pred_vals) + noise(), np.array(true_vals) + noise())
    else:
        plt.scatter(np.array(pred_vals) + noise(), np.array(true_vals) + noise(), s=1)
    if title is not None:
        plt.title(title) 
    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.show()
    print(np.corrcoef(pred_vals, true_vals))

def diagnostics(opt, stan_data):
    ''' print useful scatterplots for diagnosing MAS trained model results '''
    plt.rcParams.update({'font.size': 20})

    plt.scatter(opt["pred_distances"], stan_data["distances"])
    plt.xlabel("pred_distances")
    plt.ylabel("distances")
    plt.show()
    uerr_b = user_avg_dist(stan_data)

    plt.scatter(opt["uerr"], uerr_b)
    plt.xlabel("inferred $\gamma$")
    plt.ylabel("user average distance")
    plt.ylim()
    plt.show()

    diff = item_avg_dist(stan_data)
    diffkeys = diff.index.values - 1
    plt.scatter(opt["diff"][diffkeys], diff)
    plt.xlabel("inferred diff")
    plt.ylabel("avg item distance")
    plt.show()

def pred_item(df, pred_uerr, label_colname, item_colname, uid_colname="uid"):
    ''' pick annotation per item according to lowest predicted user error '''
    df["pred_uerr"] = pred_uerr[df[uid_colname].values]
    def pickbybestuser(data):
        best_i = np.argmin(data["pred_uerr"].values)
        return data[label_colname].values[best_i]
    return df.groupby(item_colname).apply(pickbybestuser)

def get_model_user_rankings(opt, debug=False):
    ''' MAS model's user annotation ranking by item '''
    errs = opt["dist_from_truth"]
    result = np.argsort(errs, axis=1)
    tmp = errs[0][result[0]]
    assert tmp[0] <= tmp[1]
    if debug:
        plt.figure(figsize=(8,4))
        errs[errs==666] = np.max(errs[errs<666]) * 1.1 # reset high-error value for better viz
        plt.imshow(errs.T, cmap='coolwarm', interpolation='nearest')
        plt.xlabel("items")
        plt.ylabel("users")
        plt.show()
    return result

def get_baseline_random(annodf, label_colname, item_colname):
    ''' random annotation per item '''
    def pickrandomlabel(data):
        return data.sample(1)[label_colname].values[0]
    return dict(annodf.groupby(item_colname).apply(pickrandomlabel))

def get_baseline_global_best_user(stan_data, annodf, label_colname, item_colname, uid_colname="uid"):
    ''' BAU scores per item: annotation by user with smallest average global distance '''
    uerrs = user_avg_dist(stan_data).values
    return dict(pred_item(annodf, uerrs, label_colname, item_colname, uid_colname))

def get_baseline_item_centrallest(stan_data, annodf, label_colname, item_colname, uid_colname="uid",
                                    agg_fn=None):
    ''' SAD scores per item: annotation for item with smallest distance to other annotations of that item '''
    sddf = pd.DataFrame(stan_data)
    preds = {}
    for i0 in sorted(annodf[item_colname].unique()):
        i = i0 + 1
        iddf = sddf[sddf["items"] == i]
        if iddf.empty:
            idf = annodf[annodf[item_colname] == i0]
            pred = idf[label_colname].values[0]
        else:
            uerrs = user_avg_dist(iddf)
            i_annodf = annodf[annodf[item_colname] == i-1]
            if agg_fn is None:
                best_user = uerrs.idxmin()
                pred = i_annodf[i_annodf[uid_colname] == best_user-1][label_colname].values[0]
            else:
                pred = agg_fn(i_annodf, uerrs)
        preds[i-1] = pred
    return preds

def get_oracle_preds(stan_data, annodf, label_colname, item_colname, uid_colname="uid",
                    eval_fn=None, gold_dict={}):
    ''' choose annotation closest to gold for each item according to evaluation function '''
    if eval_fn is None:
        raise ValueError("Need a evaluation function to compute oracle")
    def agg_fn(i_annodf, uerrs):
        item = i_annodf[item_colname].values[0]
        gold = gold_dict.get(item)
        if gold is None:
            return i_annodf[label_colname].values[0]
        evals = [eval_fn(gold, label) for label in i_annodf[label_colname].values]
        pred = i_annodf[label_colname].values[np.argmax(evals)]
        return pred
    return get_baseline_item_centrallest(stan_data, annodf, label_colname, item_colname, uid_colname, agg_fn)

def get_preds(annodf, per_item_user_rankings, label_colname, item_colname, user_colname="uid"):
    ''' MAS scores per item: annotation per item according to ranked annotations per item '''
    preds = {}
    for i, item in enumerate(sorted(annodf[item_colname].unique())):
        idf = annodf[annodf[item_colname] == item]
        pred = None
        for best_user in per_item_user_rankings[i]:
            uidf = idf[idf[user_colname]==best_user]
            if len(uidf) > 0:
                pred = uidf[label_colname].values[0]
                break
        preds[item] = pred
    return preds

def eval_scores_vs(baseline_scores, model_scores, badness_threshold):
    ''' print and display compar\ison two sets of scores against each other '''
    diffs = np.array(model_scores) - np.array(baseline_scores)
    print(np.mean(baseline_scores), np.mean(model_scores))
    print("t-test", ttest(baseline_scores, model_scores))
    print("z-score", np.mean(diffs) / np.std(diffs))
    maxx = np.max(np.abs(diffs))
    print("baseline below thresh", (np.array(baseline_scores) < badness_threshold).mean())
    print("model below thresh", (np.array(model_scores) < badness_threshold).mean())
    # plt.hist(diffs, bins=np.linspace(-maxx, maxx, 10))
    # plt.show()

def eval_preds(preds, golds, eval_fn):
    ''' evaluate chosen annotations per item against known gold according to evaluation function '''
    scores = []
    for i, gold in golds.items():
        score = eval_fn(gold, preds[i]) if preds.get(i) is not None else 0
        scores.append(score)
    return scores

def eval_preds_vs(baseline_preds, model_preds, golds, eval_fn, print_diffs=False, badness_threshold=0):
    ''' score two sets of chosen annotations against known gold according to evaluation function '''
    baseline_scores = eval_preds(baseline_preds, golds, eval_fn)
    model_scores = eval_preds(model_preds, golds, eval_fn)
    eval_scores_vs(baseline_scores, model_scores, badness_threshold)
    return baseline_scores, model_scores

class Experiment():
    ''' Contains raw annotations, processed data, model parameters, predictions, and known gold '''

    def __init__(self, label_colname, item_colname, uid_colname="uid"):
        self.label_colname = label_colname
        self.item_colname = item_colname
        self.uid_colname = uid_colname
        self.stan_data = None
        self.opt = None
        self.simulator = None
        self.annodf = None
        self.badness_threshold = 0
        self.extra_baseline_labels = {}

    def produce_stan_data(self):
        ''' use distance function to create distance matrices and other data in its final form before training '''
        self.stan_data = utils.calc_distances(self.annodf, self.distance_fn, label_colname=self.label_colname, item_colname=self.item_colname, uid_colname=self.uid_colname)

    def train(self, use_diff=1, dim_size=8, iter=500, **kwargs):
        ''' trains and predicts using MAS, BAU, and SAD methods '''
        if self.stan_data is None:
            raise ValueError("Must setup stan_data first")
        stan_model = utils.stanmodel("mas", overwrite=False)
        self.stan_data["use_diff"] = use_diff
        self.stan_data["DIM_SIZE"] = dim_size
        self.stan_data["eps_limit"] = 3
        self.stan_data["uerr_prior_scale"] = 1
        self.stan_data["diff_prior_scale"] = 1
        self.stan_data["uerr_prior_loc_scale"] = 8
        self.stan_data["diff_prior_loc_scale"] = 8
        self.stan_data["err_scale"] = 0.1
        uerr_b = user_avg_dist(self.stan_data).values
        init = {
            "uerr_Z": uerr_b - np.mean(uerr_b),
        }

        self.stan_data = {**self.stan_data, **kwargs}
        self.opt = stan_model.optimizing(data=self.stan_data, init=init, verbose=True, iter=iter)
        # print("sigma", self.opt["sigma"])

        per_item_user_rankings = get_model_user_rankings(self.opt, debug=False)
        self.bau_preds = get_baseline_global_best_user(self.stan_data, self.annodf, self.label_colname, self.item_colname, self.uid_colname)
        self.sad_preds = get_baseline_item_centrallest(self.stan_data, self.annodf, self.label_colname, self.item_colname, self.uid_colname)
        self.mas_preds = get_preds(self.annodf, per_item_user_rankings, self.label_colname, self.item_colname, self.uid_colname)

    def eval_model(self, random_scores, model_preds, modelname, num_samples):
        ''' display comparison of model predictions vs baseline '''
        print(modelname)
        model_scores = eval_preds(model_preds, self.golddict, self.eval_fn)
        model_scores *= num_samples
        eval_scores_vs(random_scores, model_scores, self.badness_threshold)
    
    def register_baseline(self, name, label_dict):
        ''' add additional baseline predictions if available '''
        self.extra_baseline_labels[name] = label_dict

    def test(self, num_samples=5, debug=False, **kwargs):
        ''' use known gold to test trained models and compare against each other and baselines '''
        # if self.simulator is not None:
        #     self.annodf = self.simulator.sim_df
        if self.annodf is None:
            raise ValueError("Must set annodf or create simulator")
        if self.opt is None:
            raise ValueError("Must train model first")
        random_scores = []
        for i in range(num_samples):
            random_preds = get_baseline_random(self.annodf, self.label_colname, self.item_colname)
            random_scores += eval_preds(random_preds, self.golddict, self.eval_fn)
        self.oracle_preds = get_oracle_preds(self.stan_data, self.annodf, self.label_colname, self.item_colname, self.uid_colname, self.eval_fn, self.golddict)
        self.eval_model(random_scores, self.bau_preds, "BEST AVAILABLE USER", num_samples)
        self.eval_model(random_scores, self.sad_preds, "SMALLEST AVERAGE DISTANCE", num_samples)
        self.eval_model(random_scores, self.mas_preds, "MULTIDIMENSIONAL ANNOTATION SCALING", num_samples)
        if hasattr(self, "extra_baseline_labels"):
            for baseline_name, baseline_preds in self.extra_baseline_labels.items():
                self.eval_model(random_scores, baseline_preds, baseline_name, num_samples)
        self.eval_model(random_scores, self.oracle_preds, "ORACLE", num_samples)
        if debug:
            get_model_user_rankings(self.opt, debug=True)
            if kwargs.get("diagnose_vs_sim") and self.simulator is not None:
                params_model(self.simulator, self.opt, self.stan_data)
            else:
                diagnostics(self.opt, self.stan_data)

    def debug(self, plot_stress=False, plot_vs_sad=False, plot_vs_gold=False, skip_miniplots=False):
        ''' tool for diving into results '''
        from sklearn.decomposition import PCA
        def userset(data):
            result = set(data["u1s"]).union(set(data["u2s"]))
            return result
        sddf = pd.DataFrame(self.stan_data)
        item_userset = sddf.groupby("items").apply(userset)
        bau = user_avg_dist(self.stan_data)
        map_corrs = []
        map_scores_all = []
        sad_corrs = []
        sad_scores_all = []
        bau_corrs = []
        bau_scores_all = []
        gold_scores_all = []
        for i, iue in enumerate(self.opt["item_user_errors"]):
            gold = self.golddict.get(i)
            if gold is None:
                continue
            users = item_userset.get(i+1)
            if users is None or len(users) < 2:
                continue
            dist_from_truth = self.opt["dist_from_truth"][i]
            iu_distances = sddf[sddf["items"]==i+1][["u1s", "u2s", "distances"]]
            if not skip_miniplots:
                print("item", str(i+1))
                print(iu_distances)
            sad = user_avg_dist(iu_distances)
            sad_scores = [sad.loc[u] for u in users]
            bau_scores = [bau.loc[u] for u in users]
            # if len(iue[0]) > 2:
            #     all_embeddings = PCA(n_components=2).fit_transform(iue)
            # embeddings = np.array([all_embeddings[u-1] for u in users])
            embeddings = np.array([iue[u-1] for u in users])
            if len(embeddings[0]) > 2:
                embeddings = PCA(n_components=2).fit_transform(embeddings)
            map_scores = [dist_from_truth[u-1] for u in users]
            skills = [self.opt["uerr"][u-1] for u in users]
            scale = np.max(np.abs(embeddings)) * 1.05
            
            # plot preds
            idf = self.annodf[self.annodf[self.item_colname]==i]

            if plot_vs_gold:
                labels = [idf[idf[self.uid_colname] == u-1][self.label_colname].values[0] for u in users]
                
                if gold is not None:
                    gold_scores = [self.eval_fn(gold, label) for label in labels]
                else:
                    continue
                    gold_scores = np.nan * np.zeros(len(labels))
                if not skip_miniplots:
                    diff = self.opt["diff"][i]
                    plt.scatter(map_scores, gold_scores)
                    plt.scatter(sad_scores, gold_scores, color="red")
                    plt.title(diff)
                    plt.show()
                try:
                    map_scores_all += map_scores
                    sad_scores_all += sad_scores
                    bau_scores_all += bau_scores
                    gold_scores_all += gold_scores
                except:
                    pass

            # plot stress
            if plot_stress:
                for _, row in iu_distances.iterrows():
                    u1, u2 = (int(row["u1s"]-1), int(row["u2s"]-1))
                    u1emb = embeddings[np.where(u1+1==np.array(list(users)))]
                    u2emb = embeddings[np.where(u2+1==np.array(list(users)))]
                    embs = np.concatenate((u1emb, u2emb)).T
                    stress = row["distances"] - np.linalg.norm(iue[u1] - iue[u2])
                    cmap = plt.cm.Reds if stress > 0 else plt.cm.Greens
                    plt.plot(embs[0], embs[1], color=cmap(10*stress))

            if not skip_miniplots:
                # plt.scatter(embeddings[:,0], embeddings[:,1])
                for ui, emb in enumerate(embeddings):
                    plt.plot([0,emb[0]], [0,emb[1]], "b-")
                    if plot_vs_gold:
                        pltanno = str(np.round(gold_scores[ui],2))
                    else:
                        pltanno = str(list(users)[ui]) + ":" + str(np.round(map_scores[ui],2)) + ":" + str(np.round(skills[ui],2))
                    plt.annotate(pltanno, emb)

                def plot_pred(preds, marker, color, size=50):
                    this_pred = preds.get(i)
                    this_uid = idf[[np.array_equal(v, this_pred) for v in idf[self.label_colname]]][self.uid_colname].values[0]
                    this_ui = np.where(np.array([u-1 for u in users]) == this_uid)[0][0]
                    this_emb = embeddings[this_ui]
                    plt.scatter([this_emb[0]], [this_emb[1]], marker=marker, c=color, s=size)
                plot_pred(self.oracle_preds, "o", "gold", 100)
                plot_pred(self.mas_preds, "d", "red")
                plot_pred(self.bau_preds, "+", "black")
                plot_pred(self.sad_preds, "x", "black")

                plt.xlim(-scale, scale)
                plt.ylim(-scale, scale)
                plt.show()

            if plot_vs_sad:
                plt.scatter(map_scores, sad_scores)
                plt.show()
                print("SAD vs MAS: ", self.sad_preds.get(i), self.mas_preds.get(i))
        
        if plot_vs_gold:

            plt.scatter(map_scores_all, gold_scores_all)
            plt.scatter(sad_scores_all, gold_scores_all, color="r")
            plt.scatter(bau_scores_all, gold_scores_all, color="y")
            plt.legend(["mas", "sad", "bau"])
            plt.show()
            print("\n ALL")
            print("ru", 0, np.std(gold_scores_all))
            print("bau", np.corrcoef(bau_scores_all, gold_scores_all)[0,1], np.std((1 - np.array(bau_scores_all)) - np.array(gold_scores_all)))
            print("sad", np.corrcoef(sad_scores_all, gold_scores_all)[0,1], np.std((1 - np.array(sad_scores_all)) - np.array(gold_scores_all)))
            print("mas", np.corrcoef(map_scores_all, gold_scores_all)[0,1], np.std((1 - np.array(map_scores_all)) - np.array(gold_scores_all)))

            fig, ax = plt.subplots()
            ax.scatter(1.1-np.array(map_scores_all), gold_scores_all)
            fs = 24
            # fig.suptitle('Score-all evaluation example', fontsize=fs)
            ax.set_xlabel("1 - MAS $\\varepsilon$ scores", fontsize=fs)
            ax.set_ylabel("Gold scores", fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            plt.show()


    def describe_data(self):
        ''' describes data, but must be called after producing stan_data '''
        nusers = self.stan_data["NUSERS"]
        nitems = self.stan_data["NITEMS"]
        nlabels = len(self.annodf)
        ulabels = self.annodf.groupby(self.uid_colname).count()[self.label_colname].values
        ilabels = self.annodf.groupby(self.item_colname).count()[self.label_colname].values
        lperu = str(np.mean(ulabels).round(2)) + "$\pm$" + str(2 * np.std(ulabels).round(2))
        lperi = str(np.mean(ilabels).round(2)) + "$\pm$" + str(2 * np.std(ilabels).round(2))

        self.annodf["labelstr"] = self.annodf[self.label_colname].astype(str)
        label_occurrences = self.annodf.groupby([self.item_colname, "labelstr"]).count()[self.label_colname].values
        dupes = np.sum(label_occurrences > 1)

        cols = [nusers, nitems, nlabels, lperu, lperi, dupes]
        print(" & ".join(map(str, cols)))

    def remove_supervised(self, ngoldu):
        ''' ONLY FOR SIMULATOR EXPERIMENTS: remove semi-supervised items from test set '''
        if ngoldu > 0:
            goldi = self.annodf[self.annodf[self.uid_colname] < ngoldu][self.item_colname].unique()
            for i in goldi:
                del self.golddict[i]
    
def run_square(annodf, uid_colname, item_colname, label_colname, squaredir="square-2.0/"):
    ''' put data into SQUARE format for running things like ZenCrowd '''
    outdir = squaredir + "data/test/"
    cats = pd.Categorical(map(str, annodf[label_colname])).codes
    cat_lookup = dict(zip(cats, annodf[label_colname]))
    annodf["cat"] = cats
    catdf = pd.DataFrame(np.unique(cats))
    catdf.iloc[:-1].to_csv(outdir + "categories.txt", header=False, index=False, sep=" ")
    catdf.iloc[-1:].to_csv(outdir + "categories.txt", header=False, index=False, sep=" ", mode='a', line_terminator="")
    respdf = annodf[[uid_colname, item_colname, "cat"]]
    respdf.iloc[:-1].to_csv(outdir + "responses.txt", header=False, index=False, sep=" ")
    respdf.iloc[-1:].to_csv(outdir + "responses.txt", header=False, index=False, sep=" ", mode='a', line_terminator="")
        # --responses ./data/test/responses.txt --category ./data/test/categories.txt --method Zen --estimation unsupervised --saveDir ./inferredGold/test/

class ParserExperiment(Experiment):
    ''' experiment using simulated parse data '''
    def __init__(self):
        super().__init__("parse", "sentenceId")
        from nltk.data import find as nltkfind
        from nltk.parse.bllip import BllipParser
        bllip_dir = nltkfind('models/bllip_wsj_no_aux').path
        self.BLLIP = BllipParser.from_unified_model_dir(bllip_dir)
        self.eval_fn = evalb
        self.badness_threshold = 0.9

    def setup(self, num_items, n_users, pct_items, uerr_a=1, uerr_b=4, difficulty_a=1, difficulty_b=100,
                    ngoldu=0, min_sentence_length=10):
        from nltk.corpus import brown as browncorpus
        sentences = np.random.choice(browncorpus.sents(), num_items * 3, replace=False)
        sentences = [s for s in sentences if len(s) > min_sentence_length][:num_items]
        self.simulator = ParserSimulator(self.BLLIP, sentences)
        self.stan_data = self.simulator.create_stan_data_scenario(n_users=n_users, pct_items=pct_items,
                                                    uerr_a=uerr_a, uerr_b=uerr_b,
                                                    difficulty_a=difficulty_a, difficulty_b=difficulty_b,
                                                    n_gold_users=ngoldu)
        gold_parses = [self.BLLIP.parse_one(ParsableStr(s)) for s in self.simulator.df.tokens.values]
        self.golddict = dict(enumerate(gold_parses))
        self.annodf = self.simulator.sim_df
        self.remove_supervised(ngoldu)

class RankerExperiment(Experiment):
    ''' experiment using simulated rankings data '''
    def __init__(self, base_dir):
        super().__init__("rankings", "topic_item")
        self.base_dir = base_dir
        self.eval_fn = kendaltauscore
    def setup(self, n_items, n_users, pct_items, uerr_a, uerr_b, difficulty_a, difficulty_b, ngoldu=0):
        self.simulator = RankerSimulator(self.base_dir, n_items=n_items)
        self.stan_data = self.simulator.create_stan_data_scenario(n_users=n_users, pct_items=pct_items,
                                                    uerr_a=uerr_a, uerr_b=uerr_b,
                                                    difficulty_a=difficulty_a, difficulty_b=difficulty_b,
                                                    n_gold_users=ngoldu)
        self.golddict = self.simulator.gold.to_dict()
        self.annodf = self.simulator.sim_df
        self.remove_supervised(ngoldu)
    def setup_standard(self):
        self.setup(n_items=100, n_users=20, pct_items=0.2, uerr_a=-1.0, uerr_b=0.8, difficulty_a=-2.0, difficulty_b=1.3, ngoldu=0)

class VectorExperiment(Experiment):
    ''' TODO experiment using simulated vector data '''
    def __init__(self):
        super().__init__("label", "topic_item")
        self.eval_fn = lambda x, y: -euclidist(x, y)
    def setup(self, n_items, n_users, pct_items, uerr_a, uerr_b, difficulty_a, difficulty_b, ngoldu=0, n_dims=8):
        self.simulator = VectorSimulator(n_items, n_dims)
        self.stan_data = self.simulator.create_stan_data_scenario(n_users=n_users, pct_items=pct_items,
                                                    uerr_a=uerr_a, uerr_b=uerr_b, n_gold_users=ngoldu,
                                                    difficulty_a=difficulty_a, difficulty_b=difficulty_b)
        self.annodf = self.simulator.sim_df
        self.golddict = self.simulator.df.gold.to_dict()
def setup_vector_standard(vector_experiment):
    vector_experiment.setup(n_items=100, n_users=50, pct_items=0.2, uerr_a=-1.0, uerr_b=0.8,
                            difficulty_a=-2.0, difficulty_b=.3, ngoldu=0)

class RealExperiment(Experiment):
    ''' experiment using real data provided by requester '''
    def __init__(self, eval_fn=None, label_colname="label", item_colname="item", uid_colname="uid", distance_fn=None):
        super().__init__(label_colname, item_colname, uid_colname)
        self.eval_fn = eval_fn
        self.distance_fn = distance_fn if distance_fn is not None else (lambda x,y: 1 - self.eval_fn(x, y))
    def produce_stan_data(self):
        self.stan_data = utils.calc_distances(self.annodf, self.distance_fn, label_colname=self.label_colname, item_colname=self.item_colname, uid_colname=self.uid_colname)
    def setup(self, annodf, golddf=None, c_anno_uid=None, c_anno_item=None, c_anno_label=None, c_gold_item=None, c_gold_label=None):
        renamey = lambda y: self.label_colname if "label" in y else self.item_colname if "item" in y else self.uid_colname if "uid" in y else "_"
        localargs = locals()
        colrename = {localargs[k]:renamey(k) for k in localargs if "c_" in k and localargs[k] is not None}
        self.annodf = annodf[[c_anno_uid or self.uid_colname, c_anno_item or self.item_colname, c_anno_label or self.label_colname]]
        self.annodf = self.annodf.dropna().copy().rename(columns=colrename)[[self.uid_colname, self.item_colname, self.label_colname]]
        uiddict = utils.make_categorical(self.annodf, self.uid_colname)
        itemdict = utils.make_categorical(self.annodf, self.item_colname)
        if golddf is not None:
            golddf = golddf[[c_gold_item or self.item_colname, c_gold_label or self.label_colname]]
            golddf = golddf.rename(columns=colrename)[[self.item_colname, self.label_colname]]
            golddf = utils.translate_categorical(golddf, self.item_colname, itemdict)
            self.golddict = golddf.set_index(self.item_colname).to_dict()[self.label_colname]
        self.produce_stan_data()

class CategoricalExperiment(RealExperiment):
    ''' TODO experiment using real simple data '''
    def __init__(self):
        super().__init__(distance_fn=lambda x, y: (1 if x == y else 0))


# semi-supervised learning

def set_supervised_items(expermnt, n_supervised_items, apply_fn=lambda x:x, randomize=False):
    ''' when there is known gold available for semi-supervised learning,
    call this after calling setup but before training to remove semi-supervised items from training and test set '''
    most_annotated_items = expermnt.annodf.groupby(expermnt.item_colname).count()[expermnt.label_colname].sort_values(ascending=False)
    expermnt.supervised_items = most_annotated_items.index.values[:n_supervised_items]
    if randomize:
        np.random.shuffle(expermnt.supervised_items)
    expermnt.supervised_labels = [apply_fn(expermnt.golddict.get(item)) for item in expermnt.supervised_items]
    assert expermnt.golddict is not None
    expermnt.golddict = expermnt.golddict.copy()
    for item in expermnt.supervised_items:
        try:
            del expermnt.golddict[item]
        except:
            pass

def make_supervised_standata(expermnt, model_gold_err=-4):
    ''' adds semi-supervised items back to training set (not test set) and tells MAS they are known gold '''
    # if n_supervised_items == 0:
    #     expermnt.produce_stan_data()
    #     return
    assert expermnt.golddict is not None
    
    supervised_df = pd.DataFrame({
        expermnt.uid_colname:np.zeros(len(expermnt.supervised_items), dtype=int),
        expermnt.item_colname:expermnt.supervised_items,
        expermnt.label_colname:expermnt.supervised_labels
        })
    expermnt.annodf = expermnt.annodf.copy()
    expermnt.annodf[expermnt.uid_colname] += 1
    expermnt.annodf = pd.concat([supervised_df, expermnt.annodf], sort=True).sort_values(expermnt.uid_colname)
    expermnt.produce_stan_data()
    expermnt.stan_data["n_gold_users"] = 1
    expermnt.stan_data["gold_user_err"] = model_gold_err