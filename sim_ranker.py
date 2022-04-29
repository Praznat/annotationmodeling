import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import simulation
import utils

NOISE = 0.01
TOPK = 25

def intersectscore(r1, r2):
    s1 = set(r1)
    s2 = set(r2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def kendaltauscore(r1, r2, use_spearmanr=False, topK=None):
    if topK is None:
        topK = TOPK
    rfact = pd.factorize(list(set(r1[:topK]).union(set(r2[:topK]))))
    rfact = zip(rfact[1], rfact[0])
    rfact = dict(rfact)
    # print(r1[:5], r2[:5])
    # print(rfact)
    depth = len(rfact) + 50
    r1v = len(rfact) * np.ones(depth)
    r2v = len(rfact) * np.ones(depth)
    for i in range(min([topK, len(r1), len(r2)])):
        r1v[rfact[r1[i]]] = i
        r2v[rfact[r2[i]]] = i
        # print(r1[i], rfact[r1[i]], r2[i], rfact[r2[i]])
    r1v = r1v.astype(int)
    r2v = r2v.astype(int)
    # print(r1v[:5], r2v[:5], kendalltau(r1v, r2v).correlation)
    fn = kendalltau if not use_spearmanr else spearmanr
    return fn(r1v, r2v).correlation

def create_user_data(uid, df, pct_topics, u_err, difficulty_dict=None, extraarg=None):
    items = df.topic_item.unique()
    n_items_rated = int(np.round(pct_topics * len(items)))
    items_rated = sorted(np.random.choice(items, n_items_rated, replace=False))
    rankings = []
    for item in items_rated:
        idf = df[df.topic_item == item]
        idflen = len(idf)
        i_difficulty = difficulty_dict.get(item)
        err_mu = np.random.normal(0, u_err * i_difficulty, idflen)
        if difficulty_dict is not None:
            i_difficulty = difficulty_dict.get(item)
        ratings = idf.gold + err_mu#np.random.normal(err_mu, NOISE, idflen)
        top_docs = np.array(idf.doc)[np.argsort(-ratings)][:TOPK]
        rankings.append(top_docs)
    dfdict = {
        "uid": [uid] * len(items_rated),
        "topic_item": items_rated,
        "rankings": rankings,
    }
    return pd.DataFrame(dfdict)

class RankerSimulator(simulation.Simulator):
    def __init__(self, rawdata_dir, max_docs_per_item=50, gold_sterr=0.5, n_items=0):
        self.df = pd.read_csv(rawdata_dir, sep=" ", error_bad_lines=False,
                            names=["topic_item", "na", "doc", "gold"])
        self.df = self.df[self.df.na == 0]
        self.df = self.trunc_docs(max_docs_per_item)
        topics = self.df.topic_item.unique()
        n_items = max(n_items, len(topics))
        extradfs = []
        for i in range(n_items - len(topics)):
            topic = np.random.choice(topics)
            tdf = self.df[self.df.topic_item==topic]
            new_topic = topic * (10 + i)
            newdf = tdf.copy(deep=True)
            newdf.topic_item = new_topic
            newdf.gold = np.random.permutation(newdf.gold.values)
            extradfs.append(newdf)
        if len(extradfs):
            self.df = pd.concat([self.df] + extradfs)
        self.df.gold = self.df.gold + np.random.normal(0, gold_sterr, len(self.df.gold))
        self.topic_lookup = utils.make_categorical(self.df, "topic_item")
        self.doc_lookup = utils.make_categorical(self.df, "doc")
        def rank_docs(data):
            return data.sort_values("gold", ascending=False).doc.values
        self.gold = self.df.groupby("topic_item").apply(rank_docs)

    def trunc_docs(self, maxdocs):
        result = []
        for item in self.df["topic_item"].unique():
            idf = self.df[self.df["topic_item"] == item]
            udocs = idf["doc"].unique()
            chosendocs = np.random.choice(udocs, maxdocs, replace=False) if len(udocs) > maxdocs else udocs
            result.append(idf[idf["doc"].isin(chosendocs)])
        return pd.concat(result)

    def create_stan_data(self, n_users, pct_items, err_rates, difficulty_dict):
        self.err_rates = err_rates
        self.difficulty_dict = difficulty_dict
        self.sim_df = simulation.create_sim_df(create_user_data, self.df, n_users, pct_items, err_rates, difficulty_dict)
        stan_data = utils.calc_distances(self.sim_df, (lambda x,y: 1 - self.eval_fn(x, y)), label_colname="rankings", item_colname="topic_item")
        return stan_data


    def sim_uerr_fn(self, uerr_a, uerr_b, n_users):
        z = 10 * np.random.beta(uerr_a, uerr_b, 10000)
        result = np.quantile(z, np.linspace(0,1,n_users+2)[1:-1])
        # return np.random.bea(uerr_a, uerr_b, n_users)
        return result
    
    def sim_diff_fn(self, difficulty_a, difficulty_b):
        z = 1 * np.random.beta(difficulty_a, difficulty_b, 10000)
        n_items = len(self.df.topic_item.unique())
        return dict(zip(np.arange(n_items), np.quantile(z, np.linspace(0,1,n_items+2)[1:-1])))