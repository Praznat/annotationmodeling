import pandas as pd
import numpy as np
import simulation
import utils

def evalb(parse1, parse2):
    from PYEVALB import scorer as pyscorer
    from PYEVALB import parser as pyparser
    pyparse1 = pyparser.create_from_bracket_string(str(parse1))
    pyparse2 = pyparser.create_from_bracket_string(str(parse2))
    try:
        score = pyscorer.Scorer().score_trees(pyparse1, pyparse2)
    except Exception as e:
        print("Exception!")
        print(e)
        print(pyparse1)
        print(pyparse2)
        return 0

    f1 = 2 * (score.recall * score.prec) / (score.recall + score.prec)
    return f1 * score.tag_accracy

class DecodableStr(str):
    def __init__(self, string):
        self.string = string
    def decode(self, arg):
        return self.string

class ParsableStr():
    def __init__(self, tokens, is_tokenized=True):
        if not is_tokenized:
            tokens = tokens.split(' ')
        self.tokens = [DecodableStr(s) for s in tokens]
    def __iter__(self):
        return iter(self.tokens)

def create_user_data(uid, df, pct_items, u_err, difficulty_dict=None, extraarg=None):
    n_sentences_parsed = int(np.round(pct_items * len(df)))
    rows_chosen = np.random.choice(np.arange(len(df)), n_sentences_parsed, replace=False)
    chosen_parses = []
    sentences_parsed = []
    sentenceIds = []
    parser = np.random.choice(extraarg)
    for row_i in rows_chosen:
        sentenceId, sentence, tokens, parse_scores, parses = df.iloc[row_i]
        sentences_parsed.append(sentence)
        sentenceIds.append(sentenceId)
        i_difficulty = difficulty_dict.get(sentenceId) if difficulty_dict else 0

        target_score = (1 - u_err) * (1 - i_difficulty)
        target_i = (np.abs(parse_scores - target_score)).argmin()
        chosen_parse = parses[target_i]

        chosen_parses.append(chosen_parse)
    dfdict = {
        "uid": [uid] * n_sentences_parsed,
        "sentence": sentences_parsed,
        "sentenceId": sentenceIds,
        "parse": chosen_parses
    }
    return pd.DataFrame(dfdict)

class ParserSimulator(simulation.Simulator):
    def __init__(self, bllipparser, tokenized_sentences, eval_fn=evalb):
        from nltk.parse import malt
        self.df = pd.DataFrame({"sentenceId":np.arange(len(tokenized_sentences)),
                                "sentence":[" ".join(s) for s in tokenized_sentences],
                                "tokens":tokenized_sentences})
        self.bllip = bllipparser
        self.parsers = [bllipparser]
        self.eval_fn = eval_fn
        # self.parsers.append(malt.MaltParser('maltparser-1.9.2', 'maltparser-1.9.2/engmalt.linear-1.7.mco'))
        parser = self.parsers[0]
        df_parse_scores = []
        df_parses = []
        for i, row in self.df.iterrows():
            sentenceId, sentence, tokens = row
            parses = list(parser.parse(ParsableStr(tokens, is_tokenized=True)))
            best_parse = parses[0]
            scores = []
            for parse in parses:
                scores.append(self.eval_fn(parse, best_parse))
            df_parse_scores.append(scores)
            df_parses.append(parses)
        self.df["parse_scores"] = df_parse_scores
        self.df["parses"] = df_parses

    def create_stan_data(self, n_users, pct_items, err_rates, difficulty_dict):
        self.err_rates = err_rates
        self.difficulty_dict = difficulty_dict
        self.sim_df = simulation.create_sim_df(create_user_data, self.df, n_users, pct_items,
                                                        err_rates, difficulty_dict, extraarg=self.parsers)
        stan_data = utils.calc_distances(self.sim_df, (lambda x,y: 1 - self.eval_fn(x, y)), label_colname="parse", item_colname="sentenceId")
        return stan_data
    
    def sim_uerr_fn(self, uerr_a, uerr_b, n_users):
        z = np.random.beta(uerr_a, uerr_b, 10000)
        return np.quantile(z, np.linspace(0,1,n_users+2)[1:-1])

    def sim_diff_fn(self, difficulty_a, difficulty_b):
        difficulty_dict, _ = simulation.create_item_param_dicts(self.df.sentenceId, difficulty_a, difficulty_b, 0, 0)
        return difficulty_dict
