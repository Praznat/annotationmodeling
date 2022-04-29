''' generate syntactic parse experiment data '''

import numpy as np
import pandas as pd
import sim_parser
import experiments

parser_singleton = experiments.ParserSingleton(num_items=100)

parsexp = experiments.ParserExperiment(parser_singleton)
parsexp.setup(num_items=100, n_users=8, pct_items=0.5)

with open("parser_annodf.pkl", 'wb') as file_stream:
    pickle.dump(parsexp.annodf, file_stream)
with open("parser_golddict.pkl", 'wb') as file_stream:
    pickle.dump(parsexp.golddict, file_stream)