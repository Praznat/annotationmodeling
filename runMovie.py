import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import experiments
import utils
from granularity import *


annotation_df=pd.read_csv("movie.csv")


# annotation_df=dfs
# annotation_df["annotation"]=annotation_df["answer"].values.tolist()
# annotation_df["groundtruth"]=annotation_df["truth"].values.tolist()
# annotation_df["uid"]=annotation_df["worker"].values.tolist()
# annotation_df["item"]=annotation_df["question"].values.tolist()
# annotation_df


print(annotation_df.head())











dist_fn = lambda x, y:abs(x-y)
eval_fn=lambda x,y:abs(1-dist_fn(x-y))


movie_exp=experiments.RealExperiment(eval_fn,"answer","truth","worker",distance_fn=dist_fn)
#movie_exp.setup(annotation_df,annotation_df[["question","truth"]],c_gold_label="truth")
movie_exp.setup(annodf=annotation_df,golddf=annotation_df[["question","truth"]],c_anno_uid="worker",c_anno_item="question",c_anno_label="answer",c_gold_item="question",c_gold_label="truth")
movie_exp.annodf
movie_exp.describe_data()
movie_exp.train()
movie_exp.train()
print(movie_exp.bau_preds)




