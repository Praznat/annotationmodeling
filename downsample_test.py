import itertools

import numpy
import numpy as np
import pandas
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from pandas import DataFrame

import experiments
import utils
from granularity import *
from sklearn.metrics import f1_score

input_df = pd.read_csv("downsample.csv", sep=",")
input_df.groupby('question')

annotation_df = pd.read_csv("downsample.csv")

grouped_df = annotation_df.groupby('question')
new_df = DataFrame()

sub_df=DataFrame()

print(grouped_df.groups)
for g in grouped_df.groups:
    sub_df=(grouped_df.get_group(g)).sample(2)
    print(sub_df)
    new_df=new_df.append(sub_df)

print(numpy.min(new_df.groupby('question').size()))






