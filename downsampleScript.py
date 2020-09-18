import csv
import itertools

import numpy
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from pandas import DataFrame
import tsv
import experiments
import utils
from granularity import *
from sklearn.metrics import f1_score, accuracy_score

input_df = pd.read_csv("data/answer_weather_ordinal.csv", sep=",")
truth_df = pd.read_csv("data/truth_weather_ordinal.csv")

column_names=['worker','question','answer']
input_df=input_df.reindex(columns=column_names)

grouped_df = input_df.groupby('question')




minSamples=(numpy.min(input_df.groupby('question').size()))
print(minSamples)



for i in range(4):

    annotation_df = DataFrame()
    for q in grouped_df.groups:
        annotation_df = annotation_df.append(grouped_df.get_group(q).sample(int((minSamples)-(10*i))))
        print(int(minSamples)-(10*i))
        name='waterbird_label_'+str(int((minSamples)-(10*i)))+'.csv'
        annotation_df.to_csv(name,header=False,sep='\t',index=False)


annotation_df=DataFrame()
for q in grouped_df.groups:
    annotation_df = annotation_df.append(grouped_df.get_group(q).sample(1))
    name='waterbird_label_1.csv'
    annotation_df.to_csv(name,header=False,index=False,sep='\t')




















