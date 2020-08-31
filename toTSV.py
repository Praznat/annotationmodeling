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

column_names=['worker','question','answer']
truth_df=pd.read_csv('data/answer_weather_ordinal.csv')
truth_df=truth_df.reindex(columns=column_names)
truth_df.to_csv('answer_weather.txt',header=False,index=False,sep='\t')



















