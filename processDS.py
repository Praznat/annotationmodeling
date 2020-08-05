import csv
import pandas as pd
from sklearn import metrics
import numpy as np
import sys
import os
# import argparse

# parser = argparse.ArgumentParser(description='Run experiment on categorical task')
# parser.add_argument('task_name', type=str, nargs=1,
#                     help='task/dataset name')
# parser.add_argument('--num-folds', dest='log_dir', nargs=1, type=int,
#                     help='number of kfolds', metavar='NUM_FOLDS', default=[False])

task_name = sys.argv[1]
level = sys.argv[2]
fold = sys.argv[3]
try:
	noise_level = sys.argv[4]
except:
	noise_level = 'NA'

df = pd.read_csv('object-probabilities.txt', sep='\t').sort_values(['Object']).reset_index()
df = df[['Object', 'Correct_Category', 'DS_MaxLikelihood_Category', 'MV_MaxLikelihood_Category']]
df = df.dropna()

methods = ['DS', 'MV']
for method in methods:
	col_name = method + '_MaxLikelihood_Category'
	accuracy = metrics.accuracy_score(df['Correct_Category'], df[col_name])
	print(df[['Object', 'Correct_Category', col_name]])
	f1 = metrics.f1_score(df['Correct_Category'], df[col_name], average='weighted')

	score_col_name = 'score' + method
	df[score_col_name] = (df[col_name] == df['Correct_Category']).astype(int)

	print(method, accuracy, f1, np.mean(df[score_col_name]))

	conf_matrix = metrics.confusion_matrix(df['Correct_Category'], df[col_name])
	print(conf_matrix)

	if noise_level == 'NA':
		print("NOISE LEVEL: ORIGINAL")
		filename = f'results_supervised_{task_name}.csv'
		with open (filename, 'a') as file:
			writer = csv.writer(file)
			if (os.stat(filename).st_size == 0):
				writer.writerow(['task','level','fold','accuracy','f1'])
			writer.writerow([task_name, level, fold, accuracy, f1])
	else:
		filename = f'results_noise_supervised_{task_name}.csv'
		with open (filename, 'a') as file:
			writer = csv.writer(file)
			if (os.stat(filename).st_size == 0):
				writer.writerow(['task','level','fold','noise_level','accuracy','f1'])
			writer.writerow([task_name, level, fold, noise_level, accuracy, f1])

