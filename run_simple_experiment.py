import sys
import csv
import os
import numpy as np
import pandas as pd
from eval_functions import mse, mae, accuracy, f1_weighted, f1_macro, score_predictions
from matplotlib import pyplot as plt
import experiments
from experiment_manager import *
import warnings
import glob
import argparse
import merge_functions

def get_predictions_dict(exp, run_merge):
    predictions = {
        'MAS': exp.mas_preds,
        'BAU': exp.bau_preds,
        'SAD': exp.sad_preds,
        'ORACLE': exp.oracle_preds,
        'DEM' : exp.dem_preds,
        'RANDOM': experiments.get_baseline_random(exp.annodf, exp.label_colname, exp.item_colname),
    }

    if run_merge:
        predictions['MAS_MERGE'] = exp.extra_baseline_labels['MAS Merge']
        predictions['BAU_MERGE'] = exp.extra_baseline_labels['BAU Merge']
        predictions['SAD_MERGE'] = exp.extra_baseline_labels['SAD Merge']
        predictions['DEM_MERGE'] = exp.extra_baseline_labels['DEM Merge']
        predictions['ORACLE_MERGE'] = exp.extra_baseline_labels['Oracle Merge']
        predictions['UNIFORM_MERGE'] = exp.extra_baseline_labels['Uniform Merge']

    return predictions

def process_predictions(predictions_dict, 
                        gold_dict,  
                        experiment_name, 
                        dist_name, 
                        eval_name,
                        metrics_fns,
                        score_fn):
    results = []
    scores_dict = {}
    # score_fn_exact_match = lambda x, y: 1 if x == y else 0

    for method_name, preds in predictions_dict.items():
        results_dict = {}
        results_dict['task'] = experiment_name
        results_dict['dist_fn'] = dist_name
        results_dict['eval_fn'] = eval_name
        for metric_name, metric_fn in metrics_fns.items():
            metric_val = metric_fn(gold_dict, preds)
            results_dict['method'] = method_name
            results_dict[metric_name] = metric_val
        results.append(results_dict)

        scores = score_predictions(gold_dict, preds, score_fn)
        scores_dict[method_name] = scores

    return (results, scores_dict)

def test_simple_experiment(experiment_name,
                    experiment_factory,
                    task_type,
                    full_df,
                    eval_fn_dict={"default":None},
                    dist_fn_dict={"default":None},
                    merge_fn=None,
                    dem_iter=500,
                    mas_iter=500,
                    run_merge=False,
                    supervised_items=None,
                    debug=False):
    
    results = []
    scores_dict_outer = {}
    for eval_name, eval_fn in eval_fn_dict.items():
        for dist_name, dist_fn in dist_fn_dict.items():

            exp = experiment_factory(eval_fn=eval_fn, dist_fn=dist_fn)
            exp.setup(full_df, full_df[['question', 'truth']], c_gold_label='truth')

            # TODO: setup() calls produce_stan_data(), even in semisupervised cases
            if supervised_items is not None:
                renamed_supervised_items = experiments.rename_items(exp, supervised_items)
                experiments.set_supervised_items_preset(exp, renamed_supervised_items)
                experiments.make_supervised_standata(exp)
            # else:
            #     exp.produce_stan_data()

            exp.train(dem_iter=dem_iter, mas_iter=mas_iter)

            if run_merge:
                exp.set_merge_fn(merge_fn)
                exp.register_weighted_merge()

            exp.test(debug=False)
            predictions_dict = get_predictions_dict(exp, run_merge)

            if task_type == 'categorical':
                metrics_fns = {'acc': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}
                score_fn = lambda x, y: 1 if x == y else 0
            elif task_type in ['ordinal', 'numerical']:
                metrics_fns = {'mae': mae, 'mse': mse}
                score_fn = lambda x, y: abs(x - y)

            results_level, scores_dict = process_predictions(predictions_dict, 
                                                exp.golddict,  
                                                experiment_name, 
                                                dist_name, 
                                                eval_name,
                                                metrics_fns,
                                                score_fn)

            ''' item names get renamed during the setup process.
            We want to report the scores for each question, so we reverse the renaming ''' 
            itemdict_reversed = {val: key for key, val in exp.itemdict.items()}
            scores_dict_renamed = {method: {itemdict_reversed[item]: label for item, label in d.items()} for method, d in scores_dict.items()}

            scores_dict_outer[(eval_name, dist_name)] = scores_dict_renamed
            results += results_level
    results_df = pd.DataFrame(results)
    return results_df, scores_dict_outer


def make_numerical_fns(full_df):
    max_val = max(max(full_df['answer']), max(full_df['truth']))
    min_val = min(min(full_df['answer']), min(full_df['truth']))
    dist_fn = lambda x, y: abs(x - y)/(max_val - min_val)
    eval_fn = lambda x, y: 1 - dist_fn(x, y)
    return dist_fn, eval_fn

def main():
    pd.set_option('display.max_columns', None)

    parser = argparse.ArgumentParser(description='Run experiment on simple task')

    '''positional (required) arguments'''
    parser.add_argument('task_name', type=str, help='task/dataset name')
    parser.add_argument('task_type', type=str, help='task type (e.g. categorical, ordinal, numerical...)')
    parser.add_argument('answer_file', type=str, help='location of file with answer/labels')
    parser.add_argument('truth_file', type=str, help='location of file with truths')

    '''optional arguments'''
    parser.add_argument('--merge', dest='merge', action='store_const', const=True,
                        help='run merge experiments?', metavar='MERGE', default=False)

    parser.add_argument('--log-results', dest='log_dir', type=str, 
                        help='write results to user-specified directory', metavar='RESULTS_DIR')

    parser.add_argument('--gold-file', dest='gold_file', type=str,
                        help='User-provided distance function', metavar='GOLD_FILE')

    '''just for logging purposes'''

    #could be calculated
    parser.add_argument('--semi-supervised', dest='supervision_amt', type=float, 
                        help='run task as semi-supervised', metavar='PCT_TRAINING_SET', default=0.0)
    #could all be condensed into one maybe
    parser.add_argument('--fold', dest='fold', type=int, help='which fold?')
    parser.add_argument('--noise', dest='noise', type=float, help='noise level')
    parser.add_argument('--suffix', dest='suffix', type=str, help='label your trial (e.g. 1 or "test")')

    args = parser.parse_args()
    task_name = args.task_name
    task_type = args.task_type
    answer_file = args.answer_file
    truth_file = args.truth_file

    merge = args.merge
    log_dir = args.log_dir
    gold_file = args.gold_file
    supervision_amt = args.supervision_amt
    fold = args.fold
    noise = args.noise
    suffix = args.suffix

    if supervision_amt > 0:
        gold_file = args.gold_file
        df_supervised_items = pd.read_csv(gold_file)
        supervised_items = df_supervised_items['question'].unique()
    else:
        supervised_items = None

    annotation_df = pd.read_csv(answer_file)
    gold_df = pd.read_csv(truth_file).set_index('question')
    full_df = annotation_df.join(gold_df, how='inner', on='question')
    print("USERS:", full_df['worker'].nunique())
    print("ITEMS:", full_df['question'].nunique())
    print("ANSWERS:", len(full_df))

    if task_type in ['numerical', 'ordinal']:
        dist_fn, eval_fn = make_numerical_fns(full_df)
        dist_fn = lambda x, y: abs(x - y)
        eval_fn_dict = {"diff/range e": eval_fn}
        dist_fn_dict = {"diff/range d": dist_fn}
        if task_type == 'numerical':
            merge_fn = merge_functions.numerical_mean
        else:
            merge_fn = merge_functions.numerical_mean_rounded

    elif task_type == 'categorical':
        eval_fn_dict = {"exact match eval": lambda x, y: 1 if x == y else 0}
        dist_fn_dict = {"exact match dist": lambda x, y: 0 if x == y else 1}
        merge_fn = None

    results, scores_all = test_simple_experiment(task_name, 
                                    SimpleExperiment, 
                                    task_type, 
                                    full_df, 
                                    eval_fn_dict=eval_fn_dict, 
                                    dist_fn_dict=dist_fn_dict,
                                    merge_fn=merge_fn,
                                    run_merge=merge,
                                    supervised_items=supervised_items)
    results['supervision_amt'] = supervision_amt
    results['fold'] = fold
    results['noise'] = noise
    results['suffix'] = suffix

    print(scores_all)
    print(results)
    if log_dir:

        '''TODO: address when there is more than one dist fn or eval fn so 
        len(scores_all) > 1 '''
        assert(len(scores_all) == 1)
        scores_dict = list(scores_all.items())[0][1]

        for method, scores in scores_dict.items():
            with open(f'{log_dir}/{method.lower()}_scores_{task_name}_{suffix}.csv', 'w') as file:
                writer = csv.writer(file)
                for item, val in scores.items():
                    writer.writerow([item,val])


        filename = f"{log_dir}/results_{task_name}_{suffix}.csv"
        results.to_csv(filename, mode='w', header=True, index=False)


        ''' to fit with the old formatting requirements '''
        # for idx, row in results.iterrows():
        #     filename = f"{log_dir}/{row['method'].upper()}/results_{task_type}.csv"
        #     with open(filename, 'a') as outfile:
        #         writer = csv.writer(outfile)
        #         if (os.stat(filename).st_size == 0):
        #             if task_type == 'categorical':
        #                 writer.writerow(['task', 'accuracy', 'f1'])
        #             else:
        #                 writer.writerow(['task', 'mae', 'mse'])
        #         if task_type == 'categorical':
        #             writer.writerow([row['task'], row['accuracy'], row['f1_weighted'], row['f1_macro']])
        #         else:
        #             writer.writerow([row['task'], row['mae'], row['mse']])


if __name__ == "__main__":
    main()
