'''
model validation for Issue Brief 2 (Summary of Model Testing Memo)

chris zhang 2/6/2020
'''
# TODO: for take/leave/resp_len, cross-validate using only rows with valid outvars, don't fillna. See if results change.
# TODO: in IB2 writeup - explain why ML outperforms random draw mostly for leave needs but not leave taking. own/matdis
# TODO: in writeup - pop level results overshoot is good - expected because sim design w. indep logit.
# Overshooted pop level results (total leavers/takers, total # leaves taken) should not be an issue for
# policy simulation given low take up. We just have a larger pool of takers/needers to choose from. Bias introduced by
# this larger pool is limited, as shown by random draw VS ML results that a big part of leave taking/need cannot be
# properly predicted via model, likely unobservables (health, intra-hh bargain, time preference, career choice, etc.)

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
import json
from time import time
from _5a_aux_functions import *
import sklearn.linear_model, sklearn.naive_bayes, sklearn.neighbors, sklearn.tree, sklearn.ensemble, \
    sklearn.gaussian_process, sklearn.svm
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import math
from datetime import datetime
import matplotlib.pyplot as plt
import random
import json
from datetime import datetime

# a function to get yhat via k-fold train-test split (stratified) and prediction
def get_kfold_yhat(X, y, clf, fold, random_state):
    '''
    :param y: must be pd.Series with name
    '''

    yhat = pd.Series([],name=y.name + '_hat')
    skf = StratifiedKFold(n_splits=fold)
    for train_index, test_index in skf.split(X, y):
        Xtr, Xts = X.loc[train_index, ], X.loc[test_index, ]
        ytr, yts = y.loc[train_index, ], y.loc[test_index, ]
        wtr, wts = d[col_w][Xtr.index], d[col_w][Xts.index]
        # save a copy of Xts - get_sim_col() will mutate Xts['age'] etc under Naive Bayes
        # which would make logic control infeasible
        Xts_copy = Xts.copy()
        # get sim col
        yhat_fold = get_sim_col(Xtr,ytr, wtr, Xts, clf, random_state=random_state)
        if 'matdis' in y.name:
            yhat_fold = pd.Series(yhat_fold, index=Xts[(Xts['female']==1)
                                                       & (Xts['nochildren']==0)
                                                       & (Xts['age']<=50)].index, name=yhat.name)
        elif 'bond' in y.name:
            yhat_fold = pd.Series(yhat_fold, index=Xts[(Xts['nochildren']==0)
                                                       & (Xts['age']<=50)].index, name=yhat.name)
        elif 'spouse' in y.name:
            yhat_fold = pd.Series(yhat_fold, index=Xts[(Xts['nevermarried']==0)
                                                       & (Xts['divorced']==0)].index, name=yhat.name)
        else:
            yhat_fold = pd.Series(yhat_fold, index=yts.index, name=yhat.name)

        # logic control - see simulation engine code
        if 'matdis' in yhat_fold.name:
            yhat_fold.loc[Xts['female']!=1] = 0
        if 'illspouse' in yhat_fold.name:
            yhat_fold.loc[(Xts['nevermarried'] == 1) | (Xts['divorced'] == 1)] = 0
        if 'bond' in yhat_fold.name:
            yhat_fold.loc[Xts['nochildren'] == 1] = 0
        if 'matdis' in yhat_fold.name:
            yhat_fold.loc[Xts['nochildren'] == 1] = 0
        if 'bond' in yhat_fold.name:
            yhat_fold.loc[Xts_copy['age']>50] = 0
        if 'matdis' in yhat_fold.name:
            yhat_fold.loc[Xts_copy['age']>50] = 0

        yhat = yhat.append(yhat_fold)

    yhat = yhat.sort_index()
    return yhat

# a function to use given classifier to generate kfold prediction cols
def update_df_with_kfold_cols(d, col_Xs, col_ys, clf, random_state):
    X, ys = d[col_Xs], d[col_ys]
    for col in col_ys:
        y = ys[col]
        yhat = get_kfold_yhat(X, y, clf, fold, random_state)
        if yhat.name in d.columns:
            del d[yhat.name]
        d = d.join(yhat)

    ## Get derived cols, both true/predicted
    d['taker'] = [max(x) for x in d[['take_%s' % x for x in types]].values]
    d['taker_hat'] = [max(x) for x in d[['take_%s_hat' % x for x in types]].values]
    d['needer'] = [max(x) for x in d[['need_%s' % x for x in types]].values]
    d['needer_hat'] = [max(x) for x in d[['need_%s_hat' % x for x in types]].values]
    # d['num_leave_reasons_taken'] = [np.nansum(x) for x in d[['take_%s' % x for x in types]].values] #<<use A4a_CAT
    d['num_leave_reasons_taken_hat'] = [np.nansum(x) for x in d[['take_%s_hat' % x for x in types]].values]

    return d

## Population level results of clf-based pred cols
def get_population_level_results(d, clf_name):
    # clf_name: str name of classifier
    out = {}
    out['true'], out['pred'] = {}, {}
    out['true']['n_takers'] = d[d['taker']==1]['weight'].sum()
    out['true']['n_needers'] = d[d['needer']==1]['weight'].sum()
    out['true']['n_reasons_taken'] = (d_raw['a4_cat']*d_raw['weight']).sum() # directly refer to raw FMLA var
    out['pred']['n_takers'] = d[d['taker_hat']==1]['weight'].sum()
    out['pred']['n_needers'] = d[d['needer_hat']==1]['weight'].sum()
    out['pred']['n_reasons_taken'] = (d['num_leave_reasons_taken_hat']*d['weight']).sum()
    return (clf_name, out)

## Individual level results of clf-based pred cols
def get_individual_level_results(d, clf_name, weighted_test=True):
    # clf_name: str name of classifier
    # weigthed_test: if True then weight test sample using FMLA weights, default=False
    out = {}
    out['precision'], out['recall'], out['f1'] = {}, {}, {}
    for v in ['taker', 'needer', 'resp_len'] + ['take_own', 'need_own', 'take_matdis', 'need_matdis']:
        dd = d.copy() # make a copy of original df. dd updates depending on v (removes yhat=NA rows from logic control)
        if 'matdis' in v:
            dd = dd[dd[v + '_hat'].notna()] # reduce d to rows with valid yhat (NA rows from logic control)
            dd = dd.reset_index(drop=True)
        # get weight col from dd
        sample_weight = np.ones(dd.shape[0])
        if weighted_test:
            sample_weight = dd['weight']
        # update output dict with performance scores
        out['precision'][v] = precision_score(dd[v], dd['%s_hat' % v], sample_weight=sample_weight)
        out['recall'][v] = recall_score(dd[v], dd['%s_hat' % v], sample_weight=sample_weight)
        out['f1'][v] = f1_score(dd[v], dd['%s_hat' % v], sample_weight=sample_weight)
    return (clf_name, out)

## Random state
random_seed = 12345
random_state = np.random.RandomState(random_seed)

## Read in FMLA data, get cols, fillna
fmla_wave = 2018 # 2012 or 2018
d = pd.read_csv('./data/fmla/fmla_%s/fmla_clean_%s.csv' % (fmla_wave, fmla_wave))
# drop military workers and reset index
d = d.drop(d[d['occ_11']==1].index)
d = d.reset_index(drop=True)
d_raw = d.copy()
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
col_Xs, col_ys, col_w = get_columns(fmla_wave, types)
d = d[col_Xs + col_ys + [col_w]]
d = fillna_df(d, random_state=random_state)
X, ys = d[col_Xs], d[col_ys]

## Set up
fold = 4 # cannot be more than # minor cases in most imbalanced outvar
part_size = int(round(len(d) / fold, 0))

# classifiers
clfs = [sklearn.dummy.DummyClassifier(strategy='stratified')]
clfs += [xgboost.XGBClassifier()]
clfs += [sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)]
clfs += [sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)]
clfs += [sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='ovr', random_state=random_state)]
clfs += [sklearn.naive_bayes.MultinomialNB()]
clfs += [sklearn.ensemble.RandomForestClassifier(random_state=random_state)]
clfs += [sklearn.linear_model.RidgeClassifier()]
# clfs += [sklearn.svm.SVC(probability=True, gamma='auto', random_state=random_state)]

# clfs = [sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='ovr', random_state=random_state)]


## Get output
out_pop, out_ind = {}, {}
for clf in clfs:
    # print clf name
    if type(clf)!=str: # only str clf is 'random draw'
        clf_name = clf.__class__.__name__
    else:
        clf_name = clf
    # get results
    print('Getting results for clf = %s' % clf_name)
    t0 = time()
    d = update_df_with_kfold_cols(d, col_Xs, col_ys, clf, random_state=random_state)
    print('Time needed to get results = %s' % round((time()-t0), 0))
    clf_name, out_pop_clf = get_population_level_results(d, clf_name)
    clf_name, out_ind_clf = get_individual_level_results(d, clf_name)
    if isinstance(clf, sklearn.neighbors.KNeighborsClassifier):
        clf_name += str(int(clf.get_params()['n_neighbors']))
    out_pop[clf_name] = out_pop_clf
    out_ind[clf_name] = out_ind_clf

# save as json
dir_out = 'C:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_2/'
fp_out_json_pop = dir_out + 'results/pop_level_%s_k%s.json' % (fmla_wave, fold)
fp_out_json_ind = dir_out + 'results/ind_level_%s_k%s.json' % (fmla_wave, fold)

with open(fp_out_json_pop, 'w') as f:
    json.dump(out_pop, f, sort_keys=True, indent=4)
with open(fp_out_json_ind, 'w') as f:
    json.dump(out_ind, f, sort_keys=True, indent=4)

#------------------------
# Plot
#------------------------
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from Utils import format_chart
import json

# fp out
fp_out = 'C:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_2/results/'
# outfile suffix
suffix = '_%s_k%s_%s_%s' % (fmla_wave, fold, 'SimpleImputer', 'XtsWeighted')
# classifier class names
clf_class_names = ['DummyClassifier', 'XGBClassifier', 'LogisticRegression', 'KNeighborsClassifier5', 'MultinomialNB', 'RandomForestClassifier',
         'RidgeClassifier'] # , 'SVC'
# classifier plot labels
clf_labels = ('Random Draw', 'XGB', 'Logit', 'KNN', 'Naive Bayes', 'Random Forest', 'Ridge') # , 'SVC'

# Pop level results - worker counts
with open(fp_out_json_pop, 'r') as j:
    dp = json.load(j)
dp_raw = dp.copy()
dp = {k:v['pred'] for k, v in dp_raw.items()}
dp = pd.DataFrame.from_dict(dp)
dp = dp[clf_class_names]
for c in dp.columns:
    dp[c] = [x/10**6 for x in dp[c]]
title = 'Population Level Validation Results - Worker Counts'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ys = dp.loc['n_takers'].values
zs = dp.loc['n_needers'].values
ind = np.arange(len(ys))
width = 0.2
bar1 = ax.bar(ind-width/2, ys, width, align='center', capsize=5, color='indianred', ecolor='grey')
bar2 = ax.bar(ind+width/2, zs, width, align='center', capsize=5, color='tan', ecolor='grey')
ax.set_ylabel('Millions of workers')
ax.set_xticks(ind)
ax.set_xticklabels(clf_labels)
ax.yaxis.grid(False)
ax.legend( (bar1, bar2), ('Leave Takers', 'Leave Needers') )
ax.ticklabel_format(style='plain', axis='y')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
format_chart(fig, ax, title, bg_color='white', fg_color='k')
# add horizontal bar for true numbers
n_takers_true = dp_raw['DummyClassifier']['true']['n_takers']/10**6
plt.axhline(y=n_takers_true, color='indianred', linestyle='--')
hline_offset = 1.025
hline_text = 'Actual Number of Takers: %s million' % (round(n_takers_true, 1))
plt.text(2, n_takers_true * hline_offset, hline_text, horizontalalignment='center', color='k')
n_needers_true = dp_raw['DummyClassifier']['true']['n_needers']/10**6
plt.axhline(y=n_needers_true, color='tan', linestyle='--')
hline_offset = 1.025
hline_text = 'Actual Number of Needers: %s million' % (round(n_needers_true, 1))
plt.text(2, n_needers_true * hline_offset, hline_text, horizontalalignment='center', color='k')
# save
plt.savefig(fp_out + 'pop_level_workers%s.png' % suffix, facecolor='white', edgecolor='grey') #

# Pop level results - leave counts
dp = {k:v['pred'] for k, v in dp_raw.items()}
dp = pd.DataFrame.from_dict(dp)
dp = dp[clf_class_names]
for c in dp.columns:
    dp[c] = [x/10**6 for x in dp[c]]
title = 'Population Level Validation Results - Leaves Taken'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ys = dp.loc['n_reasons_taken'].values
ind = np.arange(len(ys))
width = 0.2
bar1 = ax.bar(ind, ys, width, align='center', capsize=5, color='indianred', ecolor='grey')
ax.set_ylabel('Number of Leaves')
ax.set_xticks(ind)
ax.set_xticklabels(clf_labels)
ax.yaxis.grid(False)
ax.ticklabel_format(style='plain', axis='y')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
format_chart(fig, ax, title, bg_color='white', fg_color='k')
# add horizontal bar for true numbers
n_leaves_true = dp_raw['DummyClassifier']['true']['n_reasons_taken']/10**6
plt.axhline(y=n_leaves_true, color='indianred', linestyle='--')
hline_offset = 1.025
hline_text = 'Actual Number of Leaves: %s million' % (round(n_leaves_true, 1))
plt.text(2, n_leaves_true * hline_offset, hline_text, horizontalalignment='center', color='k')
# save
plt.savefig(fp_out + 'pop_level_leaves%s.png' % suffix, facecolor='white', edgecolor='grey') #

# Individual Level Results - focus on a given yvar
# yvar = 'taker'
yvar = 'resp_len'
for yvar in ['taker', 'needer', 'resp_len', 'take_own', 'need_own', 'take_matdis', 'need_matdis']:
    with open(fp_out_json_ind, 'r') as j:
        di = json.load(j)
    di = pd.DataFrame.from_dict(di)
    di = di[clf_class_names]
    title = 'Individual Level Validation Results - Performance Measures\nOutcome = %s' % yvar
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ys = [x[yvar] for x in di.loc['precision'].values]
    zs = [x[yvar] for x in di.loc['recall'].values]
    ws = [x[yvar] for x in di.loc['f1'].values]
    ind = np.arange(len(ys))
    width = 0.2
    bar1 = ax.bar(ind-width, ys, width, align='center', capsize=5, color='indianred', ecolor='grey')
    bar2 = ax.bar(ind, zs, width, align='center', capsize=5, color='tan', ecolor='grey')
    bar3 = ax.bar(ind+width, ws, width, align='center', capsize=5, color='slategray', ecolor='grey')
    ax.set_ylabel('Performance Measure')
    ax.set_xticks(ind)
    ax.set_xticklabels(clf_labels)
    ax.yaxis.grid(False)
    ax.legend( (bar1, bar2, bar3), ('Precision', 'Recall', 'F1') )
    ax.ticklabel_format(style='plain', axis='y')
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    format_chart(fig, ax, title, bg_color='white', fg_color='k')
    # save
    plt.savefig(fp_out + 'ind_level%s.png' % suffix, facecolor='white', edgecolor='grey') #

# Program Outlay Results
from os import listdir
from os.path import isfile, join

# Make a df of total program costs
costs = {}
sts = ['ca', 'nj', 'ri']
for s in sts:
    costs[s] = {}
methods = ['logit', 'knn5', 'nb', 'rf', 'ridge', 'svc']
for st in sts:
    for method in methods:
        if st=='nj':
            fp='./output/_mnl/%s_%s/' % (st, method)
        else:
            fp = './output/%s_%s/' % (st, method)
        fs = [f for f in listdir(fp) if isfile(join(fp, f))]
        f_dc = [f for f in fs if 'program_cost' in f][0]
        dc = pd.read_csv(fp+f_dc)
        for c in dc.columns:
            if c!='type':
                dc[c] = [x/10**6 for x in dc[c]]
        costs[st][method] = dc.loc[dc['type']=='total', 'cost'].values[0]
costs = pd.DataFrame.from_dict(costs)
# plot
title = 'Program Cost Validation Results'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
costs = costs.reindex(['logit', 'knn5', 'nb', 'rf', 'ridge', 'svc'])
ys = costs['ca']
zs = costs['nj']
ws = costs['ri']
ind = np.arange(len(ys))
width = 0.2
bar1 = ax.bar(ind-width, ys, width, align='center', capsize=5, color='indianred', ecolor='grey')
bar2 = ax.bar(ind, zs, width, align='center', capsize=5, color='tan', ecolor='grey')
bar3 = ax.bar(ind+width, ws, width, align='center', capsize=5, color='slategray', ecolor='grey')
ax.set_ylabel('Program Cost')
ax.set_xticks(ind)
ax.set_xticklabels(('Random Draw', 'Logit', 'KNN', 'Naive Bayes', 'Random Forest', 'Ridge', 'SVC'))
ax.yaxis.grid(False)
ax.legend( (bar1, bar2, bar3), ('California', 'New Jersey', 'Rhode Island') )
ax.ticklabel_format(style='plain', axis='y')
#ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
format_chart(fig, ax, title, bg_color='white', fg_color='k')
# save
plt.savefig(fp_out + 'program_costs.png', facecolor='white', edgecolor='grey') #



################################################################################################

