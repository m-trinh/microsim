'''
model validation for Issue Brief 2 (Summary of Model Testing Memo)

chris zhang 2/6/2020
'''

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
    # d['num_leave_reasons_taken'] = [sum(x) for x in d[['take_%s' % x for x in types]].values] #<<use A4a_CAT in raw
    d['num_leave_reasons_taken_hat'] = [sum(x) for x in d[['take_%s_hat' % x for x in types]].values]

    return d

## Population level results of clf-based pred cols
def get_population_level_results(d, clf_name):
    # clf_name: str name of classifier
    out = {}
    out['true'], out['pred'] = {}, {}
    out['true']['n_takers'] = d[d['taker']==1]['weight'].sum()
    out['true']['n_needers'] = d[d['needer']==1]['weight'].sum()
    out['true']['n_reasons_taken'] = (d_raw['A4a_CAT']*d_raw['weight']).sum() # refer to raw FMLA var that directly asks it
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
    sample_weight = np.ones(d.shape[0])
    if weighted_test:
        sample_weight = d['weight']
    for v in ['taker', 'needer', 'resp_len']:
        out['precision'][v] = precision_score(d[v], d['%s_hat' % v], sample_weight=sample_weight)
        out['recall'][v] = recall_score(d[v], d['%s_hat' % v], sample_weight=sample_weight)
        out['f1'][v] = f1_score(d[v], d['%s_hat' % v], sample_weight=sample_weight)
    return (clf_name, out)

## Random state
random_seed = 12345
random_state = np.random.RandomState(random_seed)

## Read in FMLA data, get cols, fillna
d = pd.read_csv('./data/fmla/fmla_2012/fmla_clean_2012.csv')
d_raw = d.copy()
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
col_Xs, col_ys, col_w = get_columns(types)
d = d[col_Xs + col_ys + [col_w]]
d = fillna_df(d, random_state=random_state)
X, ys = d[col_Xs], d[col_ys]

## Set up
fold = 8
part_size = int(round(len(d) / fold, 0))

# classifiers
clfs = [sklearn.dummy.DummyClassifier(strategy='stratified')]
clfs += [sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)]
clfs += [sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)]
clfs += [sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='ovr', random_state=random_state)]
clfs += [sklearn.naive_bayes.MultinomialNB()]
clfs += [sklearn.ensemble.RandomForestClassifier(random_state=random_state)]
clfs += [sklearn.linear_model.RidgeClassifier()]
#clfs += [sklearn.svm.SVC(probability=True, gamma='auto', random_state=random_state)]

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
with open(dir_out + 'results/pop_level_k%s.json' % fold, 'w') as f:
    json.dump(out_pop, f, sort_keys=True, indent=4)
with open(dir_out + 'results/ind_level_k%s.json' % fold, 'w') as f:
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
suffix = '_k%s_%s_%s' % (fold, 'KNNImputer', 'XtsUnweighted')

# Pop level results - worker counts
fp_p = fp_out + 'pop_level_k%s.json' % fold
with open(fp_p, 'r') as j:
    dp = json.load(j)
dp_raw = dp.copy()
dp = {k:v['pred'] for k, v in dp_raw.items()}
dp = pd.DataFrame.from_dict(dp)
dp = dp[['DummyClassifier', 'LogisticRegression', 'KNeighborsClassifier5', 'MultinomialNB', 'RandomForestClassifier',
         'RidgeClassifier']] # , 'SVC'
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
ax.set_xticklabels(('Random Draw', 'Logit', 'KNN', 'Naive Bayes', 'Random Forest', 'Ridge')) # , 'SVC'
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
dp = dp[['DummyClassifier', 'LogisticRegression', 'KNeighborsClassifier5', 'MultinomialNB', 'RandomForestClassifier',
         'RidgeClassifier']] # , 'SVC'
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
ax.set_xticklabels(('Random Draw', 'Logit', 'KNN', 'Naive Bayes', 'Random Forest', 'Ridge')) # , 'SVC'
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

# Individual Level Results - focus on takers
fp_i = fp_out + 'ind_level_k%s.json' % fold
with open(fp_i, 'r') as j:
    di = json.load(j)
di = pd.DataFrame.from_dict(di)
di = di[['DummyClassifier', 'LogisticRegression', 'KNeighborsClassifier5', 'MultinomialNB', 'RandomForestClassifier',
         'RidgeClassifier']] # , 'SVC'
title = 'Individual Level Validation Results - Performance Measures'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ys = [x['taker'] for x in di.loc['precision'].values]
zs = [x['taker'] for x in di.loc['recall'].values]
ws = [x['taker'] for x in di.loc['f1'].values]
ind = np.arange(len(ys))
width = 0.2
bar1 = ax.bar(ind-width, ys, width, align='center', capsize=5, color='indianred', ecolor='grey')
bar2 = ax.bar(ind, zs, width, align='center', capsize=5, color='tan', ecolor='grey')
bar3 = ax.bar(ind+width, ws, width, align='center', capsize=5, color='slategray', ecolor='grey')
ax.set_ylabel('Performance Measure')
ax.set_xticks(ind)
ax.set_xticklabels(('Random Draw', 'Logit', 'KNN', 'Naive Bayes', 'Random Forest', 'Ridge')) #, 'SVC'
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


D = d.copy()
outs = [] # list of results across iter times of test runs
for iter in range(100):
    #random.seed(datetime.now())
    out = {} # a dict to store all results for all methods
    clf_names = [] # a list to store classifier names in output filename
    for clf in clfs:
        clf_name = ''
        if clf=='random draw':
            clf_name = 'randomDraw'
        elif isinstance(clf, sklearn.neighbors.KNeighborsClassifier):
            clf_name = clf.__class__.__name__
            clf_name += str(int(clf.get_params()['n_neighbors']))
        else:
            clf_name = clf.__class__.__name__
        clf_names.append(clf_name)


        print('Start running model. Method = %s' % clf_name)
        t0 = time()
        d = D.copy()

        ixs = list(d.index)
        random.shuffle(ixs)
        d = d.reindex(ixs)
        d = d.reset_index(drop=True)
        ixs_parts = []
        ixs_part_end = set(d.index) # initialize last partition - contains all unassigned rows in case mod(len(d), fold)!=0
        for kk in range(fold-1):
            ixs_part = d[(kk*part_size):((kk+1)*part_size)]['empid'].values
            ixs_parts.append(ixs_part)
            ixs_part_end = ixs_part_end - set(ixs_part)
        ixs_parts.append(np.array([x for x in ixs_part_end]))
        d = D.copy() # restore d as original

        ds = pd.DataFrame([]) # initialize df to store imputed results
        for kk in range(fold):
            ixs_ts = ixs_parts[kk]
            ixs_tr = np.array([x for x in set(d.index) - set(ixs_ts)])

            col_Xs, col_ys, col_w = get_columns()
            X = d.loc[ixs_tr, col_Xs]
            w = d.loc[X.index, col_w]
            Xa = d.loc[ixs_ts, col_Xs] # a fixed set of predictors for testing sample
            dkk = d.loc[ixs_ts, :].drop(columns=col_ys) # all cols but col_ys of train sample in d to store imputed values
            for c in col_ys:
                y = d.loc[ixs_tr, c]
                simcol = get_sim_col(X, y, w, Xa, clf)
                dkk = dkk.join(get_sim_col(X, y, w, Xa, clf))
            # Post-simluation logic control
            dkk.loc[dkk['male'] == 1, 'take_matdis'] = 0
            dkk.loc[dkk['male'] == 1, 'need_matdis'] = 0
            dkk.loc[(dkk['nevermarried'] == 1) | (dkk['divorced'] == 1), 'take_illspouse'] = 0
            dkk.loc[(dkk['nevermarried'] == 1) | (dkk['divorced'] == 1), 'need_illspouse'] = 0
            dkk.loc[dkk['nochildren'] == 1, 'take_bond'] = 0
            dkk.loc[dkk['nochildren'] == 1, 'need_bond'] = 0
            dkk.loc[dkk['nochildren'] == 1, 'take_matdis'] = 0
            dkk.loc[dkk['nochildren'] == 1, 'need_matdis'] = 0

            # Conditional simulation - anypay, doctor, hospital for taker/needer sample
            dkk['taker'] = dkk[['take_%s' % t for t in types]].apply(lambda x: max(x), axis=1)
            dkk['needer'] = dkk[['need_%s' % t for t in types]].apply(lambda x: max(x), axis=1)

            X = d.loc[ixs_tr, :]
            X = X[(X['taker'] == 1) | (X['needer'] == 1)][col_Xs]
            w = d.loc[X.index][col_w]
            Xa = dkk[(dkk['taker'] == 1) | (dkk['needer'] == 1)][col_Xs]

            if len(Xa) == 0:
                pass
            else:
                for c in ['anypay', 'doctor', 'hospital']:
                    y = d.loc[X.index][c]
                    dkk = dkk.drop(columns=[c])
                    dkk = dkk.join(get_sim_col(X, y, w, Xa, clf))
                # Post-simluation logic control
                dkk.loc[dkk['hospital'] == 1, 'doctor'] = 1

            # Conditional simulation - prop_pay for anypay=1 sample
            X = d.loc[ixs_tr, :]
            X = X[(X['anypay'] == 1) & (X['prop_pay'].notna())][col_Xs]
            w = d.loc[X.index][col_w]
            Xa = d.loc[(dkk.index), ]
            Xa = Xa[(Xa['anypay']==1) & (Xa['prop_pay'].notna())][col_Xs]

            # a dict from prop_pay int category to numerical prop_pay value
            # int category used for phat 'p_0', etc. in get_sim_col
            v = d.prop_pay.value_counts().sort_index().index
            k = range(len(v))
            d_prop = dict(zip(k, v))
            D_prop = dict(zip(v, k))

            if len(Xa) == 0:
                pass
            else:
                y = d.loc[X.index]['prop_pay'].apply(lambda x: D_prop[x])
                dkk = dkk.drop(columns=['prop_pay'])
                yhat = get_sim_col(X, y, w, Xa, clf)
                # prop_pay labels are from 1 to 6, sklearn classes_ gives 0 to 5, increase label by 1
                if clf_name!='randomDraw':
                    yhat = pd.Series(data=yhat.values+1, index=Xa.index, name='prop_pay')
                print(yhat.value_counts())
                dkk = dkk.join(yhat)
                dkk.loc[dkk['prop_pay'].notna(), 'prop_pay'] = dkk.loc[dkk['prop_pay'].notna(), 'prop_pay'].apply(
                    lambda x: d_prop[x])

            ds = ds.append(dkk)
        # leaves taken of each worker
        ds = ds.drop(columns=['num_leaves_taken'])
        ds['num_leaves_taken'] = ds[['take_%s' % t for t in types]].apply(lambda x: sum(x), axis=1)
        # prop_pay = 0 if nan in d or ds
        #ds.loc[ds['prop_pay'].isna(), 'prop_pay'] = 0
        #d.loc[d['prop_pay'].isna(), 'prop_pay'] = 0
        # Compute performance metrics - apply weights as necessary

        row = {}
        wcol = 'weight'
        rpw_cols = ['rpl%s' % "{0:0=2d}".format(x) for x in range(1, 81)]
        # Aggregate level metrics
        # total number of leave takers
        # total number of leaves taken
        # total number of leave needers
        # prop_pay - mean
        vcols = ['taker', 'num_leaves_taken', 'needer', 'prop_pay']
        for vcol in vcols:
            if vcol!='prop_pay':
                mean, spread = get_mean_spread(ds, vcol, wcol, rpw_cols, how='sum')
            else:
                mean, spread = get_mean_spread(ds, vcol, wcol, rpw_cols, how='mean')
            ci_lower, ci_upper = mean - spread, mean + spread
            row[vcol] = {}
            row[vcol]['mean'] = mean
            row[vcol]['ci_lower'] = ci_lower
            row[vcol]['ci_upper'] = ci_upper

        # Individual level metrics
        # accuracy - taker
        # accuracy - needer
        # accuracy - prop_pay
        row['accuracy'] = {}
        ds = ds.rename(columns={'taker':'taker_sim','needer':'needer_sim','prop_pay':'prop_pay_sim'})
        for vcol in ['taker', 'needer', 'prop_pay']:
            if vcol=='prop_pay':
                # restore prop_pay = nan if 0 in d or ds
                # prop_pay is conditional - don't inflate metric with out-of-universe 0s
                # np.nan==np.nan will give False and not counted in num of acc metric
                ds.loc[ds['%s_sim' % vcol]==0, '%s_sim' % vcol] = np.nan
                d.loc[d[vcol]==0, vcol] = np.nan
            vv = pd.DataFrame(ds['%s_sim' % vcol]).join(d[vcol])
            acc = 0
            try:
                acc = (vv['%s_sim' % vcol]==vv[vcol]).value_counts()[True] / len(vv[vv[vcol].notna()])
            except KeyError:
                pass
            row['accuracy'][vcol] = acc

        # store row in out
        out[clf_name] = row
        print('method %s results stored in out. TElapsed = %s' % (clf_name, (time() - t0)))

    # Simplify classfier names in out
    clf_names_short = ['random',
                       'KNN_multi',
                       'KNN1',
                       'logit',
                       'Naive Bayes',
                       'random_forest',
                       'ridge_class']
    dn = dict(zip(clf_names, clf_names_short))
    for k, v in dn.items():
        out[v] = out.pop(k)

    # send a copy of out to outs
    outs.append(out.copy())
    print('Test simulation completed for iter run %s' % iter)
    print('-------------------------------------------------------------')

# Get average values across out's
OUT = {}
for k, v in outs[0].items():
        OUT[k] = {} # set structure across method
        for k1, v1 in v.items():
            OUT[k][k1] = {} # set structure across var or acc
            for k2, v2 in v1.items():
                OUT[k][k1][k2] = np.nan
for k, v in OUT.items():
    for k1, v1 in v.items():
        for k2, v2 in v1.items():
            v2 = 0
            for dct in outs:
                v2 += dct[k][k1][k2]/len(outs) # build v2 value for OUT
            OUT[k][k1][k2] = v2 # assign V2 value to OUT leaves

out = OUT.copy()

# save as json
jsn = json.dumps(out)
f = open('./output/test_within_fmla_results_all.json', 'w')
f.write(jsn)
f.close()

# # Read json
# with open('./output/test_within_fmla_results_all.json') as f:
#     outr = json.load(f)
# out['logit'] # results before saving to json
# outr['logit'] # results read from saved json


# Plots
# Aggregate measures
vars = ['taker','needer','num_leaves_taken', 'prop_pay']
txts = ['Actual Leave Takers: ',
        'Actual Leave Needers: ',
        'Actual Leaves Taken: ',
        'Actual Prop Pay Mean: ']
dtxt = dict(zip(vars, txts))
for var in vars:
    if var!='prop_pay':
        y_label = 'million'
        y_red = (d[var]*d['weight']).sum()/10**6
        text_red = '%s%s %s' % (dtxt[var], round(y_red, 1), y_label)
        ys = [out[x][var]['mean'] / 10 ** 6 for x in out.keys()]
        es = [0.5 * (out[x][var]['ci_upper'] - out[x][var]['ci_lower']) / 10 ** 6 for x in out.keys()]
    else:
        y_label = 'Proportion of Pay Received from Employer (%)'
        y_red = (d[d[var].notna()][var] * d[d[var].notna()]['weight'] / d[d[var].notna()]['weight'].sum()).sum()/10**(-2)
        text_red = '%s%s%%' % (dtxt[var], round(y_red, 1))
        ys = [out[x][var]['mean']/10**(-2) for x in out.keys()]
        es = [0.5 * (out[x][var]['ci_upper'] - out[x][var]['ci_lower'])/10**(-2) for x in out.keys()]

    N = len(clfs)
    inds = [dn[k] for k in clf_names]
    width = 0.5
    fig, ax = plt.subplots(figsize=(8.7, 5))
    ax.bar(inds, ys, width, yerr=es, align='center', capsize=5, color='khaki')
    ax.set_ylabel(y_label)
    ax.set_xticklabels(inds)
    ax.yaxis.grid(False)
    #plt.xticks(rotation=22)
    plt.axhline(y=y_red, color='r', linestyle='--')
    plt.text(2, y_red * 1.025, text_red, horizontalalignment='center', color='r')
    plt.title('Aggregate Measures: %s' % var)
    plt.savefig('./output/figs/test_within_fmla_agg_%s' % (var),bbox_inches='tight')
    plt.show()

# Individual measures
vars = ['taker', 'needer', 'prop_pay']
txts = ['Random Accuracy: ',
        'Random Accuracy: ',
        'Random Accuracy: ']
dtxt = dict(zip(vars, txts))
for var in vars:
    y_label = 'Percent'
    y_red = out['random']['accuracy'][var]*10**2
    text_red = '%s%s%%' % (dtxt[var], round(y_red, 1))
    ys = [out[x]['accuracy'][var] *10 ** 2 for x in out.keys() if x!='random']

    N = len(ys)
    inds = [dn[k] for k in clf_names[1:]] # drop first: randomDraw - now the benchmark
    width = 0.5
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(inds, ys, width, align='center', capsize=5, color='khaki')
    ax.set_ylabel(y_label)
    ax.set_xticklabels(inds)
    ax.yaxis.grid(False)
    plt.axhline(y=y_red, color='r', linestyle='--')
    plt.text(2, y_red * 1.025, text_red, horizontalalignment='center', color='r')
    plt.title('Individual Measures Accuracy: %s' % var)
    plt.savefig('./output/figs/test_within_fmla_indiv_%s' % (var))
    plt.show()

# ## Other factors
# # Leave prob factors, 6 types - TODO: code in wof in get_sim_col(), bound phat by max = 1
