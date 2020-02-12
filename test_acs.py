'''
test consistency of ACS between Python and R

chris zhang 1/24/2020
'''
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from _5a_aux_functions import get_bool_num_cols

# a function to read in data and reduce cols
def preprocess_data(fp_acs, cols):
    # Read in ACS
    d = pd.read_csv(fp_acs)
    # keep eligible workers
    try:
        d = d[d['eligworker']==1]
    except KeyError:
        pass
    # reduce cols
    d = d[cols]

    # ## Standardize
    # # use N-ddof = N-1 as in R
    # # https://stackoverflow.com/questions/27296387/difference-between-r-scale-and-sklearn-preprocessing-scale/27297618
    #
    # # TODO: check how fillna works in R - need to be same as Python
    # # drop missing rows if any col is missing
    # #d = d.dropna(subset=Xs)
    # # standardize
    # for X in Xs:
    #     d['z_%s' % X] = (d[X] - d[X].mean()) / np.std(d[X], axis=0, ddof=1)
    # print(d[['SERIALNO', 'age','z_age', 'female', 'z_female']].head())
    # cols to return
    return d


## Read in data
# Python
#dp = pd.read_csv('./data/acs/ACS_cleaned_forsimulation_2016_ri.csv')
#fp_p = './data/acs/ACS_cleaned_forsimulation_2016_ri.csv'
fp_p = './output/output_20200212_111204_main simulation/acs_sim_20200212_111204.csv'
# R
#dr = pd.read_csv('./PR_comparison/check_acs/RI_work.csv')
fp_r = './PR_comparison/check_acs/acs_20200206/test_ACS_RI.csv'

## preprocess

# id
id = 'SERIALNO'
# Xs, ys, w
Xs = ['widowed', 'divorced', 'separated', 'nevermarried',
      'female', 'age', 'agesq',
      'ltHS', 'someCol', 'BA', 'GradSch',
      'black', 'other', 'asian', 'native', 'hisp',
      'nochildren', 'faminc', 'coveligd']
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
ys = ['take_%s' % x for x in types] + ['need_%s' % x for x in types] + ['resp_len']
w = 'PWGTP'

cols = [id] + Xs + ys + [w]

dp = preprocess_data(fp_p, cols)
dr = preprocess_data(fp_r, cols)

## Check bool num cols
bool_cols, num_cols = get_bool_num_cols(dr)
for c in bool_cols:
    if c[:2]!='z_':
        print('----------------- xvar = %s -----------------------' % c)
        print('--- %s - Py ---' % c)
        print(dp[c].value_counts().sort_index())
        print('--- %s - R ---' % c)
        print(dr[c].value_counts().sort_index())

for c in num_cols:
    if c[:2]!='z_':
        print('----------------- xvar = %s -----------------------' % c)
        print('--- %s - Py ---' % c)
        if len(dp[c].isna().value_counts())==2:
            print('nrow missing = %s' % dp[c].isna().value_counts()[True])
        print(dp[c].mean())
        print('--- %s - R ---' % c)
        if len(dr[c].isna().value_counts())==2:
            print('nrow missing = %s' % dr[c].isna().value_counts()[True])
        print(dr[c].mean())

# TODO: coveligd - both Py/R no missing. But #s mismatch