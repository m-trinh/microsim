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
def preprocess_data(fp_acs):
    # Read in ACS
    d = pd.read_csv(fp_acs)
    # id
    id = 'SERIALNO'
    # Xs, ys, w
    Xs = ['widowed', 'divorced', 'separated', 'nevermarried',
              'female', 'age','agesq',
              'ltHS', 'someCol', 'BA', 'GradSch',
              'black', 'other', 'asian','native','hisp',
              'nochildren','faminc','coveligd']
    w = 'PWGTP'
    # reduce cols
    d = d[[id] + Xs + [w]]

    ## Standardize
    # use N-ddof = N-1 as in R
    # https://stackoverflow.com/questions/27296387/difference-between-r-scale-and-sklearn-preprocessing-scale/27297618

    # TODO: check how fillna works in R - need to be same as Python
    # drop missing rows if any col is missing
    d = d.dropna(subset=Xs)
    # standardize
    for X in Xs:
        d['z_%s' % X] = (d[X] - d[X].mean()) / np.std(d[X], axis=0, ddof=1)
    print(d[['SERIALNO', 'age','z_age', 'female', 'z_female']].head())
    # cols to return
    cols = (Xs, w)
    return (d, cols)


## Read in data
# Python
#dp = pd.read_csv('./data/acs/ACS_cleaned_forsimulation_2016_ri.csv')
fp_p = './data/acs/ACS_cleaned_forsimulation_2016_ri.csv'
# R
#dr = pd.read_csv('./PR_comparison/check_acs/RI_work.csv')
fp_r = './PR_comparison/check_acs/RI_work.csv'

## preprocess

dp, cols = preprocess_data(fp_p)
dr = preprocess_data(fp_r)[0]

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

#TODO: R - female=1 80+%?
#TODO: lths Py 433 R 1382, NT check
#TODO: coveligd Py 13692 R 9457
#TODO: faminc Py 117342.61 R 118825.41