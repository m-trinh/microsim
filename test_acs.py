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
    # R - keep eligible workers
    try:
        d = d[d['eligworker']==1]
    except KeyError:
        pass
    # R - rename len cols
    dct_cols = dict(zip(['squo_length_%s' % x for x in types], ['len_%s' % x for x in types]))
    dct_cols.update(dict(zip(['length_%s' % x for x in types], ['cfl_%s' % x for x in types])))
    dct_cols.update(dict(zip(['plen_%s' % x for x in types], ['cpl_%s' % x for x in types])))
    dct_cols.update(dict(zip(['takes_up_%s' % x for x in types], ['takeup_%s' % x for x in types])))
    dct_cols.update({'weeks_worked': 'wkswork'})
    d = d.rename(columns=dct_cols)

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
fp_p = './output/output_20200217_151534_main simulation/acs_sim_20200217_151534.csv'
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
lens = ['len_%s' % x for x in types]
lens+= ['mnl_%s' % x for x in types]
lens+= ['cfl_%s' % x for x in types]
lens+= ['cpl_%s' % x for x in types]
cols_cost = ['takeup_%s' % x for x in types]
cols_cost += ['wage12', 'wkswork']

cols = [id] + Xs + ys + [w] + lens + cols_cost

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

# check lens
for c in lens:
    # fillna with 0 for dp[c] for Py
    dp[c] = dp[c].fillna(0)

    if c[:2] != 'z_':
        print('----------------- xvar = %s -----------------------' % c)
        print('--- %s - Py ---' % c)
        if len(dp[c].isna().value_counts()) == 2:
            print('nrow missing = %s' % dp[c].isna().value_counts()[True])
        print(dp[c].mean())
        print('--- %s - R ---' % c)
        if len(dr[c].isna().value_counts()) == 2:
            print('nrow missing = %s' % dr[c].isna().value_counts()[True])
        print(dr[c].mean())
# TODO: coveligd - both Py/R no missing. But #s mismatch

## Compute total outlay
# params
params = {}
params['wkbene_cap'] = 795
params['rrp'] = 0.6
pow_pop_multiplier = 1.02
def get_costs(df):
    # apply take up flag and weekly benefit cap, and compute total cost, 6 types
    costs = {}
    for t in types:
        # v = capped weekly benefit of leave type
        v = [min(x, params['wkbene_cap']) for x in
             ((df['cpl_%s' % t] / 5) * (df['wage12'] / df['wkswork'] * params['rrp']))]
        # inflate weight for missing POW
        w = df['PWGTP'] * pow_pop_multiplier
    
        # get program cost for leave type t - sumprod of capped benefit, weight, and takeup flag for each ACS row
        costs[t] = (v * df['cpl_%s' % t] / 5 * w * df['takeup_%s' % t]).sum()
    costs['total'] = sum(list(costs.values()))
    return costs
cp = get_costs(dp)
cr = get_costs(dr)

# R has much larger costs on type own, check below
for c in ['len_own', 'mnl_own', 'cfl_own', 'cpl_own']:
    print('dp[%s].mean()\n' % c, dp[c].mean())
    print('dr[%s].mean()\n' % c, dr[c].mean())
    print('-----------------')

# check take up flags for own
dp[dp['takeup_own']==1]['PWGTP'].sum()
dr[dr['takeup_own']==1]['PWGTP'].sum()
# much bigger! seems like R has used wrong base for eligible worker pop
# should be ~350k but not 415k as in Excel state take up data
# TODO: LP to use ACS eligworker est pop as base to re-gen emp take up rates, apply in GUI and re-run model

# TODO: R's lens are much SHORTER than Py's for all types, check why



