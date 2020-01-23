'''
check consistency of cleaned FMLA between Py/R

chris zhang 1/22/2020
'''
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from _5a_aux_functions import get_bool_num_cols

## Read in data
dr = pd.read_csv('./PR_comparison/check_clean_fmla/fmla_clean_R.csv')
dp = pd.read_csv('./data/fmla_2012/fmla_clean_2012.csv')
dp = dp.rename(columns={'eligworker':'fmla_eligworker'})
## Get diff in cols
print(set(dr.columns) - set(dp.columns))

## Reduce to needed cols
# id
id = 'empid'
# Xs, ys, w
Xs = ['widowed', 'divorced', 'separated', 'nevermarried',
      'female', 'age', 'agesq',
      'ltHS', 'someCol', 'BA', 'GradSch',
      'black', 'other', 'asian', 'native', 'hisp',
      'nochildren', 'faminc', 'coveligd']
ys = ['take_own', 'take_matdis', 'take_bond', 'take_illchild', 'take_illspouse', 'take_illparent']
ys += ['need_own', 'need_matdis', 'need_bond', 'need_illchild', 'need_illspouse', 'need_illparent']
ys += ['resp_len']
w = 'weight'
# other vars
Vs = ['covwrkplace', 'fmla_eligworker']
#Vs = ['B6_1_CAT', 'B12_1', 'LEAVE_CAT']
# reduce cols
dr = dr[[id]+Xs+ys+[w]+Vs]
dp = dp[[id]+Xs+ys+[w]+Vs]
# get bool/num cols
bool_cols, num_cols = get_bool_num_cols(dr)

## Compare bool cols, num cols
for c in bool_cols:
    print('----------------- xvar = %s -----------------------' % c)
    print('--- %s - Py ---' % c)
    print(dp[c].value_counts().sort_index())
    print('--- %s - R ---' % c)
    print(dr[c].value_counts().sort_index())

for c in num_cols:
    print('----------------- xvar = %s -----------------------' % c)
    print('--- %s - Py ---' % c)
    if len(dp[c].isna().value_counts())==2:
        print('nrow missing = %s' % dp[c].isna().value_counts()[True])
    print(dp[c].mean())
    print('--- %s - R ---' % c)
    if len(dr[c].isna().value_counts())==2:
        print('nrow missing = %s' % dr[c].isna().value_counts()[True])
    print(dr[c].mean())
# coveligd, take_bond, need_bond,
# TODO: update Py/R - coveligd is nan if covwrkplace/fmla_eligworker=1/nan, nan/1, or nan/nan
# TODO: [done] update Py - need_bond def
# TODO: update R - need_bond, check empid = 1419. need_bond should=nan as B12_1=nan. Force need_bond=nan if B12_1=nan.

cols = ['need_bond', 'need_matdis', 'B6_1_CAT', 'B12_1', 'LEAVE_CAT']
dr = dr[['empid'] + cols]
dr['empid'] = dr['empid'] - 1
dp = dp[['empid'] + cols]
drp = pd.merge(dr, dp, how='outer', on=['empid', 'need_bond'], indicator=True)
drp[drp._merge!='both']
# >> in dr, need_bond = nan for 3 cases, which are all=0 in Py
ids = [1418]
dr[dr.empid.isin(ids)]
dp[dp.empid.isin(ids)]