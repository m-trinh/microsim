'''
make some plots using post-sim ACS data

chris zhang 12/5/2019
'''

import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from Utils import format_chart

## Read in post-sim ACS (MD using RI para, Random Forest, seed=123)
tag = '20191205_211613'
acs = pd.read_csv('./output/output_%s_Main/acs_sim_%s.csv' % (tag, tag))
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']

# dict from family size to poverty line
ks = range(1, 21)
vs = [11670, 15730, 19790, 23850, 27910, 31970, 36030, 40090] + \
     list(range(40090 + 4060, 40090 + 4060*(len(ks)-8+1), 4060))
vs = [v/1000 for v in vs]
dct_pov = dict(zip(ks, vs))

# get low-income workers in ACS (excl. missing income)
acs['pov200'] = [int(x[0]<2*dct_pov[x[1]]) if not np.isnan(x[0]) else np.nan for x in acs[['faminc', 'NPF']].values]

## How many low-income workers would get benefits?
# dict from leave type to total number of takeups (agg across takeup_type=1)
dct_bene = OrderedDict()
for t in types:
     dct_bene[t] = acs[(acs['pov200']==1) & (acs['takeup_%s' % t]==1)]['PWGTP'].sum()
## Plot
# Number of benes, by leave type
title = 'Number of low-income workers receiving leave pay, by leave type'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ind = np.arange(len(types))
ys = dct_bene.values()
width = 0.5
ax.bar(ind, ys, width, align='center', capsize=5, color='#1aff8c', ecolor='white')
ax.set_ylabel('Number of workers')
ax.set_xticks(ind)
ax.set_xticklabels(('Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent'))
ax.yaxis.grid(False)
format_chart(fig, ax, title)
plt.savefig('./draft/demo_external_20191213/MD_RI_rf_low_inc_bene_counts.png', facecolor='#333333', edgecolor='white') #

## How much benefit would go to low-income worker families?
# set annual benefit = 0 for missing amount
for t in types:
     acs.loc[acs['annual_benefit_%s' % t].isna(), 'annual_benefit_%s' % t] = 0
acs['annual_benefit_all'] = [x.sum() for x in acs[['annual_benefit_%s' % t for t in types]].values]
## Plot
# Number of benes, by benefit amount
title = 'Number of low-income worker recipients, by benefit levels'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
xs = acs[(acs['pov200']==1) & (acs['annual_benefit_all']>0)]['annual_benefit_all']
wts = acs[(acs['pov200']==1) & (acs['annual_benefit_all']>0)]['PWGTP']
binwidth = 500
plt.hist(xs, weights=wts, bins=range(0, 35000 + binwidth, binwidth), color='#1aff8c')
ax.set_ylabel('Number of workers')
ax.set_xlabel('$ Benefits Received')
ax.yaxis.grid(False)
format_chart(fig, ax, title)
plt.savefig('./draft/demo_external_20191213/MD_RI_rf_low_inc_benefit_amount.png', facecolor='#333333', edgecolor='white') #

## How much more leave taking would occur among low-income worker families?
# get increase in total leave length for each leave type
# fill missing value=0 for sq-len and cf-len
# then get a dict from leave type to agg-level growth in lengths
for t in types:
     acs.loc[acs['len_%s' % t].isna(), 'len_%s' % t] = 0
     acs.loc[acs['cfl_%s' % t].isna(), 'cfl_%s' % t] = 0
     acs['dlen_%s' % t] = (acs['cfl_%s' % t] - acs['len_%s' % t])
ks = types
vs = [(acs[acs['pov200']==1]['dlen_%s' % t]*acs[acs['pov200']==1]['PWGTP']).sum() /
      (acs[acs['pov200']==1]['len_%s' % t]*acs[acs['pov200']==1]['PWGTP']).sum() for t in types]
vs = [100*v for v in vs]
dct_dlen = OrderedDict(zip(ks, vs))

# Length growth, by leave type
title = 'Increase in leave lengths taken by low-income workers, by leave type'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ind = np.arange(len(types))
ys = dct_dlen.values()
width = 0.5
ax.bar(ind, ys, width, align='center', capsize=5, color='#1aff8c', ecolor='white')
ax.set_ylabel('Percent increase')
ax.set_xticks(ind)
ax.set_xticklabels(('Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent'))
ax.yaxis.grid(False)
format_chart(fig, ax, title)
plt.savefig('./draft/demo_external_20191213/MD_RI_rf_low_inc_length_growth.png', facecolor='#333333', edgecolor='white') #

## average dlen in maternity leave (Sherlock et al. 2008 reports 1 month increase in matdis leave>>child performance)\
x = acs[acs['dlen_matdis']>0]['dlen_matdis'].median()
print('Average increase in maternity leave among workers with any matdis increase = %s' % x)

##### check
# acs[acs['dlen_own']<0][['len_own', 'cfl_own', 'prop_pay', 'anypay', 'taker', 'needer']]
