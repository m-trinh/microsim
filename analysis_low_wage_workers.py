'''
Analysis of low wage workers (for Issue Brief 3)

chris zhang 10/6/2020
'''

import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from Utils import format_chart

## Set up local directory
fp_out = 'E:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_3/ib3_v3/'

## Read in post-sim ACS (MD using CA para, Logit Regularized, seed=12345)
tag = '20201007_170151' # tag = '20201006_093049' for xgb
st = 'md'
method = 'glm' # Simulation method label (in output PNG file name)
pow_multiplier = 1.02 # set to 1 if State of Work is unchecked in GUI
acs = pd.read_csv('./output/output_%s_main simulation/acs_sim_%s_%s.csv' % (tag,st, tag))
acs['PWGTP_POW'] = [int(x) for x in (acs['PWGTP']*pow_multiplier)]
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']



# dict from family size to poverty line
# 2020 poverty guidelines https://aspe.hhs.gov/poverty-guidelines
ks = range(1, 21)
vs = [12760, 17240, 21720, 26200, 30680, 35160, 39640, 44120] + \
     list(range(44120 + 4480, 44120 + 4480*(len(ks)-8+1), 4480))
dct_pov = dict(zip(ks, vs))

# get low-income workers in ACS (excl. missing income)
acs['pov200'] = [int(x[0]<2*dct_pov[x[1]]) if not np.isnan(x[0]) else np.nan for x in acs[['faminc', 'NPF']].values]

## How many low-income workers would get benefits?
# dict from leave type to total number of takeups (agg across takeup_type=1)
dct_bene = OrderedDict()
for t in types:
     dct_bene[t] = acs[(acs['pov200']==1) & (acs['takeup_%s' % t]==1)]['PWGTP_POW'].sum()

# ----------
# Plot
# ----------

## Number of leave takers across wage groups, no-program vs program
# get wage groups
# no-program taker status: taker
# with program taker status: cfl_type > 0  for any type
acs['taker_cf'] = [int(x>0) for x in np.nanmax(acs[['cfl_%s' % t for t in types]].values, axis=1)]

# top code wage
thre = np.percentile(acs[acs['taker_cf']==1]['wage12'], 95)
acs[(acs['taker_cf']==1) & (acs['wage12']<=thre)]['wage12'].hist()

# Number of leave takers, by income group
# title = 'Number of leave takers, by annual wage income group'
title = ''
xs = {
    'no_prog': acs[(acs['taker']==1) & (acs['wage12']<=thre)]['wage12'],
    'prog': acs[(acs['taker_cf']==1) & (acs['wage12']<=thre)]['wage12']
}
wts = {
    'no_prog': acs[(acs['taker']==1) & (acs['wage12']<=thre)]['PWGTP_POW'],
    'prog': acs[(acs['taker_cf']==1) & (acs['wage12']<=thre)]['PWGTP_POW']
}
binwidth = 5000
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
bar1 = plt.hist(xs['prog'], width=binwidth/2, align='mid',
                weights=wts['prog'], bins=range(0, int(thre) + binwidth, binwidth), color='indianred',
                alpha=0.8, edgecolor='black', label='With Hypothetical CA Program') #
bar2 = plt.hist(xs['no_prog'],
                weights=wts['no_prog'], bins=range(0, int(thre) + binwidth, binwidth), color='lightgrey',
                alpha=0.8, edgecolor='black', label='Without Paid Leave Program') #
plt.legend()
ax.set_ylabel('Number of workers')
ax.set_xlabel('$ Annual Wage Income')
ax.yaxis.grid(False)
format_chart(fig, ax, title,  bg_color='white', fg_color='k')
plt.savefig(fp_out + 'MD_CA_%s_takers.png' % method, facecolor='white', edgecolor='white') #
# get exact increase of total leave takers
x0, x1 = acs[acs['taker']==1]['PWGTP_POW'].sum(), acs[acs['taker_cf']==1]['PWGTP_POW'].sum()
print('Increase in total number of leave takers: %s percent, from %s to %s' % (round((x1-x0)/x0*100, 1), x0, x1))

## Percentage increase in number of leave takers across 10k wage income groups
# get plot data
binwidth=10000
takers = acs[acs['taker']==1][['PWGTP_POW']].groupby(pd.cut(acs[acs['taker']==1]['wage12'],
                                                        np.arange(0, thre+binwidth, binwidth))).sum()
takers.columns = ['n_takers']
takers_cf = acs[acs['taker_cf']==1][['PWGTP_POW']].groupby(pd.cut(acs[acs['taker_cf']==1]['wage12'],
                                                        np.arange(0, thre+binwidth, binwidth))).sum()
takers = takers.join(takers_cf)
takers.columns = ['n_takers', 'n_takers_cf']
takers['g_takers'] = (takers['n_takers_cf'] - takers['n_takers']) / takers['n_takers'] * 100
# plot
# title = 'Percent increase in number of leave takers, by annual wage income group'
title = ''
ys = takers['g_takers'].values
xs = np.arange(len(ys))
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
plt.bar(xs, ys, color='wheat', alpha=1, edgecolor='black') #
#bar1 = ax.bar(ind - width / 2, ys, width, align='center', capsize=5, color='indianred', ecolor='grey')
ax.set_ylabel('Simulated percent increase in number of leave takers')
ax.set_xlabel('$ Annual Wage Income')
x_tick_labels = [str(int(x.left/1000)) + '-' + str(int(x.right/1000)) + 'k' for x in takers['g_takers'].index]
ax.set_xticks(xs)
ax.set_xticklabels(x_tick_labels)
ax.yaxis.grid(False)

rects = ax.patches
labels = [str(round(y, 1)) + '%' for y in ys]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')

format_chart(fig, ax, title,  bg_color='white', fg_color='k')
plt.savefig(fp_out + 'MD_CA_%s_takers_increase.png' % method, facecolor='white', edgecolor='white') #

## How much more leave taking would occur among low-wage workers?
# key vars are len_type and cfl_type, fillna as 0
for t in types:
     acs.loc[acs['len_%s' % t].isna(), 'len_%s' % t] = 0
     acs.loc[acs['cfl_%s' % t].isna(), 'cfl_%s' % t] = 0
# get increase in leave takers across leave types (len_type>0 pop est VS cfl_type>0 pop est)
# dict from leave type to leave taker counts
dct_taker = OrderedDict()
dct_taker['len'] = OrderedDict()
dct_taker['cfl'] = OrderedDict()
dct_taker['dcfl'] = OrderedDict()
for t in types:
    dct_taker['len'][t] = acs[(acs['low_wage']==1) & (acs['len_%s' % t]>0)]['PWGTP_POW'].sum()
    dct_taker['cfl'][t] = acs[(acs['low_wage']==1) & (acs['cfl_%s' % t]>0)]['PWGTP_POW'].sum()
    dct_taker['dcfl'][t] = 100* (dct_taker['cfl'][t] - dct_taker['len'][t]) / dct_taker['len'][t]
## Plot
# Number of leave takers (without program VS with program), by leave type
# title = 'Number of low-wage workers taking leaves, by leave reason'
title = ''
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ind = np.arange(len(types))
ys = {}
ys['len'] = dct_taker['len'].values()
ys['cfl'] = dct_taker['cfl'].values()
width = 0.4
bar_l = ax.bar(ind - width / 2, ys['len'] , width, align='center', capsize=5,
               color='lightgrey', alpha=0.8, edgecolor='black', label='Without Program')
bar_h = ax.bar(ind + width / 2, ys['cfl'] , width, align='center', capsize=5,
               color='indianred', alpha=0.8, edgecolor='black', label='With Program')
plt.legend()
ax.set_ylabel('Number of workers')
ax.set_xticks(ind)
ax.set_xticklabels(('Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent'))
ax.set_xlabel('Leave Reason')
ax.yaxis.grid(False)

rects = ax.patches
labels = [format(y, ',d') for y in ys['len']]
labels += [format(y, ',d') for y in ys['cfl']]
for i, kv in enumerate(zip(rects, labels)):
    rect, label = kv
    height = rect.get_height()
    if i<len(rects)/2:
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, fontsize=9,
            ha='center', va='bottom')
    else:
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, fontsize=9,
                ha='center', va='bottom')

format_chart(fig, ax, title, bg_color='white', fg_color='k')
plt.savefig(fp_out + 'MD_CA_%s_low_wage_taker_counts.png' % method, facecolor='white', edgecolor='white') #

# get increase in leave takers and leave length among low-wage leave takers for each leave type
# then get a dict from leave type to growth in leave taker counts and growth in lengths
for t in types:
     acs['dlen_%s' % t] = (acs['cfl_%s' % t] - acs['len_%s' % t])
ks = types
vs = [(acs[acs['low_wage']==1]['dlen_%s' % t]*acs[acs['low_wage']==1]['PWGTP_POW']).sum() /
      (acs[acs['low_wage']==1]['len_%s' % t]*acs[acs['low_wage']==1]['PWGTP_POW']).sum() for t in types]

vs = [100*v for v in vs]
dct_dlen = OrderedDict(zip(ks, vs))

# Length growth, by leave type
# title = 'Increase in leave lengths taken by low-wage workers, by leave reason

title = ''
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ind = np.arange(len(types))
ys = {}
ys['num'] = dct_taker['dcfl'].values()
ys['lth'] = dct_dlen.values()
width = 0.4
bar_num = ax.bar(ind - width / 2, ys['num'] , width, align='center', capsize=5,
               color='wheat', alpha=1, edgecolor='black', label='Increase in Number of Leave Takers')
bar_lth = ax.bar(ind + width / 2, ys['lth'] , width, align='center', capsize=5,
               color='darkgoldenrod', alpha=1, edgecolor='black', label='Increase in Aggregate Leave Length')
plt.legend()
ax.set_ylabel('Percent Increase')
ax.set_xticks(ind)
ax.set_xticklabels(('Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent'))
ax.set_xlabel('Leave Reason')
ax.yaxis.grid(False)

rects = ax.patches
labels = [str(round(y, 1)) + '%' for y in ys['num']]
labels += [str(round(y, 1)) + '%' for y in ys['lth']]
for i, kv in enumerate(zip(rects, labels)):
    rect, label = kv
    height = rect.get_height()
    if i<len(rects)/2:
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, fontsize=8,
            ha='center', va='bottom')
    else:
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, fontsize=8,
                ha='center', va='bottom')
format_chart(fig, ax, title, bg_color='white', fg_color='k')
plt.savefig(fp_out + 'MD_CA_%s_low_wage_takers_len_increase.png' % method, facecolor='white', edgecolor='white') #

###########################
title = ''
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ind = np.arange(len(types))
ys = dct_dlen.values()
width = 0.5
ax.bar(ind, ys, width, align='center', capsize=5, color='beige', edgecolor='black')
ax.set_ylabel('Percent increase')
ax.set_xticks(ind)
ax.set_xticklabels(('Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent'))
ax.set_xlabel('Leave Reason')
ax.yaxis.grid(False)

rects = ax.patches
labels = [str(round(y, 1)) + '%' for y in ys]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label, fontsize=9,
        ha='center', va='bottom')

format_chart(fig, ax, title, bg_color='white', fg_color='k')
plt.savefig(fp_out + 'MD_CA_%s_low_wage_len_increase.png' % method, facecolor='white', edgecolor='white') #



## How many low-wage workers would get benefits?
# low wage workers (wage12<=30k)
acs['low_wage'] = np.where(acs['wage12']<=30000, 1, 0)
# dict from leave type to total number of takeups (agg across takeup_type=1)
dct_bene = OrderedDict()
dct_bene['low_wage'] = OrderedDict()
dct_bene['high_wage'] = OrderedDict()
for t in types:
    dct_bene['low_wage'][t] = acs[(acs['low_wage']==1) & (acs['takeup_%s' % t]==1)]['PWGTP_POW'].sum()
    dct_bene['high_wage'][t] = acs[(acs['low_wage'] == 0) & (acs['takeup_%s' % t] == 1)]['PWGTP_POW'].sum()

## Plot
# Number of benes, by leave type
# title = 'Number of low-wage workers receiving program benefits, by leave reason'
title = ''
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ind = np.arange(len(types))
ys = {}
ys['low_wage'] = dct_bene['low_wage'].values()
ys['high_wage'] = dct_bene['high_wage'].values()
width = 0.4
bar_l = ax.bar(ind - width / 2, ys['low_wage'] , width, align='center', capsize=5,
               color='maroon', edgecolor='black', label='Annual Wage Earnings <= $30,000')
bar_h = ax.bar(ind + width / 2, ys['high_wage'] , width, align='center', capsize=5,
               color='darksalmon', edgecolor='black', label='Annual Wage Earnings > $30,000')
plt.legend()
ax.set_ylabel('Number of workers')
ax.set_xticks(ind)
ax.set_xticklabels(('Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent'))
ax.set_xlabel('Leave Reason')
ax.yaxis.grid(False)

rects = ax.patches
labels = [format(y, ',d') for y in ys['low_wage']]
labels += [format(y, ',d') for y in ys['high_wage']]
for i, kv in enumerate(zip(rects, labels)):
    rect, label = kv
    height = rect.get_height()
    if i<len(rects)/2:
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, fontsize=9,
            ha='center', va='bottom')
    else:
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, fontsize=9,
                ha='center', va='bottom')

format_chart(fig, ax, title, bg_color='white', fg_color='k')
plt.savefig(fp_out + 'MD_CA_%s_low_wage_bene_counts.png' % method, facecolor='white', edgecolor='white') #

# get total across leave reasons
print('total lower-wage worker count across reasons (incl. double count for multiple leavers) = %s' % sum(list(dct_bene['low_wage'].values())))
acs['takeup_any'] = [int(x>0) for x in np.nanmax(acs[['takeup_%s' % t for t in types]].values, axis=1)]
x = acs[(acs['low_wage']==1) & (acs['takeup_any']==1)]['PWGTP_POW'].sum()
n_takeup_any = acs[acs['takeup_any']==1]['PWGTP_POW'].sum()
print('total lower-wage worker count with program take up for any reason (no double count) = %s' % x)
print('share of lower-wage workers with program take up for any reason (no double count) = %s' % (x/n_takeup_any))
n_low_wage = acs[acs['low_wage']==1]['PWGTP_POW'].sum()
print('share of low-wage workers among all eligible = %s' % (n_low_wage / acs['PWGTP_POW'].sum()))

acs['takeup_matdis_bond'] = [int(x>0) for x in np.nanmax(acs[['takeup_%s' % t for t in ['matdis','bond']]].values, axis=1)]
x = acs[(acs['low_wage']==1) & (acs['takeup_matdis_bond']==1)]['PWGTP_POW'].sum()
n_takeup_matdis_bond = acs[acs['takeup_matdis_bond']==1]['PWGTP_POW'].sum()
print('total lower-wage worker count with program take up due to matdis/bond (no double count) = %s' % x)
print('share of lower-wage workers with program take up due to matdis/bond (no double count) = %s' % (x/n_takeup_matdis_bond))


print('total higher-wage worker count across reasons (incl. double count for multiple leavers) = %s' % sum(list(dct_bene['high_wage'].values())))
acs['takeup_any'] = [int(x>0) for x in np.nanmax(acs[['takeup_%s' % t for t in types]].values, axis=1)]
x = acs[(acs['low_wage']==0) & (acs['takeup_any']==1)]['PWGTP_POW'].sum()
print('total higher-wage worker count with program take up for any reason (no double count) = %s' % x)


## How much benefit would go to low-income worker families?
# set annual benefit = 0 for missing amount
for t in types:
     acs.loc[acs['annual_benefit_%s' % t].isna(), 'annual_benefit_%s' % t] = 0
acs['annual_benefit_all'] = [x.sum() for x in acs[['annual_benefit_%s' % t for t in types]].values]
## Plot
# Number of benes, by benefit amount
title = 'Number of low-wage worker recipients, by benefit levels'
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
xs = acs[(acs['low_wage']==1) & (acs['annual_benefit_all']>0)]['annual_benefit_all']
wts = acs[(acs['low_wage']==1) & (acs['annual_benefit_all']>0)]['PWGTP_POW']
binwidth = 1000
thre = acs[(acs['low_wage']==1) & (acs['annual_benefit_all']>0)]['annual_benefit_all'].max()
plt.hist(xs, weights=wts, bins=range(0, int(thre) + binwidth, binwidth), color='tan', edgecolor='black')
ax.set_xticks(range(0, int(thre) + binwidth, 2*binwidth))
ax.set_ylabel('Number of workers')
ax.set_xlabel('$ Benefits Received')
ax.yaxis.grid(False)
format_chart(fig, ax, title, bg_color='white', fg_color='k')
plt.savefig('C:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_3/results/'
            'MD_CA_logitPen_low_wage_benefit_amount.png', facecolor='white', edgecolor='white') #

#### check sum of cpl
cpls = np.array([np.nansum(x) for x in acs[acs['low_wage']==1][['cpl_%s' % t for t in types]].values])
cols = ['annual_benefit_%s' % t for t in types]
cols += ['annual_benefit_all', 'wage12', 'wkswork']
cols += ['cpl_%s' % t for t in types]
cols += ['takeup_%s' % t for t in types]
acs_taker_needer = acs[(acs['low_wage']==1) & (acs['annual_benefit_all']>50000)][cols]

for t in types:
    v = [min(x, 1144) for x in
         ((acs_taker_needer['wage12'] / acs_taker_needer['wkswork'] * 0.57))]
    #v = np.array(v)
    print('-------v for %s = \n%s' % (t, v))
    # get annual benefit for leave type t - sumprod of capped benefit, and takeup flag for each ACS row
    acs_taker_needer['annual_benefit_%s' % t] = (v * acs_taker_needer['cpl_%s' % t] / 5 *
                                                 acs_taker_needer['takeup_%s' % t])