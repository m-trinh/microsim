'''
get marginal probabilities estimated from FMLA data

Chris Zhang 10/23/2018
'''
import pandas as pd
import numpy as np
import json
import collections

d = pd.read_csv('./data/fmla_2012/fmla_clean_2012_resp_length.csv')
types = list(d['type_recent'].value_counts().index)
# -------------
# Marginal prob of number of leaves, conditional multiple leaver=1, and type_recent
# -------------
dmp = {}
# For taker/dual, use A4a_CAT
for t in types:
    nums = np.array([sum(d[(d['taker']==1) & (d['multiple']==1) & (d['type_recent']=='%s' % t) & (d['A4a_CAT']==x)]['weight']) for x in range(2, 7)])
    ps = nums/sum(nums)
    dmp['taker_' + t] = ps
# For needers only, use B5_CAT
for t in types:
    nums = np.array([sum(d[(d['needer']==1) & (d['multiple']==1) & (d['type_recent']=='%s' % t) & (d['B5_CAT']==x)]['weight']) for x in range(2, 7)])
    ps = nums/sum(nums)
    dmp['needer_' + t] = ps

dmp = pd.DataFrame.from_dict(dmp,orient='index')
dmp.columns = ['multi_%s' % x for x in range(2,7)]
dmp = dmp.sort_index()
dmp.to_csv('./data/fmla_2012/marginal_probs_numleaves.csv', index=True)

# -------------
# Conditional prob of other leave types of another leave if another leave exists, given a known leave type
# These conditional probs will be applied recursively to simulate leave types for multiple leavers
# e.g. given type_recent=matdis, nl =2, need to simulate 1 type from the other 5 types << NT know pr(type | matdis)
# -------------
dm = d[(d['multiple']==1) & (d['type_recent'].notna()) & (d['A20']==2) & (d['A5_1_CAT'].notna())]
# for A20 = 2 takers, type_recent is from loop 2 info, so need loop 1 (longest) type
dm['type_long'] = np.nan
dm.loc[dm['A5_1_CAT']==1, 'type_long'] = 'own'
dm.loc[dm['A5_1_CAT']==11, 'type_long'] = 'illchild'
dm.loc[dm['A5_1_CAT']==12, 'type_long'] = 'illspouse'
dm.loc[dm['A5_1_CAT']==13, 'type_long'] = 'illparent'
dm.loc[(dm['A5_1_CAT']==21) & (dm['take_matdis']==1), 'type_long'] = 'matdis'
dm.loc[(dm['A5_1_CAT']==21) & (dm['take_matdis']!=1) & (dm['take_bond']==1), 'type_long'] = 'bond'
dm = dm[dm['type_long'].notna()]

dcp = {}
for t in types:
    nums = np.array([sum(dm[(dm['type_recent']==t) & (dm['type_long']==x)]['weight']) for x in types])
    ps = nums/sum(nums)
    for i in range(len(ps)):
        if ps[i]==0:
            ps[i] = 0.001 # assign positive small prob to avoid zero conditional probs, which may cause
                          # 'no-further-simulation' issue when recursively simulating multiple leave types
    dcp[t] = ps
# In this small subsample, there is no type_recent = bond
# Assign equal conditional probs for pr(type|bond) in case bond is type of x-th leave and need to simulate (x+1)-th
dcp['bond'] = np.array([1/6]*6)
# Normalize to make sure probs are conditional on OTHER types, prob(next leave type = t | current type = t)=0 for all t
for type, ps in dcp.items():
    i = types.index(type)
    ps[i] = 0
    ps = ps/ps.sum()
    dcp[type] = ps
    i+=1
dcp = pd.DataFrame.from_dict(dcp,orient='index')
dcp.columns = [t for t in types]
dcp = dcp.sort_index()
dcp = dcp.sort_index(axis=1)
dcp.to_csv('./data/fmla_2012/conditional_probs_leavetypes.csv', index=True)

# -------------
# Estimate distribution of most recent leave lengths, using var 'length' defined from FMLA cleaning.
# Estimate using taker sample (taker==1), the only group 'length' is defined and positive
# Also need type_recent non-missing
# Estimate distribution for all leave types
# -------------
# Initiate a dict from type to type-specific dict of length/probs
dlen = {}
for t in types:
    d.loc[d['length'].notna(),'length'] = d.loc[d['length'].notna(),'length'].apply(lambda x: int(round(x, 1)))
    lengths = d[(d['taker']==1) & (d['type_recent']==t)].length.value_counts().sort_index().index
    lengths = [x for x in list(lengths)]
    pops = np.array([sum(d[(d['taker']==1) & (d['type_recent']==t) & (d['length']==x)] ['weight']) for x in lengths])
    ps = pops/pops.sum()
    dlen_t = dict(zip(lengths, ps))
    dlen_t = {int(k): v for k, v in dlen_t.items()}
    dlen[t] = dlen_t
with open('./data/fmla_2012/length_distributions.json', 'w') as fp:
    json.dump(dlen, fp, sort_keys=True)

