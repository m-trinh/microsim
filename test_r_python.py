'''
compare results between R and Python

chris zhang 1/7/2020
'''

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np

## Read in data
dr = pd.read_csv("C:\workfiles\Microsimulation\microsim_R-master\output\_test_RI.csv")
dr0 = dr.copy()
dr = dr[dr['eligworker']==1]

p_run_stamp = '20200109_125429'
dp = pd.read_csv('./output/output_%s_Main/acs_sim_%s.csv' % (p_run_stamp, p_run_stamp))

## Check length of DFs
# get diff in eligible workers' IDs between dr and dp
# NOTE: SERIALNO is id for household not person.
ids_r = pd.DataFrame(dr['SERIALNO']).sort_values(by='SERIALNO')
ids_r['zr'] = 1
ids_r = ids_r.groupby(by='SERIALNO').count().reset_index()
ids_p = pd.DataFrame(dp['SERIALNO']).sort_values(by='SERIALNO')
ids_p['zp'] = 1
ids_p = ids_p.groupby(by='SERIALNO').count().reset_index()

# Case 1: find out which hh ids has persons in Python but none in R
ids_pr = pd.merge(ids_p, ids_r, how='left', on='SERIALNO', indicator=True)
ids_p_only = ids_pr[ids_pr['_merge']!='both']
# examine >> in dr, WAGP is below RI 3840 line

# Case 2: find out which hh ids has more persons in Python than in R (cond' R has any)
ids_pr['dz'] = (ids_pr['zp'] - ids_pr['zr'])
ids_p_more = ids_pr[ids_pr['dz']>0]['SERIALNO'].values
ids_p.loc[ids_p['SERIALNO'].isin(ids_p_more), ]
ids_r.loc[ids_p['SERIALNO'].isin(ids_p_more), ]
# examine >> in dr, WAGP is below RI 3840 line
# TODO: (Case 1 & 2)LP to apply ADJINC in ACS: d['wage12'] = d['WAGP'] * d['ADJINC'] / self.adjinc. This affects wage12 for elig

## Check take/need vars in DFs
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
for t in types:
    print(dr['take_%s' % t].value_counts())
    print(dp['take_%s' % t].value_counts())
    print('-----------------------------')
# TODO: logit gives very diff take vars for R/P.
# checked - sklearn logit regularization does not affect result. Set C=1e8 same diff issue

################################
# check for consistency between
# - Python sklearn logit newton-cg
# - R glm
################################
import sklearn.preprocessing, sklearn.linear_model, sklearn.naive_bayes, sklearn.neighbors, sklearn.tree, sklearn.ensemble, \
    sklearn.gaussian_process, sklearn.svm
## Read in data
import statsmodels.api as sm
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from collections import OrderedDict as od
d = pd.read_csv('./data/fmla_2012/fmla_clean_2012.csv')
# id
id = 'empid'
# Xs, ys, w
Xs = ['widowed', 'divorced', 'separated', 'nevermarried',
          'female', 'age','agesq',
          'ltHS', 'someCol', 'BA', 'GradSch',
          'black', 'other', 'asian','native','hisp',
          'nochildren','faminc','coveligd']
# a reduced Xs removing rare events
# Xs = ['divorced', 'nevermarried',
#           'female', 'age','agesq',
#           'someCol', 'BA', 'GradSch',
#           'black', 'hisp',
#           'nochildren','faminc','coveligd']
# Xs = ['female', 'black', 'faminc']
ys = ['take_own', 'take_matdis', 'take_bond', 'take_illchild', 'take_illspouse', 'take_illparent']
ys += ['need_own', 'need_matdis', 'need_bond', 'need_illchild', 'need_illspouse', 'need_illparent']
ys += ['resp_len']
w = 'weight'
# reduce df
d = d[[id] + Xs + ys+ [w]]

## Standardize
# use N-ddof = N-1 as in R
# https://stackoverflow.com/questions/27296387/difference-between-r-scale-and-sklearn-preprocessing-scale/27297618

# drop missing rows if any col is missing
d = d.dropna(how='any')
# standardize
for X in Xs:
    d['z_%s' % X] = (d[X] - d[X].mean()) / np.std(d[X], axis=0, ddof=1)

print(d[['empid', 'age','z_age', 'female', 'z_female']].head())

#d = d[:100]
## Fit model
# sklearn
solvers = ['newton-cg', 'liblinear']
bs = {} # dict from solver to coefs
use_standard_xvars = True
if use_standard_xvars:
    xvars = ['z_%s' % x for x in Xs]
else:
    xvars = Xs
for s in solvers:
    print('--- solver = %s -------------------------' % s)
    for y in ys[:1]:
        clf = sklearn.linear_model.LogisticRegression(solver=s).fit(d[xvars], d[y])
        # print coefs
        bs[s] = od(list(zip(xvars, [round(x, 6) for x in clf.coef_[0]])))
        print('const: ', clf.intercept_[0])
        for k, v in bs[s].items():
            print(k, ': ',v)
# TODO: newton-cg cannot converge with sample_weights!! Works fine if no weights
# TODO: after ddof=0 (default) standardization, newton-cg and liblinear gives very similar coefs (up to O(1e-5)). Can use liblinear as Python baseline
# TODO: both P/R need hard-coded standardization: (x-mu)/sigma where sigma divisor = (n-1). R psycho package does not affect binary. Python sklearn set divisor = n.
# TODO: after hardcoded stand', with no weighting, newton-cg~liblinear~statsmodels=glm, good.
# TODO: need to find P/R consistency with weights applied


# statsmodels
logit = sm.GLM(d[y], sm.add_constant(d[xvars]), family=sm.families.Binomial())
logit = logit.fit()
#print(logit.summary())
print('----------------------')
print(logit.params)
# check 0/1s
# for x in Xs:
#     print('--------------- x = %s --------------' % x)
#     print(d[x].value_counts().sort_index())



####################################

import pandas as pd
import statsmodels.api as sm

cuse = pd.read_table("http://data.princeton.edu/wws509/datasets/cuse.dat",
                     sep=" +")
res = sm.formula.glm("using + notUsing ~ C(age, Treatment('<25')) + "
                     "education + wantsMore",  family=sm.families.Binomial(),
                     data=cuse).fit()
res.summary()

######


res = sm.formula.glm("%s ~ %s" % (y, ' + '.join(xvars)),  family=sm.families.Binomial(),
                     data=d).fit()
res.summary()