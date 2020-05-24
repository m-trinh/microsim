for t in types:
    print('value counts for need_%s\n' % t, d['need_%s' % t].value_counts())

import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)

X = [[1, 0, np.nan], [3, 4, 0], [np.nan, 6,0], [8, 8, 1]]
X = pd.DataFrame(X)
imputer = KNNImputer(n_neighbors=2)
X = pd.DataFrame(imputer.fit_transform(X))



from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from _5a_aux_functions import *

d = pd.read_csv('./data/fmla/fmla_2012/fmla_clean_2012.csv')
d_raw = d.copy()
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
col_Xs, col_ys, col_w = get_columns(types)
d = d[col_Xs + col_ys + [col_w]]
X, ys = d[col_Xs], d[col_ys]
y = ys[ys.columns[0]]

idxs = y[y.notna()].index
X, y = X.loc[idxs, ], y.loc[idxs, ]

# split data into train and test sets
seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model no training data
w_train = d.loc[y_train.index, col_w]
model = XGBClassifier()
model.fit(X_train, y_train, sample_weight=w_train)
print(model)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
sample_weight = d.loc[y_test.index, col_w]
accuracy = accuracy_score(y_test, predictions, sample_weight=sample_weight)
precision = precision_score(y_test, predictions, sample_weight=sample_weight)
recall = recall_score(y_test, predictions, sample_weight=sample_weight)
f1 = f1_score(y_test, predictions, sample_weight=sample_weight)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))
print("F1 score: %.2f%%" % (f1 * 100.0))

#####
import numpy as np
random_seed = 12345
random_state = np.random.RandomState(random_seed)
X, y, w = X_train, y_train, w_train
Xa = X_test
clf = XGBClassifier()
pp = get_sim_col(X, y, w, Xa, clf, random_state)

#################################################
# check standard error of cost by type (NJ got large SE interval with negative lower bound)
#################################################
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
from _5a_aux_functions import get_acs_with_takeup_flags, get_weighted_draws
## Read in post-sim data
st = 'nj'
tag = '20200523_125352'
acs = pd.read_csv('./output/output_%s_main simulation/acs_sim_%s_%s.csv' % (tag, st, tag))
## params
random_seed = 12345
random_state = np.random.RandomState(random_seed)
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
params={
    'leave_types': types,
    'min_takeup_cpl': 5,
    'alpha': 0,
    'd_takeup': dict(zip(types, [0.0219, 0.0077, 0.0081, 0.0005, 0.0005, 0.0006])),
    'wkbene_cap': 594
}
pow_pop_multiplier = 1.02

out_total = None
acs_taker_needer = acs[(acs['taker'] == 1) | (acs['needer'] == 1)]
acs_neither_taker_needer = acs.drop(acs_taker_needer.index)

# apply take up flag and weekly benefit cap, and compute total cost, 6 types
costs = {}
for t in params['leave_types']:
    # v = capped weekly benefit of leave type
    v = [min(x, params['wkbene_cap']) for x in
         ((acs_taker_needer['wage12'] / acs_taker_needer['wkswork'] * acs_taker_needer['effective_rrp']))]
    # inflate weight for missing POW
    w = acs_taker_needer['PWGTP'] * pow_pop_multiplier

    # get program cost for leave type t - sumprod of capped benefit, weight, and takeup flag for each ACS row
    costs[t] = (v * acs_taker_needer['cpl_%s' % t] / 5 * w * acs_taker_needer['takeup_%s' % t]).sum()
costs['total'] = sum(list(costs.values()))

# compute standard error using replication weights, then compute confidence interval (lower bound at 0)
sesq = dict(zip(costs.keys(), [0] * len(costs.keys())))
wt_ixs = list(range(1, 2))
for wt in ['PWGTP%s' % x for x in wt_ixs]:
    # initialize costs dict for current rep weight
    costs_rep = {}
    # get takeup_type flags for acs under current rep weight
    acs = get_acs_with_takeup_flags(acs_taker_needer, acs_neither_taker_needer, wt, params, random_state)

    acs_taker_needer = acs[(acs['taker'] == 1) | (acs['needer'] == 1)]

    for t in params['leave_types']:
        v = [min(x, params['wkbene_cap']) for x in
             ((acs_taker_needer['wage12'] / acs_taker_needer['wkswork'] * acs_taker_needer['effective_rrp']))]
        # inflate weight for missing POW
        w = acs_taker_needer[wt] * pow_pop_multiplier

        # get program cost for leave type t - sumprod of capped benefit, weight, and takeup flag for each ACS
        # row
        costs_rep[t] = (v * acs_taker_needer['cpl_%s' % t] / 5 * w * acs_taker_needer['takeup_%s' % t]).sum()
    costs_rep['total'] = sum(list(costs_rep.values()))
    for k in costs_rep.keys():
        sesq[k] += 4 / len(wt_ixs) * (costs[k] - costs_rep[k]) ** 2

for k, v in sesq.items():
    sesq[k] = v ** 0.5
ci = {}
for k, v in sesq.items():
    ci[k] = (max(costs[k] - 1.96 * sesq[k], 0), costs[k] + 1.96 * sesq[k])

# Save output
out_costs = pd.DataFrame.from_dict(costs, orient='index')
out_costs = out_costs.reset_index()
out_costs.columns = ['type', 'cost']

out_ci = pd.DataFrame.from_dict(ci, orient='index')
out_ci = out_ci.reset_index()
out_ci.columns = ['type', 'ci_lower', 'ci_upper']

out = pd.merge(out_costs, out_ci, how='left', on='type')

d_tix = {'own': 1, 'matdis': 2, 'bond': 3, 'illchild': 4, 'illspouse': 5, 'illparent': 6, 'total': 7}
out['tix'] = [d_tix[x] for x in out['type']]

out = out.sort_values(by='tix')
del out['tix']

if out_total is not None:
    out_total[['cost', 'ci_lower', 'ci_upper']] += out[['cost', 'ci_lower', 'ci_upper']]
else:
    out_total = out












