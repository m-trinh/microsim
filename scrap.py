for t in types:
    print('value counts for need_%s\n' % t, d['need_%s' % t].value_counts())

import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd
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



