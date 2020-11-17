'''
check IB1 results - why SVM outlay higher than other methods
chris zhang 11/10/2020
'''

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from _5a_aux_functions import *

dir_r = 'E:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_1/ib1_v4/ri results/'
fps_r = ['RI_Logistic Regression GLM_issue_brief_1_11052020.csv']
fps_r += ['RI_Logistic Regression Regularized_issue_brief_1_11052020.csv']
fps_r += ['RI_Random Forest_issue_brief_1_11052020.csv']
fps_r += ['RI_Ridge Classifier_issue_brief_1_11052020.csv']
fps_r += ['RI_Support Vector Machine_issue_brief_1_11052020.csv']
fps_r = [dir_r + x for x in fps_r]
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']

for fp_r in fps_r:
    header = '------------ %s ---------------' % fp_r.split('/')[-1].split('_')[1]
    print(header)
    dr = pd.read_csv(fp_r)
    # check wage12
    #print(dr[dr['takeup_any']==1]['wage12'].describe())
    # check cpl_all
    # dr['cpl_all'] = dr['cpl_all'].fillna(0)
    # print(dr[dr['takeup_any']==1]['cpl_all'].describe())
    # check len_all (status quo length)
    # dr['len_all'] = [x.sum() for x in dr[['len_' + x for x in types]].values]
    # dr['len_all'] = dr['len_all'].fillna(0)
    # print(dr[dr['takeup_any']==1]['len_all'].describe())
    # check resp_len
    print(dr[dr['takeup_any']==1]['resp_len'].describe())

###############################
# check confidence interval for eligible worker counts using post-sim ACS
# formula: https://usa.ipums.org/usa/repwt.shtml
###############################
dir_r = 'E:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_1/ib1_v4/svm no topoff results/'
fp_r = 'CA_Support Vector Machine_issue_brief_1_11052020.csv'
dr = pd.read_csv(dir_r+fp_r)

mean = dr.PWGTP.sum()
sesq = 0
for x in range(1, 81):
    sesq += 4/80 * (dr['PWGTP%s' % x].sum() - mean)**2
se = sesq**0.5
lower_bound = mean - 1.96*se
upper_bound = mean + 1.96*se
print(lower_bound)
print(mean)
print(upper_bound)