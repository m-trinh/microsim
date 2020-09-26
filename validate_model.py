'''
model validation for Issue Brief 2 (Summary of Model Testing Memo)

chris zhang 9/24/2020
'''

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
from validate_model_functions import *

# fmla_wave - this affects xvars used for CV
fmla_wave = 2018

# Set up local directories
# file path to pre-processed clean FMLA file
fp_in_fmla = './data/fmla/fmla_%s/fmla_clean_%s.csv' % (fmla_wave, fmla_wave)
# directory to store figures plotted and cross-validation results (CSV files)
dir_out = 'E:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_2/results/ib2_v3_results/'
# directory to store output folders created by simulation from GUI. Folders should be renamed as 'ca_logit_glm' etc.
dir_sim_out = './output/_ib2_v3/'

#------------------------------------
# Internal: k-fold Cross Validation
#------------------------------------

# Get cross validation results based on number of folds and list of seeds, store results in folders
d = pd.read_csv(fp_in_fmla)
fold = 10  # cannot be more than # minor cases in most imbalanced outvar,
            # cannot be too small (say 2) which leads to small N_train and cause collinearity
n_seeds = 10
seeds = list(range(12345, 12345 + n_seeds))

clf_profile = get_cv_results(d, fmla_wave, fold, seeds, dir_out) # clfs, clf_class_names, dct_clf
clfs, clf_class_names_plot, dct_clf = clf_profile
# get avg_out_pop
avg_out_p = get_avg_out_pop(dir_out, seeds, true_numbers=True)
avg_out_p = avg_out_p[[x for x in avg_out_p.columns if x!='true'] + ['true']]
# get avg_out_ind
avg_out_i = get_avg_out_ind(dir_out, seeds)

# Plot with average results
# Pop level results - worker counts
suffix = '_k%s' % fold
savefig = (dir_out, suffix)
plot_pop_level_worker_counts(avg_out_p, clf_class_names_plot, clf_profile, add_title=False, savefig=savefig, figsize=(9, 7.5))
# Pop level results - leave counts
plot_pop_level_leave_counts(avg_out_p, clf_class_names_plot, clf_profile, add_title=False, savefig=savefig, figsize=(9, 7.5))
# Ind level results
dir_out = 'E:/workfiles/Microsimulation/draft/issue_briefs/issue_brief_2/ib2_v3_working/'
for yvar in ['taker', 'needer', 'resp_len'] + ['take_own', 'need_own', 'take_matdis', 'need_matdis']:
    suffix = '_%s_k%s_%s' % (fmla_wave, fold, yvar)
    savefig = (dir_out, suffix)
    plot_ind_level(avg_out_i, clf_class_names_plot, clf_profile, yvar, savefig=savefig, figsize=(9, 7.5))

#------------------------------------
# External: Compare simulated outlays
# against actual program outlays
#
# - must first run 24 simulations (3 states X 8 simulation methods),
#  and name output folders properly (e.g. ca_logit_glm)
#------------------------------------

# get costs df by states and methods
sts = ['ri', 'nj', 'ca']
methods = ['logit_glm', 'logit_reg', 'knn', 'nb', 'rf', 'xgb', 'ridge', 'svc']
costs = get_sim_costs(dir_sim_out, sts, methods)
# plot simulated costs by state and methods
plot_sim_costs(sts, costs, add_title=False, savefig=dir_out, figsize=(9, 7.5))

# get wage12 of uptakers for all sim methods, given state and leave type (CA, own)
st = 'ca' # can also set to 'nj', 'ri'
t = 'own' # can also set to 'matdis', 'bond', 'illchild', 'illspouse', 'illparent'
methods = ['logit_glm', 'logit_reg', 'knn', 'nb', 'rf', 'xgb', 'ridge', 'svc']
wage_pcts = get_wage_pcts(dir_sim_out, st, t, methods)
# plot wage percentiles, given state and leave type
plot_wage_pcts(wage_pcts, st, t, methods, add_title=False, savefig=dir_out, figsize=(9, 7.5))

######################################## END #############################################

