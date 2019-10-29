from _1_clean_FMLA import DataCleanerFMLA
from _4_clean_ACS import DataCleanerACS
from _5_simulation import SimulationEngine
from _5a_aux_functions import *
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
import bisect
import random
from time import time
import json

## Set up
st = 'ri'
yr = 16
fp_fmla_in = './data/fmla_2012/fmla_2012_employee_restrict_puf.csv'
fp_cps_in = './data/cps/CPS2014extract.csv'
fp_acsh_in = './data/acs/household_files'
fp_acsp_in = './data/acs/person_files'
acs_weight_multiplier = 1.0217029934467345 # see project acs_all
fp_fmla_out = './data/fmla_2012/fmla_clean_2012.csv'
fp_cps_out = './data/cps/cps_for_acs_sim.csv'
fp_acs_out = './data/acs/'
fp_length_distribution_out = './data/fmla_2012/length_distributions.json'
fps_in = [fp_fmla_in, fp_cps_in, fp_acsh_in, fp_acsp_in]
fps_out = [fp_fmla_out, fp_cps_out, fp_acs_out, fp_length_distribution_out]

# fullFp_acs, fullFp_fmla, fullFp_out = settings.acs_file, settings.fmla_file, settings.output_directory
# fp_fmla = '.'+fullFp_fmla[fullFp_fmla.find('/data/fmla_2012/'):]
# print(fp_fmla)
# fp_acs = '.'+fullFp_acs[fullFp_acs.find('/data/acs/'):]
# fp_out = fullFp_out
clf_name = 'Logistic Regression'

# prog_para
elig_wage12 = 11520
elig_wkswork = 1
elig_yrhours = 1
elig_empsize = 1
rrp = 0.5
wkbene_cap = 99999999

d_maxwk = {
    'own': 30,
    'matdis': 30,
    'bond': 4,
    'illchild': 4,
    'illspouse': 4,
    'illparent': 4
}

d_takeup = {
    'own': 0.25,
    'matdis': 0.25,
    'bond': 0.25,
    'illchild': 0.25,
    'illspouse': 0.25,
    'illparent': 0.25
}

incl_empgov_fed = True
incl_empgov_st = True
incl_empgov_loc = True
incl_empself = False
sim_method = 'Logistic Regression'

prog_para = [elig_wage12, elig_wkswork, elig_yrhours, elig_empsize, rrp, wkbene_cap, d_maxwk, d_takeup,
             incl_empgov_fed, incl_empgov_st, incl_empgov_loc, incl_empself, sim_method]

# initiate instances
se = SimulationEngine(st, yr, fps_in, fps_out, clf_name, prog_para, engine_type='Main')
# clean data
se.prepare_data()
# get_acs_simulated

# Read in cleaned ACS and FMLA data, and FMLA-based length distribution
acs = pd.read_csv(se.fp_acs_out + 'ACS_cleaned_forsimulation_20%s_%s.csv' % (se.yr, se.st))
pfl = 'non-PFL'  # status of PFL as of ACS sample period
d = pd.read_csv(se.fp_fmla_out, low_memory=False)
with open(se.fp_length_distribution_out) as f:
    flen = json.load(f)

# Define classifier
clf = se.d_clf[se.clf_name]

# Train models using FMLA, and simulate on ACS workers
t0 = time()
col_Xs, col_ys, col_w = get_columns()
X = d[col_Xs]
w = d[col_w]
Xa = acs[X.columns]

for c in col_ys:
    tt = time()
    y = d[c]
    acs = acs.join(get_sim_col(X, y, w, Xa, clf))
    print('Simulation of col %s done. Time elapsed = %s' % (c, (time()-tt)))
print('6+6+1 simulated. Time elapsed = %s' % (time()-t0))

# Post-simluation logic control
acs.loc[acs['male'] == 1, 'take_matdis'] = 0
acs.loc[acs['male'] == 1, 'need_matdis'] = 0
acs.loc[(acs['nevermarried'] == 1) | (acs['divorced'] == 1), 'take_illspouse'] = 0
acs.loc[(acs['nevermarried'] == 1) | (acs['divorced'] == 1), 'need_illspouse'] = 0
acs.loc[acs['nochildren'] == 1, 'take_bond'] = 0
acs.loc[acs['nochildren'] == 1, 'need_bond'] = 0
acs.loc[acs['nochildren'] == 1, 'take_matdis'] = 0
acs.loc[acs['nochildren'] == 1, 'need_matdis'] = 0

# Conditional simulation - anypay, doctor, hospital for taker/needer sample
acs['taker'] = [max(z) for z in acs[['take_%s' % t for t in se.types]].values]
acs['needer'] = [max(z) for z in acs[['need_%s' % t for t in se.types]].values]
X = d[(d['taker'] == 1) | (d['needer'] == 1)][col_Xs]
w = d.loc[X.index][col_w]
Xa = acs[(acs['taker'] == 1) | (acs['needer'] == 1)][X.columns]
if len(Xa) == 0:
    print('Warning: Both leave taker and leave needer do not present in simulated ACS persons. '
          'Simulation gives degenerate scenario of zero leaves for all workers.')
else:
    for c in ['anypay', 'doctor', 'hospital']:
        y = d.loc[X.index][c]
        acs = acs.join(get_sim_col(X, y, w, Xa, clf))
    # Post-simluation logic control
    acs.loc[acs['hospital'] == 1, 'doctor'] = 1

# Conditional simulation - prop_pay for anypay=1 sample
X = d[(d['anypay'] == 1) & (d['prop_pay'].notna())][col_Xs]
w = d.loc[X.index][col_w]
Xa = acs[acs['anypay'] == 1][X.columns]
# a dict from prop_pay int category to numerical prop_pay value
# int category used for phat 'p_0', etc. in get_sim_col
v = d.prop_pay.value_counts().sort_index().index
k = range(len(v))
d_prop = dict(zip(k, v))
D_prop = dict(zip(v, k))

if len(Xa) == 0:
    pass
else:
    y = d.loc[X.index]['prop_pay'].apply(lambda x: D_prop[x])
    yhat = get_sim_col(X, y, w, Xa, clf)
    # prop_pay labels are from 1 to 6, get_sim_col() vectorization sum gives 0~5, increase label by 1
    yhat = pd.Series(data=yhat.values + 1, index=yhat.index, name='prop_pay')
    acs = acs.join(yhat)
    acs.loc[acs['prop_pay'].notna(), 'prop_pay'] = [d_prop[x] for x in acs.loc[acs['prop_pay'].notna(), 'prop_pay']]

# Draw status-quo leave length for each type
# Without-program lengths - draw from FMLA-based distribution (pfl indicator = 0)
# note: here, cumsum/bisect is 20% faster than np/choice.
# But when simulate_wof applied as lambda to df, np/multinomial is 5X faster!
t0 = time()
for t in se.types:
    acs['len_%s' % t] = 0
    n_lensim = len(acs.loc[acs['take_%s' % t] == 1])  # number of acs workers who need length simulation
    # print(n_lensim)
    ps = [x[1] for x in flen[pfl][t]]  # prob vector of length of type t
    cs = np.cumsum(ps)
    lens = []  # initiate list of lengths
    for i in range(n_lensim):
        lens.append(flen[pfl][t][bisect.bisect(cs, np.random.random())][0])
    acs.loc[acs['take_%s' % t] == 1, 'len_%s' % t] = np.array(lens)
    # print('mean = %s' % acs['len_%s' % t].mean())
# print('te: sq length sim = %s' % (time()-t0))

# Max needed lengths (mnl) - draw from simulated without-program length distribution
# conditional on max length >= without-program length
T0 = time()
for t in se.types:
    t0 = time()
    acs['mnl_%s' % t] = 0
    # resp_len = 0 workers' mnl = status-quo length
    acs.loc[acs['resp_len'] == 0, 'mnl_%s' % t] = acs.loc[acs['resp_len'] == 0, 'len_%s' % t]
    # resp_len = 1 workers' mnl draw from length distribution conditional on new length > sq length
    dct_vw = {}  # dict from sq length to possible greater length value, and associated weight of worker who provides the length
    x_max = acs['len_%s' % t].max()
    for x in acs['len_%s' % t].value_counts().index:
        if x < x_max:
            dct_vw[x] = acs[(acs['len_%s' % t] > x)][['len_%s' % t, 'PWGTP']].groupby(by='len_%s' % t)[
                'PWGTP'].sum().reset_index()
            mx = len(acs[(acs['resp_len'] == 1) & (acs['len_%s' % t] == x)])
            vxs = np.random.choice(dct_vw[x]['len_%s' % t], mx, p=dct_vw[x]['PWGTP'] / dct_vw[x]['PWGTP'].sum())
            acs.loc[(acs['resp_len'] == 1) & (acs['len_%s' % t] == x), 'mnl_%s' % t] = vxs
        else:
            acs.loc[(acs['resp_len'] == 1) & (acs['len_%s' % t] == x), 'mnl_%s' % t] = x * 1.25
            # print('mean = %s. MNL sim done for type %s. telapse = %s' % (acs['mnl_%s' % t].mean(), t, (time()-t0)))

# logic control of mnl
acs.loc[acs['male'] == 1, 'mnl_matdis'] = 0
acs.loc[(acs['nevermarried'] == 1) | (acs['divorced'] == 1), 'mnl_illspouse'] = 0
acs.loc[acs['nochildren'] == 1, 'mnl_bond'] = 0
acs.loc[acs['nochildren'] == 1, 'mnl_matdis'] = 0

# print('All MNL sim done. TElapsed = %s' % (time()-T0))

# Compute program cost
# sample restriction
acs = acs.drop(acs[(acs['taker'] == 0) & (acs['needer'] == 0)].index)

if not se.incl_empgov_fed:
    acs = acs.drop(acs[acs['empgov_fed'] == 1].index)
if not se.incl_empgov_st:
    acs = acs.drop(acs[acs['empgov_st'] == 1].index)
if not se.incl_empgov_loc:
    acs = acs.drop(acs[acs['empgov_loc'] == 1].index)
if not se.incl_empself:
    acs = acs.drop(acs[(acs['COW'] == 6) | (acs['COW'] == 7)].index)

# program eligibility
acs['elig_prog'] = 0

elig_empsizebin = 0
if 1 <= se.elig_empsize < 10:
    elig_empsizebin = 1
elif 10 <= se.elig_empsize <= 49:
    elig_empsizebin = 2
elif 50 <= se.elig_empsize <= 99:
    elig_empsizebin = 3
elif 100 <= se.elig_empsize <= 499:
    elig_empsizebin = 4
elif 500 <= se.elig_empsize <= 999:
    elig_empsizebin = 5
elif se.elig_empsize >= 1000:
    elig_empsizebin = 6

acs.loc[(acs['wage12'] >= se.elig_wage12) &
        (acs['wkswork'] >= se.elig_wkswork) &
        (acs['wkswork'] * acs['wkhours'] >= se.elig_yrhours) &
        (acs['empsize'] >= elig_empsizebin), 'elig_prog'] = 1

# keep only eligible population
acs = acs.drop(acs[acs['elig_prog'] != 1].index)

# Given fraction of double receiver x, simulate double/single receiver status
# With state program:
# if anypay = 0, must be single receiver
# let %(anypay=0) = a, %single-receiver specified must satisfy (1-x) >= a, i.e. x <= (1-a)
s_no_emp_pay = acs[acs['anypay']==0]['PWGTP'].sum() / acs['PWGTP'].sum()
x = 0.6 # percentage of double receiver
x = min(1 - s_no_emp_pay, x) # cap x at (1 - %(anypay=0))
# simulate double receiver status
# we need x/(1-a) share of double receiver from (1-a) of all eligible workers who have anypay=1
acs['z'] = [random.random() for x in range(len(acs))]
acs['double_receiver'] = (acs['z']<x/(1-s_no_emp_pay)).astype(int)
acs.loc[acs['anypay']==0, 'double_receiver'] = 0
# treat each ACS row equally and check if post-sim weighted share = x
# using even a small state RI shows close to equality
s_double_receiver = acs[acs['double_receiver']==1]['PWGTP'].sum() / acs['PWGTP'].sum()
s_double_receiver = round(s_double_receiver, 2)
print('Specified share of double-receiver = %s. Post-sim weighted share = %s' % (x, s_double_receiver))

# Simulate counterfactual leave lengths (cf-len) for double receivers
# Given cf-len, get cp-len
for t in se.types:
    acs['cfl_%s' % t] = np.nan
    acs.loc[acs['double_receiver']==1, 'cfl_%s' % t] = \
    acs.loc[acs['double_receiver']==1, 'len_%s' % t] + (acs.loc[acs['double_receiver']==1, 'mnl_%s' % t] -
                                                     acs.loc[acs['double_receiver']==1, 'len_%s' % t]) \
                                                    * (0.5*rrp)/(1-0.5 * acs.loc[acs['double_receiver']==1, 'prop_pay'])
    # Get covered-by-program leave lengths (cp-len) for double receivers
    acs.loc[acs['double_receiver']==1, 'cpl_%s' % t] = acs.loc[acs['double_receiver']==1, 'cfl_%s' % t] * \
                                                    rrp / (rrp + acs.loc[acs['double_receiver']==1, 'prop_pay'])

# Simulate cf-len for single receivers
# Given cf-len, get cp-len
for t in se.types:
    # single receiver, rrp>rre. Assume will use state program benefit to replace employer benefit
    acs.loc[(acs['double_receiver']==0) & (acs['prop_pay'] < rrp), 'cfl_%s' % t] = \
    acs.loc[(acs['double_receiver']==0) & (acs['prop_pay'] < rrp), 'len_%s' % t] + \
    (acs.loc[(acs['double_receiver']==0) & (acs['prop_pay'] < rrp), 'mnl_%s' % t] -
     acs.loc[(acs['double_receiver']==0) & (acs['prop_pay'] < rrp), 'len_%s' % t]) *\
    (rrp-acs.loc[(acs['double_receiver']==0) & (acs['prop_pay'] < rrp), 'prop_pay'])/\
    (1-acs.loc[(acs['double_receiver']==0) & (acs['prop_pay'] < rrp), 'prop_pay'])
    # single receiver, rrp<=rre. Assume will not use any state program benefit
    # so still using same employer benefit as status-quo, thus cf-len = sq-len
    acs.loc[(acs['double_receiver'] == 0) & (acs['prop_pay'] >= rrp), 'cfl_%s' % t] = \
    acs.loc[(acs['double_receiver'] == 0) & (acs['prop_pay'] >= rrp), 'len_%s' % t]
    # Get covered-by-program leave lengths (cp-len) for single receivers
    # if rrp>rre, cp-len = cf-len
    acs.loc[(acs['double_receiver'] == 0) & (acs['prop_pay'] < rrp), 'cpl_%s' % t] = \
    acs.loc[(acs['double_receiver'] == 0) & (acs['prop_pay'] < rrp), 'cfl_%s' % t]
    # if rrp<=rre, cp-len = 0
    acs.loc[(acs['double_receiver'] == 0) & (acs['prop_pay'] >= rrp), 'cpl_%s' % t] = 0
    # set cp-len = 0 if missing
    acs.loc[acs['cpl_%s' % t].isna(), 'cpl_%s' % t] = 0

# Apply cap of coverage period (in weeks) to cpl_type (in days) for each leave type
for t in se.types:
    acs.loc[acs['cpl_%s' % t]>=0, 'cpl_%s' % t] = [min(x, 5*se.d_maxwk[t]) for x in acs.loc[acs['cpl_%s' % t]>=0, 'cpl_%s' % t]]

# Save ACS data after finishing simulation
acs.to_csv('%s/acs_sim_%s.csv' % (se.output_directory, se.out_id), index=False)
message = 'Leave lengths (status-quo/counterfactual/covered-by-program) simulated for 5-year ACS 20%s-20%s in state %s. Time needed = %s seconds' % (
(se.yr - 4), se.yr, se.st.upper(), round(time() - tsim, 0))
print(message)
self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 95})
self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
return acs



###########################################################################################
# Choice between employer and state benefits
#
#
#-----------------------------------------------------------------
# check simulated leave lengths
# for t in se.types:
#     print('--------- TYPE %s -----------' % t)
#     print(acs.loc[acs['len_%s' % t]>0, 'len_%s' % t].mean())
#     print(acs.loc[acs['len_%s' % t]>0, 'len_%s' % t].median())
#     print(acs.loc[acs['len_%s' % t]>0, 'len_%s' % t].min())
#     print(acs.loc[acs['len_%s' % t]>0, 'len_%s' % t].max())
#-----------------------------------------------------------------
#
#
# DONE - 1. A fraction (up to 100%) of workers can and choose to receive both employer/state benefits simultaneously or sequentially
# DONE - 2. For double receivers, at status-quo, rr = (rre, 0), leave length = len_type
# At full replacement, rr = (1, 1), leave length = mnl_type
# At program scenario, rr = (rre, rrp), leave length = cfl_type which is linearly interpolated between
# the normed version of [rre, 2] = [rre/2, 1] for mid-point (rre/2 + rrp/2)
# DONE - 3. For single receivers, at status-quo, rr = rre, len = len_type.
# At full replacement, rr = 1, len = mnl_type
# At program scenario, rr = rrp. If rrp<=rre, len = len_type (use no program). If rrp > rre, len = cfl_prog_type, which
# is linearly interpolated between [rre, 1] for mid-point rrp.
# DONE - 4. For double receivers, cfl_type then needs to be distributed between employer and state benefits.
# We distribute in proportion to rre/rrp.
# DONE - 5. The resulting cfl_prog_type will then be subject to max period cap,
# TODO 6. (do it in get_cost())  rrp * cfl_prog_type will be subject to $ cap.
###########################################################################################

# Save ACS data after finishing simulation
acs.to_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id), index=False)
message = 'Leaves simulated for 5-year ACS 20%s-20%s in state %s. Time needed = %s seconds' % (
(self.yr - 4), self.yr, self.st.upper(), round(time() - tsim, 0))
print(message)
self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 95})
self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
return acs

############# GET COST

## def get_cost(self):
# read simulated ACS
# acs = pd.read_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id))
# apply take up rates and weekly benefit cap, and compute total cost, 6 types
costs = {}
for t in se.types:
    # v = capped weekly benefit of leave type
    v = [min(x, se.wkbene_cap) for x in
         ((acs['cpl_%s' % t] / 5) * (acs['wage12'] / acs['wkswork'] * se.rrp))]
    # w = population that take up benefit of leave type
    # d_takeup[t] is 'official' takeup rate = pop take / pop eligible
    # so pop take = total pop * official take up
    # takeup normalization factor = pop take / pop of ACS rows where take up occurs for leave type
    takeup_factor = acs['PWGTP'].sum() * se.d_takeup[t] / acs[acs['cpl_%s' % t]>0]['PWGTP'].sum()
    w = acs['PWGTP'] * takeup_factor
    costs[t] = (v * w).sum()
costs['total'] = sum(list(costs.values()))

# compute standard error using replication weights, then compute confidence interval
sesq = dict(zip(costs.keys(), [0] * len(costs.keys())))
for wt in ['PWGTP%s' % x for x in range(1, 81)]:
    costs_rep = {}
    for t in se.types:
        v = [min(x, se.wkbene_cap) for x in
             ((acs['cpl_%s' % t] / 5) * (acs['wage12'] / acs['wkswork'] * se.rrp))]
        takeup_factor = acs[wt].sum() * se.d_takeup[t] / acs[acs['cpl_%s' % t] > 0][wt].sum()
        w = acs[wt] * takeup_factor
        costs_rep[t] = (v * w).sum()
    costs_rep['total'] = sum(list(costs_rep.values()))
    for k in costs_rep.keys():
        sesq[k] += 4 / 80 * (costs[k] - costs_rep[k]) ** 2
for k, v in sesq.items():
    sesq[k] = v ** 0.5
ci = {}
for k, v in sesq.items():
    ci[k] = (costs[k] - 1.96 * sesq[k], costs[k] + 1.96 * sesq[k])

# Save output
out_costs = pd.DataFrame.from_dict(costs, orient='index')
out_costs = out_costs.reset_index()
out_costs.columns = ['type', 'cost']

out_ci = pd.DataFrame.from_dict(ci, orient='index')
out_ci = out_ci.reset_index()
out_ci.columns = ['type', 'ci_lower', 'ci_upper']

out = pd.merge(out_costs, out_ci, how='left', on='type')

d_tix = {'own': 1, 'matdis': 2, 'bond': 3, 'illchild': 4, 'illspouse': 5, 'illparent': 6, 'total': 7}
out['tix'] = out['type'].apply(lambda x: d_tix[x])
out = out.sort_values(by='tix')
del out['tix']

out.to_csv('%s/program_cost_%s_%s.csv' % (se.output_directory, se.st, se.out_id), index=False)

message = 'Output saved. Total cost = $%s million 2012 dollars' % (round(costs['total'] / 1000000, 1))
print(message)
se.__put_queue({'type': 'progress', 'engine': se.engine_type, 'value': 100})
se.__put_queue({'type': 'message', 'engine': se.engine_type, 'value': message})
return out  # df of leave type specific costs and total cost, along with ci's