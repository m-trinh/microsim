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
fp_acsh_in = './data/acs/pow_household_files'
fp_acsp_in = './data/acs/pow_person_files'
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
needers_fully_participate = False
state_of_work = True
clone_factor = 1
dual_receivers_share = 0.6

prog_para = [elig_wage12, elig_wkswork, elig_yrhours, elig_empsize, rrp, wkbene_cap, d_maxwk, d_takeup,
             incl_empgov_fed, incl_empgov_st, incl_empgov_loc, incl_empself, sim_method, needers_fully_participate,
             state_of_work, clone_factor, dual_receivers_share]

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

#########################################
# check post-sim ACS against state data
# RI rules: http://www.dlt.ri.gov/tdi/tdifaqs.htm
# RI data: http://www.dlt.ri.gov/lmi/pdf/tdi/2018.pdf (earlier years URL 201X available)
#########################################
# read in post-sim acs
tags = {}
tags['logit'] = '20191125_161045'
# tags['ridge'] = '20191114_152939'
# tags['knn'] = '20191114_153443'
# tags['nb'] = '20191114_153643'
# tags['svm'] = '20191114_153825'
# tags['rf'] = '20191114_154830'

types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
takeups=dict(zip(types, [0.043, 0.033, 0.014, 0.001, 0.002, 0.001])) # using 2018 RI data / ACS eligible pop


for k, v in tags.items():
    print('--- BELOW ARE FOR sim_method = %s -----------------------------------' % k)
    acs = pd.read_csv('./output/output_%s_Main/acs_sim_%s.csv' % (v, v))
    # get average leave lengths
    for t in types:
        avg_len = (np.average(np.array(acs[acs['cpl_%s' % t] > 0]['cpl_%s' % t]), weights=acs[acs['cpl_%s' % t] > 0]['PWGTP']))
        avg_len = round(avg_len, 3)
        print('Average cp-len in days for type %s = %s'
              % (t, avg_len))
    for t in types:
        acs.loc[acs['cpl_%s' % t].isna(), 'cpl_%s' % t] = 0
        avg_len = (np.average(np.array(acs['cpl_%s' % t]), weights=acs['PWGTP']))
        avg_len = round(avg_len, 3)
        print('Average UNCOND\' cp-len in days for type %s = %s'
              % (t, avg_len))
# sumstats for mn-len
for k, v in tags.items():
    print('--- BELOW ARE FOR sim_method = %s -----------------------------------' % k)
    acs = pd.read_csv('./output/output_%s_Main/acs_sim_%s.csv' % (v, v))
    # get average leave lengths
    for t in types:
        avg_len = (
        np.average(np.array(acs[acs['mnl_%s' % t] > 0]['mnl_%s' % t]), weights=acs[acs['mnl_%s' % t] > 0]['PWGTP']))
        avg_len = round(avg_len, 3)
        print('Average mn-len in days for type %s = %s'
              % (t, avg_len))
    for t in types:
        acs.loc[acs['mnl_%s' % t].isna(), 'mnl_%s' % t] = 0
        avg_len = (np.average(np.array(acs['mnl_%s' % t]), weights=acs['PWGTP']))
        avg_len = round(avg_len, 3)
        print('Average UNCOND\' mn-len in days for type %s = %s'
              % (t, avg_len))

    # get total population with cp-len>0 and takeup_type=1 for each leave type
    for t in types:
        dx = acs[(acs['takeup_%s' % t]==1)]['PWGTP'].sum()
        #print('Pop with cp-len >0 and takeup=1 for type %s = %s' % (t, dx))
        s_positive_cpl = acs[acs['cpl_%s' % t] > 0]['PWGTP'].sum() / acs['PWGTP'].sum()
        #print('s_positive_cpl = %s' % round(s_positive_cpl, 4))
        takeup = takeups[t]
        takeup = min(s_positive_cpl, takeup)
        p_draw = takeup / s_positive_cpl
        target = sum(acs[acs['cpl_%s' % t] > 0]['PWGTP']) * p_draw

        #print('acs shape =\n', acs.shape)
        print('sum(acs[acs[\'cpl_s\'] > 0][\'PWGTP\']) = %s' % (sum(acs[acs['cpl_%s' % t] > 0]['PWGTP'])))
        #print('acs[\'PWGTP\'].sum() = %s' % (acs['PWGTP'].sum()))
        print('s_positive_cpl = %s' % s_positive_cpl)
        print('takeup = %s' % takeup)
        print('p_draw = %s' % round(p_draw, 4))
        # target = sum(acs[acs['cpl_%s' % t] > 0]['PWGTP']) * takeup * acs['PWGTP'].sum() / acs[acs['cpl_%s' % t] > 0]['PWGTP'].sum()
        print('target pop = %s' % int(target))
        # print('Age distribution: (min, mean, max) = (%s, %s, %s)' %
        #       (acs[acs['cpl_%s' % t]>0].age.min(), acs[acs['cpl_%s' % t]>0].age.mean(), acs[acs['cpl_%s' % t]>0].age.max()))

    # get number of takers/needers
    for t in types:
        n_taker = acs[acs['take_%s' % t]==1]['PWGTP'].sum()
        n_needer = acs[acs['need_%s' % t]==1]['PWGTP'].sum()
        print('[%s] Taker/Needer pops = (%s, %s)' % (t, n_taker, n_needer))


    # get total population with cp-len>0 for each leave type
    for t in types:
        dx = acs[(acs['cpl_%s' % t] > 0)]['PWGTP'].sum()
        print('Pop with cp-len >0 for type %s = %s' % (t, dx))



# get average payment per week
# apply weekly benefit bounds
# apply child dependency add-up (up to 5 children, max(10, 7% bene rate), add-up until rrp=1)
wkbene_bounds = [98, 867]
rrp = 0.6
acs['rrp'] = rrp
acs.loc[acs['ndep_kid']>0, 'rrp'] = [min(1, rrp + 0.07*min(x, 5)) for x in acs.loc[acs['ndep_kid']>0, 'ndep_kid']]
for t in types:
    v = [max(min(x, wkbene_bounds[1]), min(x, wkbene_bounds[0])) for x in
         ((acs['cpl_%s' % t] / 5) * (acs['wage12'] / acs['wkswork'] * rrp))]
    acs['wkbene_%s' % t] = v
    print('Average weekly benefit for type %s = %s ' % (t, acs[acs['wkbene_%s' % t]>0]['wkbene_%s' % t].mean()))


# get average leave length in weeks
for t in types:
    print('Average cp-len in weeks for type %s = %s' % (t, (acs[acs['cpl_%s' % t]>0]['cpl_%s' % t] / 5).mean()))

for t in types:
    print('Average mn-len in weeks for type %s = %s' % (t, (acs[acs['mnl_%s' % t]>0]['mnl_%s' % t] / 5).mean()))

# get share and level of pop with positive cp-len
for t in types:
    num = acs[acs['cpl_%s' % t]>0]['PWGTP'].sum()
    s_pos_cpl =  num / acs['PWGTP'].sum()
    print('Share of pop with positive cpl_%s = %s. Level = %s' % (t, round(s_pos_cpl, 2), num))

# get total program payment to recipients, by leave type
cost = 0
for t in types:
    cost_type = (acs['wkbene_%s' % t] * acs['cpl_%s' % t] / 5 * acs['takeup_%s' % t] * acs['PWGTP']).sum()/10**6
    print('Annual cost of type %s = %s' % (t, round(cost_type, 1)))
    cost += cost_type
print('Total cost = %s' % cost)

# get total approved claims based on takeup_type vars
n_claims = 0
for t in types:
    n_claims_type = acs[(acs['cpl_%s' % t]>0) & (acs['takeup_%s' % t]==1)]['PWGTP'].sum()
    print('Total program claims for type %s = %s' % (t, n_claims_type))
    n_claims += n_claims_type
print('Total claims = %s' % n_claims)

# get relative cases of own and matdis using FMLA data
# in principle we restrict to workers receiving state pay to compute this ratio
# but recStatePay=1 only gives 96 rows in FMLA, with 6 take_matdis=1 and 64 take_own=1
# using recStatePay=0/1 to compute the ratio - get similar ratio, so perhaps own/matdis constrained by no-program similarly
d = pd.read_csv('./data/fmla_2012/fmla_clean_2012.csv')
n_take_own = d[(d['recStatePay']==1) & (d['take_own']==1)]['weight'].sum()
n_take_matdis = d[(d['recStatePay']==1) & (d['take_matdis']==1)]['weight'].sum()
print('Proportion of take_own in own + matdis = %s' % round(100*n_take_own/ (n_take_own + n_take_matdis), 1))
'''
RI sim results -

Share of pop with positive cpl_own = 0.59. Level = 67411
Share of pop with positive cpl_matdis = 0.11. Level = 12544
Share of pop with positive cpl_bond = 0.13. Level = 15235
Share of pop with positive cpl_illchild = 0.15. Level = 16854
Share of pop with positive cpl_illspouse = 0.11. Level = 13166
Share of pop with positive cpl_illparent = 0.18. Level = 20394


RI 2018 data (approved cases only, excluding lending cases):
http://www.dlt.ri.gov/lmi/pdf/tdi/2018.pdf

claims for own = 29000 * 0.8 = 23200 (0.8 estimated using FMLA data)
claims for matdis = 29000 - 23200 = 5800
--- subtotal (own + matdis claims) = 29000 (own condition, 30 week coverage)
claims for bond = 5250
claims for illchild = 200 (spouse can help out)
claims for illspouse = 750 (spouse ill)
claims for illparent = 375 (spouse can help out)

so for testing set following take up rates (=n_claims / total eligible pop)
these numbers are independent from sim results since both n_claims and eligible pop are just data

[round(x, 3) for x in np.array([23200, 5800, 5250, 200, 750, 375])/383712]


vs = [0.06, 0.015, 0.014, 0.001, 0.002, 0.001]
takeups = dict(zip(types, vs))

'''


### check R results
for k, v in tags.items():
    print('--- R RESULTS ---')
    print('--- BELOW ARE FOR sim_method = %s -----------------------------------' % k)
    acs = pd.read_csv('./from_luke/RI_sim.csv')
    # get average leave lengths
    for t in types:
        avg_len = (np.average(np.array(acs[acs['mnl_%s' % t] > 0]['mnl_%s' % t]), weights=acs[acs['mnl_%s' % t] > 0]['PWGTP']))
        avg_len = round(avg_len, 3)
        print('Average mn-len in days for type %s = %s'
              % (t, avg_len))
    for t in types:
        acs.loc[acs['mnl_%s' % t].isna(), 'mnl_%s' % t] = 0
        avg_len = (np.average(np.array(acs['mnl_%s' % t]), weights=acs['PWGTP']))
        avg_len = round(avg_len, 3)
        print('Average UNCOND\' mn-len in days for type %s = %s'
              % (t, avg_len))

########## check histogram

acs = pd.read_csv('./output/output_20191205_132017_Main/acs_sim_20191205_132017.csv')
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
acs['cpl'] = [x.sum() for x in acs[['cpl_%s' % x for x in types]].values]

######## check why simulated anypay = nan


import pandas as pd
import numpy as np
import bisect
import json
from time import time
from _5a_aux_functions import get_columns, get_sim_col, get_weighted_draws
import sklearn.linear_model, sklearn.naive_bayes, sklearn.neighbors, sklearn.tree, sklearn.ensemble, \
    sklearn.gaussian_process, sklearn.svm
from datetime import datetime
import matplotlib.pyplot as plt
import os
import csv
from _1_clean_FMLA import DataCleanerFMLA
from _4_clean_ACS import DataCleanerACS
from Utils import format_chart


# Read in cleaned ACS and FMLA data, and FMLA-based length distribution
acs = pd.read_csv('./data/acs/ACS_cleaned_forsimulation_2016_md.csv')
pfl = 'non-PFL'  # status of PFL as of ACS sample period
d = pd.read_csv('./data/fmla_2012/fmla_clean_2012.csv', low_memory=False)
with open('./data/fmla_2012/length_distributions_exact_days.json') as f:
    flen = json.load(f)

# Sample restriction - reduce to eligible workers (all elig criteria indep from simulation below)

# drop government workers if desired
acs = acs.drop(acs[acs['empgov_fed'] == 1].index)
acs = acs.drop(acs[acs['empgov_st'] == 1].index)
acs = acs.drop(acs[acs['empgov_loc'] == 1].index)
acs = acs.drop(acs[(acs['COW'] == 6) | (acs['COW'] == 7)].index)
# check other program eligibility
acs['elig_prog'] = 0
acs.loc[(acs['wage12'] >= 3840) &
        (acs['wkswork'] >= 1) &
        (acs['wkswork'] * acs['wkhours'] >= 1) &
        (acs['empsize'] >= 1), 'elig_prog'] = 1
# drop ineligible workers (based on wage/work/empsize)
acs = acs.drop(acs[acs['elig_prog'] != 1].index)

# Define classifier
clf = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto', random_state=123)

# Train models using FMLA, and simulate on ACS workers
t0 = time()
col_Xs, col_ys, col_w = get_columns()
X = d[col_Xs]
w = d[col_w]
Xa = acs[X.columns]
random_seed = 123
random_state = np.random.RandomState(random_seed)

for c in col_ys:
    tt = time()
    y = d[c]
    acs[c] = get_sim_col(X, y, w, Xa, clf, random_state)
    print('Simulation of col %s done. Time elapsed = %s' % (c, (time() - tt)))
print('6+6+1 simulated. Time elapsed = %s' % (time() - t0))

# Post-simluation logic control
acs.loc[acs['male'] == 1, 'take_matdis'] = 0
acs.loc[acs['male'] == 1, 'need_matdis'] = 0
acs.loc[(acs['nevermarried'] == 1) | (acs['divorced'] == 1), 'take_illspouse'] = 0
acs.loc[(acs['nevermarried'] == 1) | (acs['divorced'] == 1), 'need_illspouse'] = 0
acs.loc[acs['nochildren'] == 1, 'take_bond'] = 0
acs.loc[acs['nochildren'] == 1, 'need_bond'] = 0
acs.loc[acs['nochildren'] == 1, 'take_matdis'] = 0
acs.loc[acs['nochildren'] == 1, 'need_matdis'] = 0
acs.loc[acs['age'] > 50, 'take_matdis'] = 0
acs.loc[acs['age'] > 50, 'need_matdis'] = 0
acs.loc[acs['age'] > 50, 'take_bond'] = 0
acs.loc[acs['age'] > 50, 'need_bond'] = 0

# Conditional simulation - anypay for taker/needer sample
acs['taker'] = [max(z) for z in acs[['take_%s' % t for t in types]].values]
acs['needer'] = [max(z) for z in acs[['need_%s' % t for t in types]].values]
X = d[(d['taker'] == 1) | (d['needer'] == 1)][col_Xs]
w = d.loc[X.index][col_w]
Xa = acs[(acs['taker'] == 1) | (acs['needer'] == 1)][X.columns]
if len(Xa) == 0:
    print('Warning: Neither leave taker nor leave needer present in simulated ACS persons. '
          'Simulation gives degenerate scenario of zero leaves for all workers.')
else:
    for c in ['anypay']:
        y = d.loc[X.index][c]
        simcol_indexed = get_sim_col(X, y, w, Xa, clf, random_state)
        simcol_indexed = pd.Series(simcol_indexed, index=Xa.index, name=c)
        acs = acs.join(simcol_indexed)

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
    y = [D_prop[x] for x in d.loc[X.index]['prop_pay']]
    yhat = get_sim_col(X, y, w, Xa, clf, random_state)
    # prop_pay labels are from 1 to 6, get_sim_col() vectorization sum gives 0~5, increase label by 1
    yhat = pd.Series(data=[x+1 for x in yhat], index=Xa.index, name='prop_pay')
    acs = acs.join(yhat)
    acs.loc[acs['prop_pay'].notna(), 'prop_pay'] = [d_prop[x] for x in
                                                    acs.loc[acs['prop_pay'].notna(), 'prop_pay']]

acs[acs['dlen_own']<0][['len_own', 'mnl_own', 'cfl_own', 'prop_pay', 'anypay', 'taker', 'needer', 'len_all', 'cfl_all', 'mnl_all']]

