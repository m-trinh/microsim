from _1_clean_FMLA import DataCleanerFMLA
from _4_clean_ACS import DataCleanerACS
from _5_simulation import SimulationEngine
from _5a_aux_functions import *
import pandas as pd
import numpy as np
import bisect

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
    'own': 0.5,
    'matdis': 0.5,
    'bond': 0.5,
    'illchild': 0.5,
    'illspouse': 0.5,
    'illparent': 0.5
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
# TODO: takeup rates denominator = all eligible OR all who choose prog? Table 1&2 of ACM doc, back out pay schedule

# elig_wage12 = 3440
# elig_wkswork = 20
# elig_yrhours = 1
# elig_empsizebin = 1
# rrp = 0.67
# wkbene_cap = 650
# d_maxwk = dict(zip(self.types, 6*np.ones(6)))
# d_takeup = dict(zip(self.types, 1*np.ones(6)))
# incl_empgov = False
# incl_empself = False

# get individual cost
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

# program eligibility - TODO: port to GUI input, program eligibility determinants
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

acs = acs.drop(acs[acs['elig_prog'] != 1].index)

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
# 1. A fraction (up to 100%) of workers can and choose to receive both employer/state benefits simultaneously or sequentially
# 2. For double receivers, at status-quo, rr = (rre, 0), leave length = len_type
# At full replacement, rr = (1, 1), leave length = mnl_type
# At program scenario, rr = (rre, rrp), leave length = cfl_type which is linearly interpolated between [rre, 2]
# cfl_type then needs to be distributed between employer and state benefits. We distribute in proportion to rre/rrp.
# The resulting cfl_prog_type will then be subject to max period cap, and rrp * cfl_prog_type will be subject to $ cap.
# 3. For single receivers, at status-quo, rr = rre, len = len_type.
# At full replacement, rr = 1, len = mnl_type
# At program scenario, rr = rrp. If rrp<=rre, len = len_type (use no program). If rrp > rre, len = cfl_prog_type, which
# is linearly interpolated between [rre, 1].

###########################################################################################



# assumption 1: choice between employer and state benefits
# rre, rrp = replacement ratio of employer, of state
# if rre >= rrp, use employer pay indefinitely (assuming employer imposes no max period)
# if rre <  rrp, use employer pay if weekly wage*rre > state weekly cap (the larger weekly wage*rrp would be capped)
# so only case of using state benefits (thus induced to take longer leave) is rre < rrp and weekly wage*rre < cap
# TODO: assumption 1 perhaps too strict - use of employer pay may be limited by (shorter) max length!

# identify workers who have rre < rrp, and are 'uncapped' by state weekly benefit cap under current weekly wage and prop_pay
# thus would prefer state benefits over employer
acs['uncapped'] = True
acs['uncapped'] = ((acs['prop_pay'] < self.rrp) & (acs['wage12'] / acs['wkswork'] * acs['prop_pay'] < self.wkbene_cap))

# assumption 2: if using state benefits, choice of leave length is a function of replacement ratio
# at current rr = prop_pay, leave length = status-quo (sql), at 100% rr, leave length = max needed length (mnl)
# at proposed prog rr = rrp, if rrp> (1-prop_pay), then leave length = mnl. Leave length covered by program = mnl - sql
# OW, linearly interpolate at rr = (prop_pay + rrp). Leave length covered by program = sql+(mnl - sql)*(rrp-prop_pay)/(1-prop_pay)




# set prop_pay = 0 if missing (ie anypay = 0 in acs)
acs.loc[acs['prop_pay'].isna(), 'prop_pay'] = 0

# cpl: covered-by-program leave length, 6 types - derive by interpolation at rrp (between rre and 1)
for t in self.types:
    # under assumptions 1 + 2:
    # acs['cpl_%s' % t] = 0
    # acs.loc[(acs['prop_pay'] < rrp) & (acs['uncapped']), 'cpl_%s' % t] = \
    #     acs['len_%s' % t] + (acs['mnl_%s' % t] - acs['len_%s' % t]) * (rrp - acs['prop_pay'])/ (1-acs['prop_pay'])

    # assumption 1A: asssuming short max length of employer benefit, use state benefit if sq len >= 5/10 days regardless of rr value
    # motivation: with leave need>=a week workers do not consider employer benefit like PTO but take program benefit
    # if rre < rrp <=1, interpolation of leave length is possible
    # if rre = 1, cannot interpolate length using rre/length relationship. These workers are likely due to bad need
    # of leave length but no much of wage replacement. Assume cpl = mnl then apply max period cap.
    # if rre >=rrp, workers make choice between get large rre over short period (say 5 days) VS
    # get less rrp over longer period (mnl). To avoid underestimating cost, we assume all workers choose latter.
    # under assumptions 1A + 2:
    acs['cpl_%s' % t] = 0

    # if rre < rrp <=1
    acs.loc[acs['prop_pay'] < self.rrp, 'cpl_%s' % t] = \
        acs['len_%s' % t] + (acs['mnl_%s' % t] - acs['len_%s' % t]) * (self.rrp - acs['prop_pay']) / (
        1 - acs['prop_pay'])
    # if rre >=rrp and MNL > 5
    acs.loc[(acs['prop_pay'] >= self.rrp) & (acs['mnl_%s' % t] > 5), 'cpl_%s' % t] = acs['mnl_%s' % t]
    # finally no program use if cpl <= 5
    acs.loc[acs['cpl_%s' % t] <= 5, 'cpl_%s' % t] = 0
    # take integer cpl
    acs['cpl_%s' % t] = acs['cpl_%s' % t].apply(lambda x: math.ceil(x))

    # apply max number of covered weeks
    acs['cpl_%s' % t] = acs['cpl_%s' % t].apply(lambda x: min(x, self.d_maxwk[t] * 5))

    # does max # covered weeks cause employer pay more attractive?
    # under rre: get Be = acs['len_%s' % t] * acs['prop_pay']
    # under rrp > rre: get Bp = acs['cpl_%s' % t] * rrp
    # assumption 3: use state benefits if total benefit under state is higher, i.e. Bp > Be
    # so assign cpl_[type] = 0 if Bp <= Be
    # TODO: assumption 3 perhaps too strict - higher Be with lower rre in longer period may not be preferred!
    # because with length>x days in program using employer benefit can cause undesirable emp bene depletion (PTO?)
    # TODO: no benefit crowdout on matdis/bond if have paternity/maternity pay (A46e, A46f). Impute on ACS.
    # acs.loc[(acs['cpl_%s' % t] * rrp <= acs['len_%s' % t] * acs['prop_pay']), 'cpl_%s' % t] = 0

# Save ACS data after finishing simulation
acs.to_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id), index=False)
message = 'Leaves simulated for 5-year ACS 20%s-20%s in state %s. Time needed = %s seconds' % (
(self.yr - 4), self.yr, self.st.upper(), round(time() - tsim, 0))
print(message)
self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 95})
self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
return acs
