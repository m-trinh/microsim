'''
test consistency of ACS between Python and R

chris zhang 1/24/2020
'''
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from _5a_aux_functions import get_bool_num_cols

# a function to read in data and reduce cols
def preprocess_data(fp_acs, cols):
    # Read in ACS
    d = pd.read_csv(fp_acs)
    # R - keep eligible workers
    try:
        d = d[d['eligworker']==1]
    except KeyError:
        pass
    # R - rename len cols
    dct_cols = dict(zip(['squo_length_%s' % x for x in types], ['len_%s' % x for x in types]))
    dct_cols.update(dict(zip(['length_%s' % x for x in types], ['cfl_%s' % x for x in types])))
    dct_cols.update(dict(zip(['plen_%s' % x for x in types], ['cpl_%s' % x for x in types])))
    dct_cols.update(dict(zip(['ptake_%s' % x for x in types], ['takeup_%s' % x for x in types])))
    dct_cols.update({'weeks_worked': 'wkswork'})
    d = d.rename(columns=dct_cols)

    # TODO: remove below
    if 'takeup_illparent' not in d.columns:
        d.loc[d['take_illparent']==1, 'takeup_illparent']=1

    # reduce cols
    d = d[cols]

    # ## Standardize
    # # use N-ddof = N-1 as in R
    # # https://stackoverflow.com/questions/27296387/difference-between-r-scale-and-sklearn-preprocessing-scale/27297618
    #
    # # TODO: check how fillna works in R - need to be same as Python
    # # drop missing rows if any col is missing
    # #d = d.dropna(subset=Xs)
    # # standardize
    # for X in Xs:
    #     d['z_%s' % X] = (d[X] - d[X].mean()) / np.std(d[X], axis=0, ddof=1)
    # print(d[['SERIALNO', 'age','z_age', 'female', 'z_female']].head())
    # cols to return


    return d


## Read in data
# Python
#dp = pd.read_csv('./data/acs/ACS_cleaned_forsimulation_2016_ri.csv')
#fp_p = './data/acs/ACS_cleaned_forsimulation_2016_ri.csv'
fp_p = './output/output_20200220_113100_main simulation/acs_sim_20200220_113100.csv'
# R
#dr = pd.read_csv('./PR_comparison/check_acs/RI_work.csv')
fp_r = './PR_comparison/check_acs/acs_R/RI_logitfull_test.csv'

## preprocess

# id
id = 'SERIALNO'
# Xs, ys, w
Xs = ['widowed', 'divorced', 'separated', 'nevermarried',
      'female', 'age', 'agesq',
      'ltHS', 'someCol', 'BA', 'GradSch',
      'black', 'other', 'asian', 'native', 'hisp',
      'nochildren', 'faminc', 'coveligd']
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
ys = ['take_%s' % x for x in types] + ['need_%s' % x for x in types] + ['resp_len']
w = 'PWGTP'
lens = ['len_%s' % x for x in types]
lens+= ['mnl_%s' % x for x in types]
lens+= ['cfl_%s' % x for x in types]
lens+= ['cpl_%s' % x for x in types]
cols_cost = ['takeup_%s' % x for x in types]
cols_cost += ['wage12', 'wkswork']

cols = [id] + Xs + ys + [w] + lens + cols_cost

dp = preprocess_data(fp_p, cols)
dr = preprocess_data(fp_r, cols)

## Check bool num cols
bool_cols, num_cols = get_bool_num_cols(dr)
for c in bool_cols:
    if c[:2]!='z_':
        print('----------------- xvar = %s -----------------------' % c)
        print('--- %s - Py ---' % c)
        print(dp[c].value_counts().sort_index())
        print('--- %s - R ---' % c)
        print(dr[c].value_counts().sort_index())

for c in num_cols:
    if c[:2]!='z_':
        print('----------------- xvar = %s -----------------------' % c)
        print('--- %s - Py ---' % c)
        if len(dp[c].isna().value_counts())==2:
            print('nrow missing = %s' % dp[c].isna().value_counts()[True])
        print(dp[c].mean())
        print('--- %s - R ---' % c)
        if len(dr[c].isna().value_counts())==2:
            print('nrow missing = %s' % dr[c].isna().value_counts()[True])
        print(dr[c].mean())

# check lens
for c in lens:
    # fillna with 0 for dp[c] for Py
    dp[c] = dp[c].fillna(0)

    if c[:2] != 'z_':
        print('----------------- xvar = %s -----------------------' % c)
        print('--- %s - Py ---' % c)
        if len(dp[c].isna().value_counts()) == 2:
            print('nrow missing = %s' % dp[c].isna().value_counts()[True])
        print(dp[c].mean())
        print('--- %s - R ---' % c)
        if len(dr[c].isna().value_counts()) == 2:
            print('nrow missing = %s' % dr[c].isna().value_counts()[True])
        print(dr[c].mean())
# TODO: coveligd - both Py/R no missing. But #s mismatch

## Compute total outlay
# params
params = {}
params['wkbene_cap'] = 1216
params['rrp'] = 0.55
pow_pop_multiplier = 1.02
def get_costs(df):
    # apply take up flag and weekly benefit cap, and compute total cost, 6 types
    costs = {}
    for t in types:
        # v = capped weekly benefit of leave type
        v = [min(x, params['wkbene_cap']) for x in
             ((df['wage12'] / df['wkswork'] * params['rrp']))]
        # inflate weight for missing POW
        w = df['PWGTP'] * pow_pop_multiplier
    
        # get program cost for leave type t - sumprod of capped benefit, weight, and takeup flag for each ACS row
        costs[t] = (v * df['cpl_%s' % t] / 5 * w * df['takeup_%s' % t]).sum()
    costs['total'] = sum(list(costs.values()))
    return costs

cp = get_costs(dp)
cr = get_costs(dr)


'''
  d <- d %>% mutate(base_benefits=wage12/(round(weeks_worked*5))*particip_length*benefit_prop)
  d <- d %>% mutate(actual_benefits=base_benefits)

  d <- d %>% mutate(actual_benefits= ifelse(actual_benefits>ceiling(week_bene_cap*particip_length)/5,
                                            ceiling(week_bene_cap*particip_length)/5, actual_benefits))
'''

acsr = pd.read_csv(fp_r)
acsr = acsr[acsr['eligworker']==1]
acsr['sum_plen'] = [sum(x) for x in acsr[['plen_%s' % x for x in types]].values]
(acsr['sum_plen'] - acsr['particip_length'])

# get total cost as in R code
def get_cost_R(df):
    # apply take up flag and weekly benefit cap, and compute total cost, 6 types
    uncapped_total_benefits = (df['wage12']/df['weeks_worked'])*params['rrp']*(df['particip_length']/5)
    total_caps = params['wkbene_cap']*df['particip_length']/5
    capped_total_benefits = [min(x[0], x[1]) for x in pd.concat([uncapped_total_benefits, total_caps], axis=1).values]
    total_cost = (capped_total_benefits*df['PWGTP']).sum()
    return total_cost
acsr = pd.read_csv(fp_r)
get_cost_R(acsr)

# get total cost as in Py code
def get_cost_Py(df):
    # apply take up flag and weekly benefit cap, and compute total cost, 6 types
    costs = {}
    for t in types:
        # v = capped weekly benefit of leave type
        v = [min(x, params['wkbene_cap']) for x in
             ((df['wage12'] / df['wkswork'] * params['rrp']))]
        # TODO: use this in simulation code - see underestimation! Consider up cpl to (close-to) cfl as in R
        # cpl_all = sum of cpl across types
        cpl_all = (df['cpl_%s' % t] / 5)
        # inflate weight for missing POW
        w = df['PWGTP'] * pow_pop_multiplier

        # get program cost for leave type t - sumprod of capped benefit, weight, and takeup flag for each ACS row
        costs[t] = (v * cpl_all * w * df['takeup_%s' % t]).sum()
    total_cost = sum(list(costs.values()))
    return total_cost
get_cost_Py(dp)

# R has much larger costs on type own, check below
for x in types:
    for c in ['len_%s' % x, 'mnl_%s' % x, 'cfl_%s' % x, 'cpl_%s' % x]:
    #for c in ['cfl_%s' % x, 'cpl_%s' % x]:
        print('dp[dp[%s]>0][%s].mean()\n' % (c, c), dp[dp[c]>0][c].mean())
        #print('dr[dr[%s]>0][%s].mean()\n' % (c, c), dr[dr[c]>0][c].mean())
    print('-----------------')

# check take up flags for types
for t in types:
    print('dp[dp[takeup_%s]==1][PWGTP].sum()\n' % t, dp[dp['takeup_%s' % t]==1]['PWGTP'].sum())
    #print('dr[dr[takeup_%s]==1][PWGTP].sum()\n' % t, dr[dr['takeup_%s' % t]==1]['PWGTP'].sum())
    print('--------------------')
# much bigger! seems like R has used wrong base for eligible worker pop
# should be ~350k but not 415k as in Excel state take up data
# TODO: LP to use ACS eligworker est pop as base to re-gen emp take up rates, apply in GUI and re-run model

# TODO: R's lens are much SHORTER than Py's for all types, check why


fp_p = './output/output_20200217_234056_main simulation/acs_sim_20200217_234056.csv'
acsp = pd.read_csv(fp_p)
for t in types:
    print('mean positive cpl for type = %s\n' % t, acsp[acsp['cpl_%s' % t]>0]['cpl_%s' % t].mean())

##
for t in types:
    for c in acsr.columns:
        if t in c:
            print(c)
    print('-------------')

##
# TODO: R post-sim ACS has no takes_up_illparent
# TODO: in roundstats file RI eligible pop =/= sum of PWGTP from R ACS (357531) which is the same as Py's
# TODO: R's plen still way bigger than Py's, causing over estimation of RI's outlay. line 712 in R4 code shows cpl/cfl link may be problematic
# TODO: cpl should <= cfl. cfl is close in Py/R but cpl diverges a lot. Also see cpl>cfl in R (own) not possible!
# TODO: What GUI takeup rate did R use? Rates in roundstats file look low - why need them?

##
# check: cfl>=cpl for everyone in R ACS?
for t in types:
    dr['c_%s' % t] = (dr['cfl_%s' % t] >= dr['cpl_%s' % t])
    print(dr['c_%s' % t].value_counts().sort_index())
for x in types:
    #for c in ['len_%s' % x, 'mnl_%s' % x, 'cfl_%s' % x, 'cpl_%s' % x]:
    for c in ['cfl_%s' % x, 'cpl_%s' % x]:
        #print('dp[dp[%s]>0][%s].mean()\n' % (c, c), dp[dp[c]>0][c].mean())
        print('dr[dr[%s]>0][%s].mean()\n' % (c, c), dr[dr[c]>0][c].mean())
    print('-----------------')

# use actual benefits col in R to get outlay
acsr = pd.read_csv(fp_r)
1.02*sum([x[0]*x[1] for x in acsr[acsr['eligworker']==1][['actual_benefits', 'PWGTP']].values])

# re-draw take up flags by updating min_takeup_cpl
from _5a_aux_functions import get_columns, get_sim_col, get_weighted_draws
acs = dp.copy()
col_w='PWGTP'
def get_dp_with_updated_takeup_flag(acs, min_takeup_cpl):
    params = {}
    params['d_takeup'] = dict(zip(types, [0.0438, 0.0127, 0.0154, 0.0032, 0.0052, 0.0052]))
    for t in types:
        # cap user-specified take up for type t by max possible takeup = s_positive_cpl, in pop per sim results
        s_positive_cpl = acs[acs['cpl_%s' % t] >= min_takeup_cpl][col_w].sum() / acs[col_w].sum()
        # display warning for unable to reach target pop from simulated positive cpl_type pop
        if col_w == 'PWGTP':
            if params['d_takeup'][t] > s_positive_cpl:
                print('Warning: User-specified take up for type -%s- is capped '
                      'by maximum possible take up rate (share of positive covered-by-program length) '
                      'based on simulation results, at %s.' % (t, s_positive_cpl))
        takeup = min(s_positive_cpl, params['d_takeup'][t])
        p_draw = takeup / s_positive_cpl  # need to draw w/ prob=p_draw from cpl>min_takeup_cpl subpop, to get desired takeup
        # print('p_draw for type -%s- = %s' % (t, p_draw))
        # get take up indicator for type t - weighted random draw from cpl_type>min_takeup_cpl until target is reached
        acs['takeup_%s' % t] = 0
        draws = get_weighted_draws(acs[acs['cpl_%s' % t] >= min_takeup_cpl][col_w], p_draw, random_state=np.random.RandomState(12345))
        # print('draws = %s' % draws)
        acs.loc[acs['cpl_%s' % t] >= min_takeup_cpl, 'takeup_%s' % t] \
            = draws

        # for main weight, check if target pop is achieved among eligible ACS persons
        if col_w == 'PWGTP':
            s_takeup = acs[acs['takeup_%s' % t] == 1][col_w].sum() / acs[col_w].sum()
            s_takeup = round(s_takeup, 4)
            print('Specified takeup for type %s = %s. '
                  'Effective takeup = %s. '
                  'Post-sim weighted share = %s' % (t, params['d_takeup'][t], takeup, s_takeup))
    return acs
dp = get_dp_with_updated_takeup_flag(acs, 25)
for t in types:
    print('conditional mean of cpl_%s among takeup persons' % t,
          dp[(dp['cpl_%s' % t]>0) & (dp['takeup_%s' % t]==1)]['cpl_%s' % t].mean())

