'''
check FMLA 2018 data

chris zhang 4/9/2020

-- key vars
main weight: combo_trimmed_weight
rep weight: cmb_bsrw[x] x=1~200
fmla_eligible: now available, also with counterfactual versions (_if15/30hrwk, _if30/20emp)
most recent/longest leave loops: _mr and _long varname suffixes, _mr if 2 are same (_long skipped)
./.S/.U/.N all read as nan by pd.read_stata() - e.g. ne16_coded
paid_leave_state: now available
time conversion:  1 month = 4.5 weeks, 1 week = 5 work days, 1 day = 8 work hours

-- recent/long choice
use recent (as FMLA 2012) because:
1. recent leave provides granular leave length (long does not), see a19_mr_cat
2. using recent leave is mathematically robust:
(2.1) recent leave length can be thought of as a representative sample of leaves at a timestamp t in a year.
This holds as long as we assume recent leave length/type is random for multiple leavers.
(2.2) due to above, the leave taker sample and their leaves are not limited to 'recent'. It is just a random sample of
leave types of leave takers
(2.3) for a sim-ed ACS person, she may take multiple (say 2) types (a and b) if she 'looks like' both type-a and type-b
takers. We therefore let her to be multiple leave taker with types (a,b).
(2.4) Doing above implies relaxed criteria for being sim-ed as multiple takers - one in ACS just needs to look like
type a and type b separately to be multiple taker (a, b), regardless of whether she resembles an FMLA multiple taker.
This is a conservative method for avoiding underestimation of program costs (although overestimation of program impact).

--leave length
a19_mr_cat: granularity loss for 5+ days. Can do average/impute. Use 2012 data to check effect of these methods

--prop_pay (use recStatePay=0 subsample to ensure pay from company only)
get prop_pay (average) from vars below
a43d_cat: # days full pay
a43f_cat: # days partial pay
a43g_cat: partial pay % groups

--resp_len
a55: would take longer leave if more pay
na62b: return to work due to leaves running out
na62f: return to work due to no longer needing leaves (resp_len=0)
b15e: couldn't afford to take unpaid leave

--covelig
e0c_cat, e1a, e1b, etc. AND ne6 (self perception of eligibility)

--ind, occ
ne15_ocded, ne16_coded


'''

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np

## Read in data
fp_in = './data/fmla/fmla_2018/FMLA 2018 PUF/FMLA_2018_Employee_PUF.dta'
d = pd.read_stata(fp_in, convert_categoricals=False)

# Make empid to follow 0-order to be consistent with Python standard (e.g. indices output from kNN)
d['empid'] = d['empid'] - 1

# FMLA eligibility - use fmla_eligible, same as coveligd in FMLA 2012

# Union status - in CPS but not ACS, can sim in ACS if FMLA data shows necessary
d['union'] = d['d3']

# Hours per week (main job if multiple jobs, mid-point)
# a common dict_wkhours for e0b_cat, e0g_cat
dict_wkhours = {
    0: 0,
    1: 2.5,
    2: 7,
    3: 12,
    4: 17,
    5: 22,
    6: 27,
    7: 32,
    8: 37,
    9: 42,
    10: 47,
    11: 52,
    12: 57, # boundary cat for e0b_cat, so approximation
    13: 62,
    14: 67,
    15: 77,
    16: 85, # boundary cat for e0g_cat, so approximation
    np.nan: np.nan
}
d['wkhours'] = [dict_wkhours[x] if not np.isnan(x) else np.nan for x in d['e0b_cat']]
d['wkhours_m'] = [dict_wkhours[x] if not np.isnan(x) else np.nan for x in d['e0g_cat']] # main job hours if 2+ jobs
d['wkhours'] = np.where((d['wkhours'].isna()) & (d['wkhours_m'].notna()), d['wkhours_m'], d['wkhours'])
del d['wkhours_m']

# Employment at government - use govt_emp=1 (no fed/state/local info in FMLA 2018)

# Age
dct_age={
    1:21,
    2:27,
    3:32,
    4:37,
    5:42,
    6:47,
    7:52,
    8:57,
    9:63,
    10:70 # boundary category
}
d['age'] = [dct_age[x] if not np.isnan(x) else x for x in d['age_cat']]

# Sex
d['female'] = np.where(d['gender_cat']==2, 1, 0)
d['female'] = np.where(d['gender_cat'].isna(), np.nan, d['female'])

