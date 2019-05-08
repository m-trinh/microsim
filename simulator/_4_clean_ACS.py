
"""
This program loads in the raw ACS files, creates the necessary variables
for the simulation and saves a master dataset to be used in the simulations.

2 March 2018
hautahi

To do:
- The biggest missing piece is the imputation of ACS variables using the CPS. These are currently just randomly generated.
- Check if the ACS variables are the same as those in the C++ code

"""

# -------------------------- #
# Housekeeping
# -------------------------- #

import pandas as pd
import numpy as np
from time import time

t0 = time()
# -------------------------- #
# ACS Household File
# -------------------------- #

# Load data
st = 'ca'
d_hh = pd.read_csv("C:/workfiles/Microsimulation/git/large_data_files/ss15h%s.csv" % st)

# Create Variables
d_hh["nochildren"]  = pd.get_dummies(d_hh["FPARC"])[4]
d_hh['faminc'] = d_hh['FINCP']*d_hh['ADJINC'] / 1042852 / 1000 # adjust to 2012 thousand dollars to conform with FMLA 2012 data
d_hh.loc[(d_hh['faminc']<=0), 'faminc'] = 0.01/1000 # force non-positive income to be epsilon to get meaningful log-income
d_hh["lnfaminc"]    = np.log(d_hh["faminc"])

# Number of dependents
d_hh['ndep_kid'] = d_hh.NOC
d_hh['ndep_old'] = d_hh.R65

# -------------------------- #
# ACS Personal File
# -------------------------- #

chunk_size = 100000
dout = pd.DataFrame([])
ichunk = 1

# Load data
for d in pd.read_csv("C:/workfiles/Microsimulation/git/large_data_files/ss15p%s.csv" % st, chunksize=chunk_size):

    # Merge with the household level variables
    d = pd.merge(d,d_hh[['SERIALNO', 'nochildren', 'faminc','lnfaminc','PARTNER', 'ndep_kid', 'ndep_old']],
                     on='SERIALNO')

    # Rename ACS variables to be consistent with FMLA data
    rename_dic = {'AGEP': 'age'}
    d.rename(columns=rename_dic, inplace=True)

    # duplicating age column for meshing with CPS estimate output
    d['a_age']=d['age']

    # Create new ACS Variables
    d['married'] = pd.get_dummies(d["MAR"])[1]
    d["widowed"]        = pd.get_dummies(d["MAR"])[2]
    d["divorced"]       = pd.get_dummies(d["MAR"])[3]
    d["separated"]      = pd.get_dummies(d["MAR"])[4]
    d["nevermarried"]   = pd.get_dummies(d["MAR"])[5]
        # use PARTNER in household data to tease out unmarried partners
    d['partner'] = np.where((d['PARTNER']==1) | (d['PARTNER']==2) | (d['PARTNER']==3) | (d['PARTNER']==4), 1, 0)
    for m in ['married', 'widowed', 'divorced', 'separated', 'nevermarried']:
        d.loc[d['partner']==1, m] = 0

    d["male"]           = pd.get_dummies(d["SEX"])[1]
    d["female"]         = 1 - d["male"]
    d["agesq"]          = d["age"]**2

    # Educational level
    d['sku'] = np.where(d['SCHL'].isna(), 0, d['SCHL'])
    d['ltHS']    = np.where(d['sku']<=11,1,0)
    d['someHS']  = np.where((d['sku']>=12) & (d['sku']<=15),1,0)
    d['HSgrad']  = np.where((d['sku']>=16) & (d['sku']<=17),1,0)
    d['someCol']  = np.where((d['sku']>=18) & (d['sku']<=20),1,0)
    d["BA"]      = np.where(d['sku']==21,1,0)
    d["GradSch"]  = np.where(d['sku']>=22,1,0)

    d["noHSdegree"]  = np.where(d['sku']<=15,1,0)
    d["BAplus"]  = np.where(d['sku']>=21,1,0)
        # variables for imputing hourly status, using CPS estimates from original model
    d["maplus"]  = np.where(d['sku']>=22,1,0)
    d["ba"]      = d['BA']

    # race
    d['hisp'] = np.where(d['HISP']>=2, 1, 0)
    d['white'] = np.where((d['hisp']==0) & (d['RAC1P']==1), 1, 0)
    d['black'] = np.where((d['hisp']==0) & (d['RAC1P']==2), 1, 0)
    d['asian'] = np.where((d['hisp']==0) & (d['RAC1P']==6), 1, 0)
    d['native'] = np.where((d['hisp'] == 0) & ((d['RAC1P'] == 3) |
                                               (d['RAC1P'] == 4) |
                                               (d['RAC1P'] == 5) |
                                               (d['RAC1P'] == 7)), 1, 0)
    d['other'] = np.where((d['hisp']==0) & (d['white']==0) & (d['black']==0) & (d['asian']==0) & (d['native']==0), 1, 0)

    # Employed
    d['employed'] = np.where((d['ESR']== 1) |
                             (d['ESR'] == 2) |
                             (d['ESR'] == 4) |
                             (d['ESR'] == 5) ,
                             1, 0)
    d['employed'] = np.where(np.isnan(d['ESR']),np.nan,d['employed'])

    # Hours per week, total working weeks
    d['wkhours'] = d['WKHP']
    dict_wks = {
        1: 51,
        2: 48.5,
        3: 43.5,
        4: 33,
        5: 20,
        6: 7
    }
    d['weeks_worked_cat'] = d['WKW'].map(dict_wks)
    d['weeks_worked_cat'] = np.where(d['weeks_worked_cat'].isna(), 0, d['weeks_worked_cat'])
    # Total wage past 12m, adjusted to 2012, and its log
    d['wage12'] = d['WAGP'] *d['ADJINC'] / 1042852
    d['lnearn'] = np.nan
    d.loc[d['wage12']>0, 'lnearn'] = d.loc[d['wage12']>0, 'wage12'].apply(lambda x: np.log(x))

    # health insurance from employer
    d['hiemp'] = np.where(d['HINS1']==1, 1, 0)
    d['hiemp'] = np.where(d['HINS1'].isna(), np.nan, d['hiemp'])

    # Employment at government
        # missing = age<16, or NILF over 5 years, or never worked
    d['empgov_fed'] = np.where(d['COW']==5, 1, 0)
    d['empgov_fed'] = np.where(np.isnan(d['COW']),np.nan,d['empgov_fed'])
    d['empgov_st'] = np.where(d['COW']==4, 1, 0)
    d['empgov_st'] = np.where(np.isnan(d['COW']),np.nan,d['empgov_st'])
    d['empgov_loc'] = np.where(d['COW']==3, 1, 0)
    d['empgov_loc'] = np.where(np.isnan(d['COW']),np.nan,d['empgov_loc'])

    # Presence of children for females
    d['fem_cu6'] = np.where(d['PAOC']==1, 1, 0)
    d['fem_c617'] = np.where(d['PAOC']==2, 1, 0)
    d['fem_cu6and617'] = np.where(d['PAOC']==3, 1, 0)
    d['fem_nochild'] = np.where(d['PAOC']==4, 1, 0)
    for x in ['fem_cu6','fem_c617','fem_cu6and617','fem_nochild']:
        d.loc[d['PAOC'].isna(), x] = np.nan

    # Occupation

    # make numeric OCCP = OCCP10 if ACS 2011-2015, or OCCP12 if ACS 2012-2016
    if 'N.A.' in d['OCCP12'].value_counts().index:
        d.loc[d['OCCP12']=='N.A.', 'OCCP12'] = d.loc[d['OCCP12']=='N.A.', 'OCCP12'].apply(lambda x: np.nan)
    d.loc[d['OCCP12'].notna(), 'OCCP12'] = d.loc[d['OCCP12'].notna(), 'OCCP12'].apply(lambda x: int(x))

    if 'N.A.' in d['OCCP10'].value_counts().index:
        d.loc[d['OCCP10']=='N.A.', 'OCCP10'] = d.loc[d['OCCP10']=='N.A.', 'OCCP10'].apply(lambda x: np.nan)
    d.loc[d['OCCP10'].notna(), 'OCCP10'] = d.loc[d['OCCP10'].notna(), 'OCCP10'].apply(lambda x: int(x))

    d['OCCP'] = np.nan
    d['OCCP'] = np.where(d['OCCP12'].notna(), d['OCCP12'], d['OCCP'])
    d['OCCP'] = np.where((d['OCCP'].isna()) & (d['OCCP12'].isna()) & (d['OCCP10'].notna()), d['OCCP10'], d['OCCP'])


    d['occ_1']=0
    d['occ_2']=0
    d['occ_3']=0
    d['occ_4']=0
    d['occ_5']=0
    d['occ_6']=0
    d['occ_7']=0
    d['occ_8']=0
    d['occ_9']=0
    d['occ_10']=0
    d['maj_occ']=0
    d.loc[(d['OCCP']>=10) & (d['OCCP']<=950), 'occ_1'] =1
    d.loc[(d['OCCP']>=1000) & (d['OCCP']<=3540), 'occ_2'] =1
    d.loc[(d['OCCP']>=3600) & (d['OCCP']<=4650), 'occ_3'] =1
    d.loc[(d['OCCP']>=4700) & (d['OCCP']<=4965), 'occ_4'] =1
    d.loc[(d['OCCP']>=5000) & (d['OCCP']<=5940), 'occ_5'] =1
    d.loc[(d['OCCP']>=6000) & (d['OCCP']<=6130), 'occ_6'] =1
    d.loc[(d['OCCP']>=6200) & (d['OCCP']<=6940), 'occ_7'] =1
    d.loc[(d['OCCP']>=7000) & (d['OCCP']<=7630), 'occ_8'] =1
    d.loc[(d['OCCP']>=7700) & (d['OCCP']<=8965), 'occ_9'] =1
    d.loc[(d['OCCP']>=9000) & (d['OCCP']<=9750), 'occ_10'] =1
        # make sure occ_x gets nan if OCCP code is nan
    for x in range(1, 11):
        d.loc[d['OCCP'].isna(), 'occ_%s' % x] = np.nan

    # Industry
    d['ind_1']=0
    d['ind_2']=0
    d['ind_3']=0
    d['ind_4']=0
    d['ind_5']=0
    d['ind_6']=0
    d['ind_7']=0
    d['ind_8']=0
    d['ind_9']=0
    d['ind_10']=0
    d['ind_11']=0
    d['ind_12']=0
    d['ind_13']=0
    d.loc[(d['INDP']>=170) & (d['INDP']<=290), 'ind_1'] =1
    d.loc[(d['INDP']>=370) & (d['INDP']<=490), 'ind_2'] =1
    d.loc[(d['INDP']==770), 'ind_3'] =1
    d.loc[(d['INDP']>=1070) & (d['INDP']<=3990), 'ind_4'] =1
    d.loc[(d['INDP']>=4070) & (d['INDP']<=5790), 'ind_5'] =1
    d.loc[((d['INDP']>=6070) & (d['INDP']<=6390))|((d['INDP']>=570) & (d['INDP']<=690)), 'ind_6'] =1
    d.loc[(d['INDP']>=6470) & (d['INDP']<=6780), 'ind_7'] =1
    d.loc[(d['INDP']>=6870) & (d['INDP']<=7190), 'ind_8'] =1
    d.loc[(d['INDP']>=7270) & (d['INDP']<=7790), 'ind_9'] =1
    d.loc[(d['INDP']>=7860) & (d['INDP']<=8470), 'ind_10'] =1
    d.loc[(d['INDP']>=8560) & (d['INDP']<=8690), 'ind_11'] =1
    d.loc[(d['INDP']>=8770) & (d['INDP']<=9290), 'ind_12'] =1
    d.loc[(d['INDP']>=9370) & (d['INDP']<=9590), 'ind_13'] =1
        # make sure ind_x gets nan if INDP code is nan
    for x in range(1, 14):
        d.loc[d['INDP'].isna(), 'ind_%s' % x] = np.nan
    # -------------------------- #
    # Remove ineligible workers
    # -------------------------- #

    # Restrict dataset to civilian employed workers (check this)
        # d = d[(d['ESR'] == 1) | (d['ESR'] == 2)]

    #  Restrict dataset to those that are not self-employed
        # d = d[(d['COW'] != 6) & (d['COW'] != 7)]

    # -------------------------- #
    # CPS Imputation
    # -------------------------- #

    """
    Not all the required behavioral independent variables are available
    within the ACS. These therefore need to be imputed CPS

    d["weeks_worked"] =
    d["weekly_earnings"] =
    """

    # These are just placeholders for now
    # d["coveligd"] = np.nan
    # d['union'] = np.nan

    # adding in the prhourly worker imputation
    # Double checked C++ code, and confirmed this is how they did hourly worker imputation.
    hr_est=pd.read_csv('estimates/CPS_paid_hrly.csv').set_index('var').to_dict()['est']
    d['prhourly']=0
    for dem in hr_est.keys():
        if dem!='Intercept':
            d['prhourly']+=d[dem].fillna(0)*hr_est[dem]
    d['prhourly']+=hr_est['Intercept']
    d['prhourly']=np.exp(d['prhourly'])/(1+np.exp(d['prhourly']))
    d['rand']=pd.Series(np.random.rand(d.shape[0]), index=d.index)
    d['hourly']= np.where(d['prhourly']>d['rand'],1,0)
    d=d.drop('rand', axis=1)

    # -------------------------- #
    # Save the resulting dataset
    # -------------------------- #
    cols = ['SERIALNO', 'PWGTP', 'ST',
    'employed', 'empgov_fed','empgov_st','empgov_loc',
    'wkhours', 'weeks_worked_cat', 'wage12', 'lnearn','hiemp',
    'a_age','age', 'agesq',
    'male','female',
    'nochildren', 'ndep_kid', 'ndep_old',
    'ltHS', 'someHS', 'HSgrad', 'someCol', 'BA', 'GradSch', 'noHSdegree', 'BAplus' ,
    'faminc', 'lnfaminc',
    'married', 'partner', 'separated', 'divorced', 'widowed', 'nevermarried',
    'asian', 'black', 'white', 'native','other', 'hisp',
    'fem_cu6','fem_cu6and617','fem_c617','fem_nochild',
    'ESR', 'COW' # original ACS vars for restricting samples later
            ]
    cols += ['ind_%s' % x for x in range(1, 14)]
    cols += ['OCCP'] + ['occ_%s' % x for x in range(1, 11)]
    # cols += ['hourly'] # optional, move imputation of hourly to somewhere else?

    d_reduced = d[cols]
    dout = dout.append(d_reduced)
    print('ACS data cleaned for chunk %s of personal data...' % ichunk)
    ichunk += 1
dout.to_csv("./data/acs/ACS_cleaned_forsimulation_%s.csv" % st, index=False, header=True)

    # a wage only file for testing ABF
# d2 = d_reduced[['wage12','PWGTP']]
# d2.to_csv("./data/acs/ACS_cleaned_forsimulation_%s_wage.csv" % st, index=False, header=True)

t1 = time()
print('ACS data cleaning finished for state %s. Time elapsed = %s' % (st.upper(), (t1-t0)))

