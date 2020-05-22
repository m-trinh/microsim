
"""
This program loads in the raw ACS files, creates the necessary variables
for the simulation and saves a master dataset to be used in the simulations.

chris zhang 2/28/2019
"""
# -------------------------- #
# Housekeeping
# -------------------------- #

import pandas as pd
import numpy as np
from _5a_aux_functions import fillna_df
import sklearn.linear_model
import mord
from time import time
from Utils import STATE_CODES


class DataCleanerACS:

    def __init__(self, st, yr, fp_h, fp_p, fp_out, state_of_work, random_state, adjinc=1056030):
        '''
        :param st: state name, e.g.'ma'
        :param yr: end year of 5-year ACS, e.g. 2016
        :param fp_h: file path to ACS 5-year state household data, excl. CSV file name
        :param fp_p: file path to ACS 5-year state person data, excl. CSV file name
        :param fp_out: file path for output, excl. CSV file name
        :param state_of_work: if True then use POW data files, if False use state-of-residence files (original ACS)
        :param adjinc: ACS income adjustment factor (for now adjust to 2012 $ for FMLA 2012)
        '''

        self.st = st
        self.yr = yr
        self.fp_h = fp_h
        self.fp_p = fp_p
        self.fp_out = fp_out
        self.state_of_work = state_of_work
        self.random_state = random_state
        self.adjinc = adjinc
        # a dict from st to state code (e.g. 'ri' to 44, 'ca' to 6)
        self.dct_st = dict(zip(
            [x.lower() for x in
             ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
              "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
              "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]
             ],
            ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19", "20", "21", "22",
             "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
             "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"]
                               ))

    def load_data(self, st):
        '''
        load and prepare other data needed for cleaning up large ACS 5-year person file
        :return: prepared ACS household data, and ready-for-sim CPS file
        '''

        # Load ACS household data and create some variables
        fp_d_hh = self.fp_h + '/ss%sh%s.csv' % (str(self.yr)[-2:], st)
        if self.state_of_work:
            fp_d_hh = self.fp_h + '/h%s_%s_pow.csv' % (self.dct_st[st], st)
        d_hh = pd.read_csv(fp_d_hh)
        d_hh["nochildren"] = pd.get_dummies(d_hh["FPARC"])[4]
        d_hh['faminc'] = d_hh['FINCP'] * d_hh['ADJINC'] / self.adjinc / 1  # adjust to 2012 dollars to conform with FMLA 2012 data
        d_hh.loc[(d_hh['faminc'] <= 0), 'faminc'] = 0.01 / 1  # force non-positive income to be epsilon to get meaningful log-income
        d_hh["lnfaminc"] = np.log(d_hh["faminc"])
        # Number of dependents
        d_hh['ndep_kid'] = d_hh['NOC']
        d_hh['ndep_old'] = d_hh['R65']
        d_hh['noelderly'] = np.where(d_hh['ndep_old']==0, 1, 0)
        d_hh['noelderly'] = np.where(d_hh['ndep_old'].isna(), np.nan, d_hh['noelderly'])
        d_hh['ndep_spouse'] = np.where(d_hh['FES'].isin([2,3]), 1, 0)
        d_hh['ndep_spouse_kid'] = d_hh['ndep_kid'] + d_hh['ndep_spouse']
        return d_hh[['SERIALNO', 'NPF', 'nochildren', 'noelderly', 'faminc', 'lnfaminc', 'PARTNER', 'ndep_kid', 'ndep_old',
                     'ndep_spouse', 'ndep_spouse_kid']]

    def clean_person_state_data(self, st, cps, chunk_size=100000):
        ## First handle CPS data outside the ACS person data chunk loop
        # identify industry and occupation CPS codes for creating same cats in ACS in chunk loop
        top_ind = [cps[cps.ind_top1==1].a_mjind.value_counts().index[0]]
        top_ind += [cps[cps.ind_top2==1].a_mjind.value_counts().index[0]]
        top_ind += [cps[cps.ind_top3==1].a_mjind.value_counts().index[0]]
        top_occ = [cps[cps.occ_top1==1].a_mjocc.value_counts().index[0]]
        top_occ += [cps[cps.occ_top2==1].a_mjocc.value_counts().index[0]]
        top_occ += [cps[cps.occ_top3==1].a_mjocc.value_counts().index[0]]

        # use cps df to train models for predicting needed yvars in ACS person data chunks
        # fillna for cps
        cps = fillna_df(cps, self.random_state)
        ## for faster testing
        # cps = cps.dropna(how='any')

        # weight col
        w = cps['marsupwt']
        # dict from yvar to impute to xvars
        xvars_in_acs = ['female', 'black', 'asian', 'native', 'other', 'age', 'agesq', 'BA', 'GradSch', 'married']
        xvars_in_acs += ['wage12', 'wkswork', 'wkhours', 'emp_gov']
        xvars_in_acs += ['ind_top1', 'ind_top2', 'ind_top3', 'ind_other', 'ind_na']
        xvars_in_acs += ['occ_top1', 'occ_top2', 'occ_top3', 'occ_other', 'occ_na']
        dct_cps_yx = {
            'hourly': xvars_in_acs,
            'empsize': xvars_in_acs,
            'oneemp':xvars_in_acs,
            'union': xvars_in_acs + ['hourly', 'empsize', 'oneemp']
           }
        # dict to store CPS-based classifiers
        dct_cps_clfs = {}
        # empsize - mord
        y = cps['empsize']
        X = cps[dct_cps_yx['empsize']]
        clf = mord.LogisticAT().fit(X, y)
        dct_cps_clfs['empsize'] = clf
        # paid hourly, oneemp, union - logit
        for col_y in ['hourly', 'oneemp', 'union']:
            y = cps[col_y]
            X = cps[dct_cps_yx[col_y]]
            clf = sklearn.linear_model.LogisticRegression(solver='liblinear').fit(X, y, sample_weight=w)
            dct_cps_clfs[col_y] = clf

        ## Get household data for merging to person data chunks
        d_hh = self.load_data(st)

        ## Work on ACS person data in chunks
        # initiate output master person data to store cleaned up chunks
        dout = pd.DataFrame([])

        # clean person data by chunks
        ichunk = 1
        print('Cleaning ACS data. State chosen = %s. Chunk size = %s ACS rows' % (st.upper(), chunk_size))

        # set file path to person file, per state_of_work value
        fp_d_p = self.fp_p + "/ss%sp%s.csv" % (str(self.yr)[-2:], st)
        if self.state_of_work:
            fp_d_p = self.fp_p + '/p%s_%s_pow.csv' % (self.dct_st[st], st)

        # process person data by chunk
        for d in pd.read_csv(fp_d_p, chunksize=chunk_size):
            # Merge with the household level variables
            d = pd.merge(d, d_hh, on='SERIALNO')

            # -------------------------- #
            # Remove ineligible workers, include only
            # civilian employed (ESR=1/2), and
            # have paid work (COW=1/2/3/4/5), including self-employed(COW=6/7)
            # -------------------------- #
            d = d[((d['ESR'] == 1) | (d['ESR'] == 2)) &
                  ((d['COW'] == 1) | (d['COW'] == 2) | (d['COW'] == 3) | (d['COW'] == 4) | (d['COW'] == 5) |
                   (d['COW'] == 6) | (d['COW'] == 7))]

            # Rename ACS variables to be consistent with FMLA data
            rename_dic = {'AGEP': 'age'}
            d.rename(columns=rename_dic, inplace=True)

            # duplicating age column for meshing with CPS estimate output
            d['a_age'] = d['age']

            # Create new ACS Variables
            mar_dummies = pd.get_dummies(d["MAR"])
            d['married'] = mar_dummies[1]
            d["widowed"] = mar_dummies[2]
            d["divorced"] = mar_dummies[3]
            d["separated"] = mar_dummies[4]
            d["nevermarried"] = mar_dummies[5]
            # use PARTNER in household data to tease out unmarried partners
            d['partner'] = np.where(
                (d['PARTNER'] == 1) | (d['PARTNER'] == 2) | (d['PARTNER'] == 3) | (d['PARTNER'] == 4), 1, 0)
            for m in ['married', 'widowed', 'divorced', 'separated', 'nevermarried']:
                d.loc[d['partner'] == 1, m] = 0

            d["male"] = pd.get_dummies(d["SEX"])[1]
            d["female"] = 1 - d["male"]
            d["agesq"] = d["age"] ** 2

            # Educational level
            d['sku'] = np.where(d['SCHL'].isna(), 0, d['SCHL'])
            d['ltHS'] = np.where(d['sku'] <= 15, 1, 0)
            d['HSgrad'] = np.where((d['sku'] >= 16) & (d['sku'] <= 17), 1, 0)
            d['someCol'] = np.where((d['sku'] >= 18) & (d['sku'] <= 20), 1, 0)
            d["BA"] = np.where(d['sku'] == 21, 1, 0)
            d["GradSch"] = np.where(d['sku'] >= 22, 1, 0)

            d["noHSdegree"] = np.where(d['sku'] <= 15, 1, 0)
            d["BAplus"] = np.where(d['sku'] >= 21, 1, 0)
            # variables for imputing hourly status, using CPS estimates from original model
            d["maplus"] = np.where(d['sku'] >= 22, 1, 0)
            d["ba"] = d['BA']

            # race
            d['hisp'] = np.where(d['HISP'] >= 2, 1, 0)
            d['white'] = np.where((d['hisp'] == 0) & (d['RAC1P'] == 1), 1, 0)
            d['black'] = np.where((d['hisp'] == 0) & (d['RAC1P'] == 2), 1, 0)
            d['asian'] = np.where((d['hisp'] == 0) & (d['RAC1P'] == 6), 1, 0)
            d['native'] = np.where((d['hisp'] == 0) & ((d['RAC1P'] == 3) |
                                                       (d['RAC1P'] == 4) |
                                                       (d['RAC1P'] == 5) |
                                                       (d['RAC1P'] == 7)), 1, 0)
            d['other'] = np.where(
                (d['hisp'] == 0) & (d['white'] == 0) & (d['black'] == 0) & (d['asian'] == 0) & (d['native'] == 0), 1, 0)

            # Employed
            d['employed'] = np.where((d['ESR'] == 1) |
                                     (d['ESR'] == 2) |
                                     (d['ESR'] == 4) |
                                     (d['ESR'] == 5),
                                     1, 0)
            d['employed'] = np.where(np.isnan(d['ESR']), np.nan, d['employed'])

            # Hours per week, total working weeks
            d['wkhours'] = d['WKHP']
            # Weeks worked - average of categories
            dict_wks = {
                1: 51,
                2: 48.5,
                3: 43.5,
                4: 33,
                5: 20,
                6: 7
            }
            d['wkswork'] = d['WKW'].map(dict_wks)
            d['wkswork'] = np.where(d['wkswork'].isna(), 0, d['wkswork'])
            # Weeks worked - WKW category bounds, for bounding imputed weeks worked from CPS
            dict_wkwBounds = {
                1: (50, 52),
                2: (48, 49),
                3: (40, 47),
                4: (27, 39),
                5: (14, 26),
                6: (1, 13)
            }
            # d['wkw_min'] = d['WKW'].apply(lambda x: dict_wkwBounds[x][0] if not np.isnan(x) else np.nan)
            # d['wkw_max'] = d['WKW'].apply(lambda x: dict_wkwBounds[x][1] if not np.isnan(x) else np.nan)
            d['wkw_min'] = [dict_wkwBounds[x][0] if not np.isnan(x) else np.nan for x in d['WKW']]
            d['wkw_max'] = [dict_wkwBounds[x][1] if not np.isnan(x) else np.nan for x in d['WKW']]

            # Total wage past 12m, adjusted to 2012, and its log
            d['wage12'] = d['WAGP'] * d['ADJINC'] / self.adjinc
            d['lnearn'] = np.nan
            # d.loc[d['wage12'] > 0, 'lnearn'] = d.loc[d['wage12'] > 0, 'wage12'].apply(lambda x: np.log(x))
            d.loc[d['wage12'] > 0, 'lnearn'] = [np.log(x) for x in d.loc[d['wage12'] > 0, 'wage12']]

            # health insurance from employer
            d['hiemp'] = np.where(d['HINS1'] == 1, 1, 0)
            d['hiemp'] = np.where(d['HINS1'].isna(), np.nan, d['hiemp'])

            # Employment at government
            # missing = age<16, or NILF over 5 years, or never worked
            d['emp_gov'] = np.where(d['COW'].isin([3,4,5]), 1, 0)
            d['emp_gov'] = np.where(d['COW'].isna(), np.nan, d['emp_gov'])
            d['empgov_fed'] = np.where(d['COW'] == 5, 1, 0)
            d['empgov_fed'] = np.where(np.isnan(d['COW']), np.nan, d['empgov_fed'])
            d['empgov_st'] = np.where(d['COW'] == 4, 1, 0)
            d['empgov_st'] = np.where(np.isnan(d['COW']), np.nan, d['empgov_st'])
            d['empgov_loc'] = np.where(d['COW'] == 3, 1, 0)
            d['empgov_loc'] = np.where(np.isnan(d['COW']), np.nan, d['empgov_loc'])

            # Employment at non-profit
            d['emp_nonprofit'] = np.where(d['COW']==2, 1, 0)
            d['emp_nonprofit'] = np.where(d['COW'].isna(), np.nan, d['emp_nonprofit'])

            # Presence of children for females
            d['fem_cu6'] = np.where(d['PAOC'] == 1, 1, 0)
            d['fem_c617'] = np.where(d['PAOC'] == 2, 1, 0)
            d['fem_cu6and617'] = np.where(d['PAOC'] == 3, 1, 0)
            d['fem_nochild'] = np.where(d['PAOC'] == 4, 1, 0)
            for x in ['fem_cu6', 'fem_c617', 'fem_cu6and617', 'fem_nochild']:
                d.loc[d['PAOC'].isna(), x] = np.nan

            # Occupation
            # use numeric OCCP = OCCP12 if ACS 2011-2015, or OCCP = OCCP if ACS 2012-2016
            if self.yr == 2015:
                if 'N.A.' in d['OCCP12'].value_counts().index:
                    d.loc[d['OCCP12'] == 'N.A.', 'OCCP12'] = np.nan
                d.loc[d['OCCP12'].notna(), 'OCCP12'] = [int(x) for x in d.loc[d['OCCP12'].notna(), 'OCCP12']]

                if 'N.A.' in d['OCCP10'].value_counts().index:
                    d.loc[d['OCCP10'] == 'N.A.', 'OCCP10'] = np.nan
                d.loc[d['OCCP10'].notna(), 'OCCP10'] = [int(x) for x in d.loc[d['OCCP10'].notna(), 'OCCP10']]

                d['OCCP'] = np.nan
                d['OCCP'] = np.where(d['OCCP12'].notna(), d['OCCP12'], d['OCCP'])
                d['OCCP'] = np.where((d['OCCP'].isna()) & (d['OCCP12'].isna()) & (d['OCCP10'].notna()), d['OCCP10'],
                                     d['OCCP'])
            elif self.yr == 2016:
                if 'N.A.' in d['OCCP'].value_counts().index:
                    d.loc[d['OCCP'] == 'N.A.', 'OCCP'] = np.nan
                d.loc[d['OCCP'].notna(), 'OCCP'] = [int(x) for x in d.loc[d['OCCP'].notna(), 'OCCP']]

            for c in range(1, 11):
                d['occ_%s' % c] = 0
            d.loc[(d['OCCP'] >= 10) & (d['OCCP'] <= 950), 'occ_1'] = 1
            d.loc[(d['OCCP'] >= 1000) & (d['OCCP'] <= 3540), 'occ_2'] = 1
            d.loc[(d['OCCP'] >= 3600) & (d['OCCP'] <= 4650), 'occ_3'] = 1
            d.loc[(d['OCCP'] >= 4700) & (d['OCCP'] <= 4965), 'occ_4'] = 1
            d.loc[(d['OCCP'] >= 5000) & (d['OCCP'] <= 5940), 'occ_5'] = 1
            d.loc[(d['OCCP'] >= 6000) & (d['OCCP'] <= 6130), 'occ_6'] = 1
            d.loc[(d['OCCP'] >= 6200) & (d['OCCP'] <= 6940), 'occ_7'] = 1
            d.loc[(d['OCCP'] >= 7000) & (d['OCCP'] <= 7630), 'occ_8'] = 1
            d.loc[(d['OCCP'] >= 7700) & (d['OCCP'] <= 8965), 'occ_9'] = 1
            d.loc[(d['OCCP'] >= 9000) & (d['OCCP'] <= 9750), 'occ_10'] = 1
            # make sure occ_x gets nan if OCCP code is nan
            for x in range(1, 11):
                d.loc[d['OCCP'].isna(), 'occ_%s' % x] = np.nan
            # set maj_occ in 1 col
            d['maj_occ'] = 0
            for x in range(1, 11):
                d.loc[d['occ_%s' % x]==1, 'maj_occ'] = x
            # get cols of top-3 occ codes, other, NA
            for rank in range(1, 4):
                d['occ_top%s' % rank] = np.where(d['occ_%s' % top_occ[rank-1]]==1, 1, 0)
            d['occ_other'] = np.where((~d['maj_occ'].isin(top_occ)), 1, 0)
            d['occ_na'] = np.where(d['maj_occ'].isna(), 1, 0)
            
            # Industry
            for c in range(1, 14):
                d['ind_%s' % c] = 0
            d.loc[(d['INDP'] >= 170) & (d['INDP'] <= 290), 'ind_1'] = 1
            d.loc[(d['INDP'] >= 370) & (d['INDP'] <= 490), 'ind_2'] = 1
            d.loc[(d['INDP'] == 770), 'ind_3'] = 1
            d.loc[(d['INDP'] >= 1070) & (d['INDP'] <= 3990), 'ind_4'] = 1
            d.loc[(d['INDP'] >= 4070) & (d['INDP'] <= 5790), 'ind_5'] = 1
            d.loc[((d['INDP'] >= 6070) & (d['INDP'] <= 6390)) | ((d['INDP'] >= 570) & (d['INDP'] <= 690)), 'ind_6'] = 1
            d.loc[(d['INDP'] >= 6470) & (d['INDP'] <= 6780), 'ind_7'] = 1
            d.loc[(d['INDP'] >= 6870) & (d['INDP'] <= 7190), 'ind_8'] = 1
            d.loc[(d['INDP'] >= 7270) & (d['INDP'] <= 7790), 'ind_9'] = 1
            d.loc[(d['INDP'] >= 7860) & (d['INDP'] <= 8470), 'ind_10'] = 1
            d.loc[(d['INDP'] >= 8560) & (d['INDP'] <= 8690), 'ind_11'] = 1
            d.loc[(d['INDP'] >= 8770) & (d['INDP'] <= 9290), 'ind_12'] = 1
            d.loc[(d['INDP'] >= 9370) & (d['INDP'] <= 9590), 'ind_13'] = 1
            # make sure ind_x gets nan if INDP code is nan
            for x in range(1, 14):
                d.loc[d['INDP'].isna(), 'ind_%s' % x] = np.nan
            # set maj_ind in 1 col
            d['maj_ind'] = 0
            for x in range(1, 14):
                d.loc[d['ind_%s' % x]==1, 'maj_ind'] = x

            # get cols of top-3 ind codes, other, NA
            for rank in range(1, 4):
                d['ind_top%s' % rank] = np.where(d['ind_%s' % top_ind[rank - 1]] == 1, 1, 0)
            d['ind_other'] = np.where((~d['maj_ind'].isin(top_ind)), 1, 0)
            d['ind_na'] = np.where(d['maj_ind'].isna(), 1, 0)

            # -------------------------- #
            # impute ACS vars using CPS-based classifiers built before the chunk loop
            # -------------------------- #
            Xd = fillna_df(d[xvars_in_acs], self.random_state)
            # paid hourly, oneemp, empsize, union
            for c in ['hourly', 'oneemp', 'empsize', 'union']: # imputing union needs previous vars, set union to end
                clf = dct_cps_clfs[c]
                if c=='union':
                    Xd = Xd.join(d[['hourly', 'oneemp', 'empsize']])
                d[c] = pd.Series(clf.predict(Xd), index=d.index)

            # get fmla_eligible col based on imputed vars above
            d['fmla_eligible'] = np.where((d['oneemp'] == 1) &
                                          (d['wkhours'] * d['wkswork'] >= 1250) &
                                          (d['empsize'] >= 3), 1, 0)
            print('fmla_eligible value count \n', d['fmla_eligible'].value_counts())
            # -------------------------- #
            # Save the resulting dataset
            # -------------------------- #
            cols = ['SERIALNO', 'SPORDER', 'PWGTP', 'ST', 'POWSP', 'NPF',
                    'employed', 'emp_gov', 'emp_nonprofit', 'empgov_fed', 'empgov_st', 'empgov_loc',
                    'wkhours', 'wkswork', 'wage12', 'lnearn', 'hiemp',
                    'a_age', 'age', 'agesq',
                    'male', 'female',
                    'nochildren', 'noelderly', 'ndep_kid', 'ndep_old', 'ndep_spouse', 'ndep_spouse_kid',
                    'ltHS', 'HSgrad', 'someCol', 'BA', 'GradSch', 'noHSdegree', 'BAplus',
                    'faminc', 'lnfaminc',
                    'married', 'partner', 'separated', 'divorced', 'widowed', 'nevermarried',
                    'asian', 'black', 'white', 'native', 'other', 'hisp',
                    'fem_cu6', 'fem_cu6and617', 'fem_c617', 'fem_nochild',
                    'ESR', 'COW'  # original ACS vars for restricting samples later
                    ]
            cols += ['INDP'] + ['ind_%s' % x for x in range(1, 14)]
            cols += ['OCCP'] + ['occ_%s' % x for x in range(1, 11)]
            cols += ['hourly', 'oneemp', 'wkswork', 'empsize', 'fmla_eligible', 'union']
            cols += ['WKW', 'wkw_min', 'wkw_max']
            cols += ['PWGTP%s' % x for x in range(1, 81)]

            d_reduced = d[cols]
            dout = dout.append(d_reduced)
            print('ACS data cleaned for chunk %s of person data...' % ichunk)
            ichunk += 1

        return dout

    def clean_person_data(self, chunk_size=100000):
        '''
        clean large ACS 5-year person file
        :return:
        '''
        t0 = time()

        # Load CPS data from impute_FMLA_CPS
        cps = pd.read_csv('./data/cps/cps_for_acs_sim.csv')
        if self.st.lower() == 'all':
            for i, st in enumerate(STATE_CODES):
                dout = self.clean_person_state_data(st.lower(), cps, chunk_size=chunk_size)
                if i == 0:
                    dout.to_csv(self.fp_out + "ACS_cleaned_forsimulation_%s_%s.csv" % (self.yr, self.st), index=False,
                                header=True)
                else:
                    dout.to_csv(self.fp_out + "ACS_cleaned_forsimulation_%s_%s.csv" % (self.yr, self.st), index=False,
                                mode='a', header=False)
        else:
            dout = self.clean_person_state_data(self.st, cps, chunk_size=chunk_size)
            dout.to_csv(self.fp_out + "ACS_cleaned_forsimulation_%s_%s.csv" % (self.yr, self.st), index=False,
                        header=True)

        t1 = time()
        message = 'ACS data cleaning finished for state %s. Time elapsed = %s seconds' % (self.st.upper(), round((t1 - t0), 0))
        print(message)
        return message

# # -------------------------- #
# # ACS Household File
# # -------------------------- #
#
# # Load data
# st = 'nj'
# yr = 16
# adjinc_2012 = 1022342 # to conform with 2012 dollars in FMLA
# d_hh = pd.read_csv("C:/workfiles/Microsimulation/git/large_data_files/ss%sh%s.csv" % (yr, st))
#
# # Create Variables
# d_hh["nochildren"]  = pd.get_dummies(d_hh["FPARC"])[4]
# d_hh['faminc'] = d_hh['FINCP']*d_hh['ADJINC'] / adjinc_2012 / 1000 # adjust to 2012 thousand dollars to conform with FMLA 2012 data
# d_hh.loc[(d_hh['faminc']<=0), 'faminc'] = 0.01/1000 # force non-positive income to be epsilon to get meaningful log-income
# d_hh["lnfaminc"]    = np.log(d_hh["faminc"])
#
# # Number of dependents
# d_hh['ndep_kid'] = d_hh.NOC
# d_hh['ndep_old'] = d_hh.R65
#
# # -------------------------- #
# # CPS - read in first for estimation/imputation during handling each chunk of ACS personal file
# # -------------------------- #
# cps = pd.read_csv('./data/cps/cps_for_acs_sim.csv')

# -------------------------- #
# ACS Personal File
# -------------------------- #

# chunk_size = 100000
# dout = pd.DataFrame([])
# ichunk = 1

# clean ACS data in chunks
# print('Cleaning ACS data. State chosen = %s' % st)
# for d in pd.read_csv("C:/workfiles/Microsimulation/git/large_data_files/ss%sp%s.csv" % (yr, st), chunksize=chunk_size):
#
#     # Merge with the household level variables
#     d = pd.merge(d,d_hh[['SERIALNO', 'nochildren', 'faminc','lnfaminc','PARTNER', 'ndep_kid', 'ndep_old']],
#                      on='SERIALNO')
#
#     # -------------------------- #
#     # Remove ineligible workers, include only
#     # civilian employed (ESR=1/2), and
#     # have paid work (COW=1/2/3/4/5), including self-employed(COW=6/7)
#     # -------------------------- #
#     d = d[((d['ESR']==1) | (d['ESR']==2)) &
#           ((d['COW']==1) | (d['COW']==2) | (d['COW']==3) | (d['COW']==4) | (d['COW']==5) |
#            (d['COW']==6) | (d['COW']==7))]
#
#     # Rename ACS variables to be consistent with FMLA data
#     rename_dic = {'AGEP': 'age'}
#     d.rename(columns=rename_dic, inplace=True)
#
#     # duplicating age column for meshing with CPS estimate output
#     d['a_age']=d['age']
#
#     # Create new ACS Variables
#     d['married'] = pd.get_dummies(d["MAR"])[1]
#     d["widowed"]        = pd.get_dummies(d["MAR"])[2]
#     d["divorced"]       = pd.get_dummies(d["MAR"])[3]
#     d["separated"]      = pd.get_dummies(d["MAR"])[4]
#     d["nevermarried"]   = pd.get_dummies(d["MAR"])[5]
#         # use PARTNER in household data to tease out unmarried partners
#     d['partner'] = np.where((d['PARTNER']==1) | (d['PARTNER']==2) | (d['PARTNER']==3) | (d['PARTNER']==4), 1, 0)
#     for m in ['married', 'widowed', 'divorced', 'separated', 'nevermarried']:
#         d.loc[d['partner']==1, m] = 0
#
#     d["male"]           = pd.get_dummies(d["SEX"])[1]
#     d["female"]         = 1 - d["male"]
#     d["agesq"]          = d["age"]**2
#
#     # Educational level
#     d['sku'] = np.where(d['SCHL'].isna(), 0, d['SCHL'])
#     d['ltHS']    = np.where(d['sku']<=11,1,0)
#     d['someHS']  = np.where((d['sku']>=12) & (d['sku']<=15),1,0)
#     d['HSgrad']  = np.where((d['sku']>=16) & (d['sku']<=17),1,0)
#     d['someCol']  = np.where((d['sku']>=18) & (d['sku']<=20),1,0)
#     d["BA"]      = np.where(d['sku']==21,1,0)
#     d["GradSch"]  = np.where(d['sku']>=22,1,0)
#
#     d["noHSdegree"]  = np.where(d['sku']<=15,1,0)
#     d["BAplus"]  = np.where(d['sku']>=21,1,0)
#         # variables for imputing hourly status, using CPS estimates from original model
#     d["maplus"]  = np.where(d['sku']>=22,1,0)
#     d["ba"]      = d['BA']
#
#     # race
#     d['hisp'] = np.where(d['HISP']>=2, 1, 0)
#     d['white'] = np.where((d['hisp']==0) & (d['RAC1P']==1), 1, 0)
#     d['black'] = np.where((d['hisp']==0) & (d['RAC1P']==2), 1, 0)
#     d['asian'] = np.where((d['hisp']==0) & (d['RAC1P']==6), 1, 0)
#     d['native'] = np.where((d['hisp'] == 0) & ((d['RAC1P'] == 3) |
#                                                (d['RAC1P'] == 4) |
#                                                (d['RAC1P'] == 5) |
#                                                (d['RAC1P'] == 7)), 1, 0)
#     d['other'] = np.where((d['hisp']==0) & (d['white']==0) & (d['black']==0) & (d['asian']==0) & (d['native']==0), 1, 0)
#
#     # Employed
#     d['employed'] = np.where((d['ESR']== 1) |
#                              (d['ESR'] == 2) |
#                              (d['ESR'] == 4) |
#                              (d['ESR'] == 5) ,
#                              1, 0)
#     d['employed'] = np.where(np.isnan(d['ESR']),np.nan,d['employed'])
#
#     # Hours per week, total working weeks
#     d['wkhours'] = d['WKHP']
#     # Weeks worked - average of categories
#     dict_wks = {
#         1: 51,
#         2: 48.5,
#         3: 43.5,
#         4: 33,
#         5: 20,
#         6: 7
#     }
#     d['weeks_worked_cat'] = d['WKW'].map(dict_wks)
#     d['weeks_worked_cat'] = np.where(d['weeks_worked_cat'].isna(), 0, d['weeks_worked_cat'])
#     # Weeks worked - WKW category bounds, for bounding imputed weeks worked from CPS
#     dict_wkwBounds ={
#         1: (50, 52),
#         2: (48, 49),
#         3: (40, 47),
#         4: (27, 39),
#         5: (14, 26),
#         6: (1, 13)
#     }
#     d['wkw_min'] = d['WKW'].apply(lambda x: dict_wkwBounds[x][0] if not np.isnan(x) else np.nan)
#     d['wkw_max'] = d['WKW'].apply(lambda x: dict_wkwBounds[x][1] if not np.isnan(x) else np.nan)
#     # Total wage past 12m, adjusted to 2012, and its log
#     d['wage12'] = d['WAGP'] *d['ADJINC'] / adjinc_2012
#     d['lnearn'] = np.nan
#     d.loc[d['wage12']>0, 'lnearn'] = d.loc[d['wage12']>0, 'wage12'].apply(lambda x: np.log(x))
#
#     # health insurance from employer
#     d['hiemp'] = np.where(d['HINS1']==1, 1, 0)
#     d['hiemp'] = np.where(d['HINS1'].isna(), np.nan, d['hiemp'])
#
#     # Employment at government
#         # missing = age<16, or NILF over 5 years, or never worked
#     d['empgov_fed'] = np.where(d['COW']==5, 1, 0)
#     d['empgov_fed'] = np.where(np.isnan(d['COW']),np.nan,d['empgov_fed'])
#     d['empgov_st'] = np.where(d['COW']==4, 1, 0)
#     d['empgov_st'] = np.where(np.isnan(d['COW']),np.nan,d['empgov_st'])
#     d['empgov_loc'] = np.where(d['COW']==3, 1, 0)
#     d['empgov_loc'] = np.where(np.isnan(d['COW']),np.nan,d['empgov_loc'])
#
#     # Presence of children for females
#     d['fem_cu6'] = np.where(d['PAOC']==1, 1, 0)
#     d['fem_c617'] = np.where(d['PAOC']==2, 1, 0)
#     d['fem_cu6and617'] = np.where(d['PAOC']==3, 1, 0)
#     d['fem_nochild'] = np.where(d['PAOC']==4, 1, 0)
#     for x in ['fem_cu6','fem_c617','fem_cu6and617','fem_nochild']:
#         d.loc[d['PAOC'].isna(), x] = np.nan
#
#     # Occupation
#
#     # use numeric OCCP = OCCP12 if ACS 2011-2015, or OCCP = OCCP if ACS 2012-2016
#     if yr==15:
#         if 'N.A.' in d['OCCP12'].value_counts().index:
#             d.loc[d['OCCP12']=='N.A.', 'OCCP12'] = d.loc[d['OCCP12']=='N.A.', 'OCCP12'].apply(lambda x: np.nan)
#         d.loc[d['OCCP12'].notna(), 'OCCP12'] = d.loc[d['OCCP12'].notna(), 'OCCP12'].apply(lambda x: int(x))
#
#         if 'N.A.' in d['OCCP10'].value_counts().index:
#             d.loc[d['OCCP10']=='N.A.', 'OCCP10'] = d.loc[d['OCCP10']=='N.A.', 'OCCP10'].apply(lambda x: np.nan)
#         d.loc[d['OCCP10'].notna(), 'OCCP10'] = d.loc[d['OCCP10'].notna(), 'OCCP10'].apply(lambda x: int(x))
#
#         d['OCCP'] = np.nan
#         d['OCCP'] = np.where(d['OCCP12'].notna(), d['OCCP12'], d['OCCP'])
#         d['OCCP'] = np.where((d['OCCP'].isna()) & (d['OCCP12'].isna()) & (d['OCCP10'].notna()), d['OCCP10'], d['OCCP'])
#     elif yr==16:
#         if 'N.A.' in d['OCCP'].value_counts().index:
#             d.loc[d['OCCP']=='N.A.', 'OCCP'] = d.loc[d['OCCP']=='N.A.', 'OCCP'].apply(lambda x: np.nan)
#         d.loc[d['OCCP'].notna(), 'OCCP'] = d.loc[d['OCCP'].notna(), 'OCCP'].apply(lambda x: int(x))
#
#
#     d['occ_1']=0
#     d['occ_2']=0
#     d['occ_3']=0
#     d['occ_4']=0
#     d['occ_5']=0
#     d['occ_6']=0
#     d['occ_7']=0
#     d['occ_8']=0
#     d['occ_9']=0
#     d['occ_10']=0
#     d['maj_occ']=0
#     d.loc[(d['OCCP']>=10) & (d['OCCP']<=950), 'occ_1'] =1
#     d.loc[(d['OCCP']>=1000) & (d['OCCP']<=3540), 'occ_2'] =1
#     d.loc[(d['OCCP']>=3600) & (d['OCCP']<=4650), 'occ_3'] =1
#     d.loc[(d['OCCP']>=4700) & (d['OCCP']<=4965), 'occ_4'] =1
#     d.loc[(d['OCCP']>=5000) & (d['OCCP']<=5940), 'occ_5'] =1
#     d.loc[(d['OCCP']>=6000) & (d['OCCP']<=6130), 'occ_6'] =1
#     d.loc[(d['OCCP']>=6200) & (d['OCCP']<=6940), 'occ_7'] =1
#     d.loc[(d['OCCP']>=7000) & (d['OCCP']<=7630), 'occ_8'] =1
#     d.loc[(d['OCCP']>=7700) & (d['OCCP']<=8965), 'occ_9'] =1
#     d.loc[(d['OCCP']>=9000) & (d['OCCP']<=9750), 'occ_10'] =1
#         # make sure occ_x gets nan if OCCP code is nan
#     for x in range(1, 11):
#         d.loc[d['OCCP'].isna(), 'occ_%s' % x] = np.nan
#
#     # Industry
#     d['ind_1']=0
#     d['ind_2']=0
#     d['ind_3']=0
#     d['ind_4']=0
#     d['ind_5']=0
#     d['ind_6']=0
#     d['ind_7']=0
#     d['ind_8']=0
#     d['ind_9']=0
#     d['ind_10']=0
#     d['ind_11']=0
#     d['ind_12']=0
#     d['ind_13']=0
#     d.loc[(d['INDP']>=170) & (d['INDP']<=290), 'ind_1'] =1
#     d.loc[(d['INDP']>=370) & (d['INDP']<=490), 'ind_2'] =1
#     d.loc[(d['INDP']==770), 'ind_3'] =1
#     d.loc[(d['INDP']>=1070) & (d['INDP']<=3990), 'ind_4'] =1
#     d.loc[(d['INDP']>=4070) & (d['INDP']<=5790), 'ind_5'] =1
#     d.loc[((d['INDP']>=6070) & (d['INDP']<=6390))|((d['INDP']>=570) & (d['INDP']<=690)), 'ind_6'] =1
#     d.loc[(d['INDP']>=6470) & (d['INDP']<=6780), 'ind_7'] =1
#     d.loc[(d['INDP']>=6870) & (d['INDP']<=7190), 'ind_8'] =1
#     d.loc[(d['INDP']>=7270) & (d['INDP']<=7790), 'ind_9'] =1
#     d.loc[(d['INDP']>=7860) & (d['INDP']<=8470), 'ind_10'] =1
#     d.loc[(d['INDP']>=8560) & (d['INDP']<=8690), 'ind_11'] =1
#     d.loc[(d['INDP']>=8770) & (d['INDP']<=9290), 'ind_12'] =1
#     d.loc[(d['INDP']>=9370) & (d['INDP']<=9590), 'ind_13'] =1
#         # make sure ind_x gets nan if INDP code is nan
#     for x in range(1, 14):
#         d.loc[d['INDP'].isna(), 'ind_%s' % x] = np.nan
#
#     # -------------------------- #
#     # CPS Imputation
#     # -------------------------- #
#     # predictors and weights
#     X = cps[['female', 'black', 'age', 'agesq', 'BA', 'GradSch'] +
#             ['ind_%s' % x for x in range(1, 14)] +
#             ['occ_%s' % x for x in range(1, 11)]]
#     w = cps['marsupwt']
#     #Xd = fillna_binary(d[X.columns])
#     Xd = fillna_df(d[X.columns])
#     # paid hourly
#     y = cps['hourly']
#     clf = sklearn.linear_model.LogisticRegression(solver='liblinear').fit(X, y, sample_weight=w)
#     d['hourly'] = pd.Series(clf.predict(Xd), index=d.index)
#     # one employer
#     y = cps['oneemp']
#     clf = sklearn.linear_model.LogisticRegression(solver='liblinear').fit(X, y, sample_weight=w)
#     d['oneemp'] = pd.Series(clf.predict(Xd), index=d.index)
#     # weeks worked - impose condition of categorical WKW from ACS itself
#     y = cps['wkswork']
#     clf = sklearn.linear_model.LinearRegression().fit(X, y, sample_weight=w)
#     d['wkswork_dec'] = pd.Series(clf.predict(Xd), index=d.index)
#     d['wkswork'] = d[['wkswork_dec', 'wkw_min', 'wkw_max']].apply(lambda x: min(max(int(x[0]), x[1]), x[2]),axis=1)
#     # employer size
#     y = cps['empsize']
#     clf = mord.LogisticAT().fit(X, y)
#     d['empsize'] = pd.Series(clf.predict(Xd), index=d.index)
#
#     # -------------------------- #
#     # Save the resulting dataset
#     # -------------------------- #
#     cols = ['SERIALNO', 'PWGTP', 'ST',
#     'employed', 'empgov_fed','empgov_st','empgov_loc',
#     'wkhours', 'weeks_worked_cat', 'wage12', 'lnearn','hiemp',
#     'a_age','age', 'agesq',
#     'male','female',
#     'nochildren', 'ndep_kid', 'ndep_old',
#     'ltHS', 'someHS', 'HSgrad', 'someCol', 'BA', 'GradSch', 'noHSdegree', 'BAplus' ,
#     'faminc', 'lnfaminc',
#     'married', 'partner', 'separated', 'divorced', 'widowed', 'nevermarried',
#     'asian', 'black', 'white', 'native','other', 'hisp',
#     'fem_cu6','fem_cu6and617','fem_c617','fem_nochild',
#     'ESR', 'COW' # original ACS vars for restricting samples later
#             ]
#     cols += ['ind_%s' % x for x in range(1, 14)]
#     cols += ['OCCP'] + ['occ_%s' % x for x in range(1, 11)]
#     cols += ['hourly', 'oneemp', 'wkswork', 'empsize']
#     cols += ['WKW', 'wkswork_dec','wkw_min', 'wkw_max']
#     cols += ['PWGTP%s' % x for x in range(1, 81)]
#
#     d_reduced = d[cols]
#     dout = dout.append(d_reduced)
#     print('ACS data cleaned for chunk %s of personal data...' % ichunk)
#     ichunk += 1

# dout.to_csv("./data/acs/ACS_cleaned_forsimulation_20%s_%s.csv" % (yr, st), index=False, header=True)
#
# t1 = time()
# print('ACS data cleaning finished for state %s. Time elapsed = %s' % (st.upper(), (t1-t0)))

##