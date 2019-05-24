"""
This program takes the FMLA data and cleans it into a format to be used
for behavioral estimation
"""

import pandas as pd
import numpy as np
import sklearn.linear_model
import mord
import json
from time import time
from _5a_aux_functions import simulate_wof, get_marginal_probs, get_adj_cps, get_adj_ups, fillna_binary

class DataCleanerFMLA:

    def __init__(self, fp_fmla_in, fp_fmla_out):
        '''

        :param fp_fmla_in: file path to original FMLA data
        :param fp_fmla_out: file path to cleaned FMLA data, before CPS simulation
        '''

        self.fp_fmla_in = fp_fmla_in
        self.fp_fmla_out = fp_fmla_out

    def clean_data(self):

        # Read in FMLA data
        d = pd.read_csv(self.fp_fmla_in, low_memory=False)

        # Make empid to follow 0-order to be consistent with Python standard (e.g. indices output from kNN)
        d['empid'] = d['empid'].apply(lambda x: x - 1)

        # FMLA eligible worker
        d['eligworker'] = np.nan
        # eligible workers
        d.loc[(d['E13'] == 1) & ((d['E14'] == 1) | ((d['E15_CAT'] >= 5) & (d['E15_CAT'] <= 8))), 'eligworker'] = 1
        # ineligible workers
        d.loc[(d['E13'].notna()) & (d['E13'] != 1), 'eligworker'] = 0  # E13 same job past year fails
        d.loc[(d['E14'].notna()) & (d['E14'] != 1)
              & (d['E15_CAT'].notna()) & (
                  (d['E15_CAT'] < 5) | (d['E15_CAT'] > 8)), 'eligworker'] = 0  # E14 (FT) and E15 (hrs) fails

        # Covered workplace
        d['covwrkplace'] = np.where((d['E11'] == 1) | ((d['E12'] >= 6) & (d['E12'] <= 9)), 1, 0)
        d['covwrkplace'] = np.where(np.isnan(d['covwrkplace']), 0, d['covwrkplace'])
        d['cond1'] = np.where(np.isnan(d['E11']) & np.isnan(d['E12']), 1, 0)
        d['cond2'] = np.where((d['E11'] == 2) & (np.isnan(d['E11']) == False) & np.isnan(d['E12']), 1, 0)
        d['miscond'] = np.where((d['cond1'] == 1) | (d['cond2'] == 1), 1, 0)
        d['covwrkplace'] = np.where(d['miscond'] == 1, np.nan, d['covwrkplace'])

        # Covered and eligible
        d['coveligd'] = np.where((d['covwrkplace'] == 1) & (d['eligworker'] == 1), 1, 0)
        d['coveligd'] = np.where(np.isnan(d['covwrkplace']) & np.isnan(d['eligworker']), np.nan, d['coveligd'])
        d['coveligd'] = np.where(np.isnan(d['covwrkplace']) & (d['eligworker'] == 1), np.nan, d['coveligd'])
        d['coveligd'] = np.where((d['covwrkplace'] == 1) & np.isnan(d['eligworker']), np.nan, d['coveligd'])

        # Hourly worker
        d['hourly'] = np.where(d['E9_1'] == 2, 1, 0)
        d['hourly'] = np.where(np.isnan(d['E9_1']), np.nan, d['hourly'])

        # Union member
        d['union'] = np.where(d['D3'] == 1, 1, 0)
        d['union'] = np.where(np.isnan(d['D3']), np.nan, d['union'])

        # Employment Status
        d['employed'] = np.where(d['E1'] == 1, 1, 0)
        d['employed'] = np.where(np.isnan(d['E1']), np.nan, d['employed'])

        # Hours per week
        # a dict to map code into average numeric hours per week
        dict_wkhours = {
            1: 4,
            2: 11.5,
            3: 17,
            4: 21.5,
            5: 26.5,
            6: 32,
            7: 37.5,
            8: 45,
            'nan': np.nan
        }
        d['wkhours'] = d['E15_CAT_REV'].map(dict_wkhours)

        # Employment at government
        # all rows should be valid for empgov_[] indicators, given FMLA sample
        d['empgov_fed'] = np.where(d['D2'] == 1, 1, 0)
        d['empgov_st'] = np.where(d['D2'] == 2, 1, 0)
        d['empgov_loc'] = np.where(d['D2'] == 3, 1, 0)

        # Age at midpoint of category
        conditions = [(d['AGE_CAT'] == 1), (d['AGE_CAT'] == 2),
                      (d['AGE_CAT'] == 3), (d['AGE_CAT'] == 4),
                      (d['AGE_CAT'] == 5), (d['AGE_CAT'] == 6),
                      (d['AGE_CAT'] == 7), (d['AGE_CAT'] == 8),
                      (d['AGE_CAT'] == 9), (d['AGE_CAT'] == 10)]
        choices = [21, 27, 32, 37, 42, 47, 52, 57, 63.5, 70]
        d['age'] = np.select(conditions, choices, default=np.nan)
        d['agesq'] = np.array(d['age']) ** 2

        # Sex
        d['male'] = np.where(d['GENDER_CAT'] == 1, 1, 0)
        d.loc[d['GENDER_CAT'].isna(), 'male'] = np.nan
        d['female'] = np.where(d['GENDER_CAT'] == 2, 1, 0)
        d.loc[d['GENDER_CAT'].isna(), 'female'] = np.nan

        # No children
        d['nochildren'] = np.where(d['D7_CAT'] == 0, 1, 0)
        d['nochildren'] = np.where(np.isnan(d['D7_CAT']), np.nan, d['nochildren'])

        # No spouse
        d['nospouse'] = np.where((d['D10'] == 3) | (d['D10'] == 4) | (d['D10'] == 5) | (d['D10'] == 6), 1, 0)
        d['nospouse'] = np.where(np.isnan(d['D10']), np.nan, d['nospouse'])

        # Number of dependents
        d['ndep_kid'] = d.D7_CAT
        d['ndep_old'] = d.D8_CAT

        # Educational level
        d['ltHS'] = np.where(d['D1_CAT'] == 1, 1, 0)
        d['ltHS'] = np.where(np.isnan(d['D1_CAT']), np.nan, d['ltHS'])

        d['someHS'] = np.where(d['D1_CAT'] == 2, 1, 0)
        d['someHS'] = np.where(np.isnan(d['D1_CAT']), np.nan, d['someHS'])

        d['HSgrad'] = np.where(d['D1_CAT'] == 3, 1, 0)
        d['HSgrad'] = np.where(np.isnan(d['D1_CAT']), np.nan, d['HSgrad'])

        d['someCol'] = np.where(d['D1_CAT'] == 5, 1, 0)
        d['someCol'] = np.where(np.isnan(d['D1_CAT']), np.nan, d['someCol'])

        d['BA'] = np.where(d['D1_CAT'] == 6, 1, 0)
        d['BA'] = np.where(np.isnan(d['D1_CAT']), np.nan, d['BA'])

        d['GradSch'] = np.where(d['D1_CAT'] == 7, 1, 0)
        d['GradSch'] = np.where(np.isnan(d['D1_CAT']), np.nan, d['GradSch'])

        d['noHSdegree'] = np.where((d['ltHS'] == 1) | (d['someHS'] == 1), 1, 0)
        d['noHSdegree'] = np.where(np.isnan(d['ltHS']) & np.isnan(d['someHS']), np.nan, d['noHSdegree'])

        d['BAplus'] = np.where((d['BA'] == 1) | (d['GradSch'] == 1), 1, 0)
        d['BAplus'] = np.where(np.isnan(d['BA']) & np.isnan(d['GradSch']), np.nan, d['BAplus'])

        # Family income using midpoint of category
        conditions = [(d['D4_CAT'] == 3), (d['D4_CAT'] == 4),
                      (d['D4_CAT'] == 5), (d['D4_CAT'] == 6),
                      (d['D4_CAT'] == 7), (d['D4_CAT'] == 8),
                      (d['D4_CAT'] == 9), (d['D4_CAT'] == 10)]
        choices = [15, 25, 32.5, 37.5, 45, 62.5, 87.5, 130]
        d['faminc'] = np.select(conditions, choices, default=np.nan)
        d['lnfaminc'] = np.log(d['faminc'])

        # Marital status
        d['married'] = np.where(d['D10'] == 1, 1, 0)
        d['married'] = np.where(np.isnan(d['D10']), np.nan, d['married'])

        d['partner'] = np.where(d['D10'] == 2, 1, 0)
        d['partner'] = np.where(np.isnan(d['D10']), np.nan, d['partner'])

        d['separated'] = np.where(d['D10'] == 3, 1, 0)
        d['separated'] = np.where(np.isnan(d['D10']), np.nan, d['separated'])

        d['divorced'] = np.where(d['D10'] == 4, 1, 0)
        d['divorced'] = np.where(np.isnan(d['D10']), np.nan, d['divorced'])

        d['widowed'] = np.where(d['D10'] == 5, 1, 0)
        d['widowed'] = np.where(np.isnan(d['D10']), np.nan, d['widowed'])

        d['nevermarried'] = np.where(d['D10'] == 6, 1, 0)
        d['nevermarried'] = np.where(np.isnan(d['D10']), np.nan, d['nevermarried'])

        # Race/ethnicity
        d['raceth'] = np.where((np.isnan(d['D5']) == False) & (d['D5'] == 1), 7, d['D6_1_CAT'])

        d['native'] = np.where(d['raceth'] == 1, 1, 0)
        d['native'] = np.where(np.isnan(d['raceth']), np.nan, d['native'])

        d['asian'] = np.where(d['raceth'] == 2, 1, 0)
        d['asian'] = np.where(np.isnan(d['raceth']), np.nan, d['asian'])

        d['black'] = np.where(d['raceth'] == 4, 1, 0)
        d['black'] = np.where(np.isnan(d['raceth']), np.nan, d['black'])

        d['white'] = np.where(d['raceth'] == 5, 1, 0)
        d['white'] = np.where(np.isnan(d['raceth']), np.nan, d['white'])

        d['other'] = np.where(d['raceth'] == 6, 1, 0)
        d['other'] = np.where(np.isnan(d['raceth']), np.nan, d['other'])

        d['hisp'] = np.where(d['raceth'] == 7, 1, 0)
        d['hisp'] = np.where(np.isnan(d['raceth']), np.nan, d['hisp'])

        # leave reason for most recent leave
        d['reason_take'] = np.where((np.isnan(d['A20']) == False) & (d['A20'] == 2), d['A5_2_CAT'], d['A5_1_CAT'])

        # leave reason for most recent leave (revised)
        d['reason_take_rev'] = np.where((np.isnan(d['A20']) == False) & (d['A20'] == 2), d['A5_2_CAT_REV'], d['A5_1_CAT_rev'])

        # taken doctor
        d['YNdoctor_take'] = np.where((np.isnan(d['A20']) == False) & (d['A20'] == 2), d['A11_2'], d['A11_1'])
        d['doctor_take'] = np.where(d['YNdoctor_take'] == 1, 1, 0)
        d['doctor_take'] = np.where(np.isnan(d['YNdoctor_take']), np.nan, d['doctor_take'])

        # taken hospital
        d['YNhospital_take'] = np.where((np.isnan(d['A20']) == False) & (d['A20'] == 2), d['A12_2'], d['A12_1'])
        d['hospital_take'] = np.where(d['YNhospital_take'] == 1, 1, 0)
        d['hospital_take'] = np.where(np.isnan(d['YNhospital_take']), np.nan, d['hospital_take'])
        d['hospital_take'] = np.where(np.isnan(d['hospital_take']) & (d['doctor_take'] == 0), 0, d['hospital_take'])

        # need doctor
        d['doctor_need'] = np.where(d['B12_1'] == 1, 1, 0)
        d['doctor_need'] = np.where(np.isnan(d['B12_1']), np.nan, d['doctor_need'])

        # need hospital
        d['hospital_need'] = np.where(d['B13_1'] == 1, 1, 0)
        d['hospital_need'] = np.where(np.isnan(d['B13_1']), np.nan, d['hospital_need'])
        d['hospital_need'] = np.where(np.isnan(d['hospital_need']) & (d['doctor_need'] == 0), 0, d['hospital_need'])

        # taken or needed doctor or hospital for leave category
        d['doctor1'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & (d['LEAVE_CAT'] == 2), d['doctor_need'], d['doctor_take'])
        d['doctor2'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 4)),
                                d['doctor_need'], d['doctor_take'])

        d['hospital1'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & (d['LEAVE_CAT'] == 2), d['hospital_need'],
                                  d['hospital_take'])
        d['hospital2'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 4)),
                                  d['hospital_need'], d['hospital_take'])

        d['doctor'] = np.where(d['doctor_take']==1, 1, 0)
        d['doctor'] = np.where(d['doctor_need']==1, 1, d['doctor'])
        d['doctor'] = np.where((d['doctor_take'].isna()) & (d['doctor_need'].isna()), np.nan, d['doctor'])

        d['hospital'] = np.where(d['hospital_take']==1, 1, 0)
        d['hospital'] = np.where(d['hospital_need']==1, 1, d['hospital'])
        d['hospital'] = np.where((d['hospital_take'].isna()) & (d['hospital_need'].isna()), np.nan, d['hospital'])

        # length of leave for most recent leave
        d['length'] = np.where((np.isnan(d['A20']) == False) & (d['A20'] == 2), d['A19_2_CAT_rev'], d['A19_1_CAT_rev'])
        d['lengthsq'] = np.array(d['length']) ** 2
        d['lnlength'] = np.log(d['length'])
        d['lnlengthsq'] = np.log(d['length']) ** 2

        # any pay received
        d['anypay'] = np.where(d['A45'] == 1, 1, 0)
        d['anypay'] = np.where(np.isnan(d['A45']), np.nan, d['anypay'])

        # state program
        d['recStateFL'] = np.where(d['A48b'] == 1, 1, 0)
        d['recStateFL'] = np.where(np.isnan(d['A48b']), np.nan, d['recStateFL'])
        d['recStateFL'] = np.where(np.isnan(d['recStateFL']) & (d['anypay'] == 0), 0, d['recStateFL'])

        d['recStateDL'] = np.where(d['A48c'] == 1, 1, 0)
        d['recStateDL'] = np.where(np.isnan(d['A48c']), np.nan, d['recStateDL'])
        d['recStateDL'] = np.where(np.isnan(d['recStateDL']) & (d['anypay'] == 0), 0, d['recStateDL'])

        d['recStatePay'] = np.where((d['recStateFL'] == 1) | (d['recStateDL'] == 1), 1, 0)
        d['recStatePay'] = np.where(np.isnan(d['recStateFL']) | np.isnan(d['recStateDL']), np.nan, d['recStatePay'])

        # --------------------------
        # Leave taking variables
        # --------------------------

        # fully paid
        d['fullyPaid'] = np.where(d['A49'] == 1, 1, 0)
        d['fullyPaid'] = np.where(np.isnan(d['A49']), np.nan, d['fullyPaid'])

        # longer leave if more pay
        d['longerLeave'] = np.where(d['A55'] == 1, 1, 0)
        d['longerLeave'] = np.where(np.isnan(d['A55']), np.nan, d['longerLeave'])

        # could not afford to take leave
        d['unaffordable'] = np.where(d['B15_1_CAT'] == 5, 1, 0)
        d['unaffordable'] = np.where(np.isnan(d['B15_1_CAT']), np.nan, d['unaffordable'])

        # weights
        w_emp = np.mean(d[d['LEAVE_CAT'] == 3]['weight'])
        w_leave = np.mean(d[d['LEAVE_CAT'] != 3]['weight'])
        d['fixed_weight'] = np.where(d['LEAVE_CAT'] == 3, w_emp, w_leave)
        d['freq_weight'] = np.round(d['weight'])

        # --------------------------
        # dummies for leave type
        # --------------------------

        # there are three variables for each leave type:
        # (1) taking a leave
        # (2) needing a leave
        # (3) taking or needing a leave

        # maternity disability
        # d['reason_take'] = np.where((np.isnan(d['A20'])==False) & (d['A20']==2),d['A5_2_CAT'],d['A5_1_CAT'])
        # if A20 = 1 or missing the go to A5_1_CAT for most recent leave
        # if A20 = 2 the go to A5_2_CAT for most recent leave (no such cases in FMLA 2012)

        # d['take_matdis'] = np.where(((d['A5_1_CAT']==21)&(d['A11_1']==1)&(d['GENDER_CAT']==2))&((d['A20']!=2) | (d['A20'].isna())) | (d['A5_1_CAT_rev']==32),1,0)
        # d['take_matdis'] = np.where(((d['A5_2_CAT']==21)&(d['A11_1']==1)&(d['GENDER_CAT']==2))&((d['A20']==1)) | (d['A5_1_CAT_rev']==32),1,0)
        d['take_matdis'] = np.where(
            (((d['A5_1_CAT'] == 21) & (d['A11_1'] == 1) & (d['GENDER_CAT'] == 2)) | (d['A5_1_CAT_rev'] == 32)) & (
                (d['A20'] != 2) | (d['A20'].isna())), 1, 0)
            # follow makes no change for FMLA 2012 since A5_2_CAT = 21 has 0 case but include code for later FMLA data
        d['take_matdis'] = np.where(
            (((d['A5_2_CAT'] == 21) & (d['A11_1'] == 1) & (d['GENDER_CAT'] == 2)) | (d['A5_1_CAT_rev'] == 32)) & (
                d['A20'] == 2), 1, d['take_matdis'])

        d['take_matdis'] = np.where(np.isnan(d['take_matdis']), 0, d['take_matdis'])
        d['take_matdis'] = np.where((np.isnan(d['A5_1_CAT'])) & (np.isnan(d['A5_2_CAT'])), np.nan, d['take_matdis'])
        d['take_matdis'] = np.where(np.isnan(d['take_matdis']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                                    d['take_matdis'])
        d['take_matdis'] = np.where(np.isnan(d['take_matdis']) & (d['male'] == 1), 0, d['take_matdis'])
        # # make sure take_matdis = 0 for matdis taker whose most recent leave is not matdis
        d['take_matdis'] = np.where((d['take_matdis'] == 1) &
                                    (d['A20'] == 2) &
                                    (d['A5_1_CAT'] == 21) &
                                    (d['A5_2_CAT'] != 21), 0, d['take_matdis'])

        d['need_matdis'] = np.where(
            ((d['B6_1_CAT'] == 21) & (d['B12_1'] == 1) & (d['GENDER_CAT'] == 2)) | (d['B6_1_CAT_rev'] == 32), 1, 0)
        d['need_matdis'] = np.where(np.isnan(d['need_matdis']), 0, d['need_matdis'])
        d['need_matdis'] = np.where(np.isnan(d['B6_1_CAT']), np.nan, d['need_matdis'])
        d['need_matdis'] = np.where(np.isnan(d['need_matdis']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                                    d['need_matdis'])
        d['need_matdis'] = np.where(np.isnan(d['need_matdis']) & (d['male'] == 1), 0, d['need_matdis'])

        d['type_matdis'] = np.where((d['take_matdis'] == 1) | (d['need_matdis'] == 1), 1, 0)
        d['type_matdis'] = np.where(np.isnan(d['take_matdis']) | np.isnan(d['need_matdis']), np.nan, d['type_matdis'])

        # new child/bond
        d['take_bond'] = np.where(((d['take_matdis']==0) | (d['take_matdis'].isna())) &
                                  (
                                      (d['A5_1_CAT_rev']==31) |
                                      ((d['A5_1_CAT']==21) & (d['GENDER_CAT']==1)) |
                                      ((d['A5_1_CAT']==21) & (d['GENDER_CAT']==2) & (d['A5_1_CAT_rev']!=32))
                                  ) &
                                  ((d['A20']!=2) | (d['A20'].isna())), 1, 0)
        d['take_bond'] = np.where(np.isnan(d['A5_1_CAT']), np.nan, d['take_bond'])
        d['take_bond'] = np.where(np.isnan(d['take_bond']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0, d['take_bond'])

        d['need_bond'] = np.where(((d['need_matdis']==0) | (d['need_matdis'].isna())) &
                                  (
                                      (d['B6_1_CAT_rev']==31) |
                                      ((d['B6_1_CAT']==21) & (d['GENDER_CAT']==1)) |
                                      ((d['B6_1_CAT']==21) & (d['GENDER_CAT']==2) & (d['B6_1_CAT_rev']!=32))
                                  ), 1, 0)
        d['need_bond'] = np.where(np.isnan(d['B6_1_CAT']), np.nan, d['need_bond'])
        d['need_bond'] = np.where(np.isnan(d['need_bond']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0, d['need_bond'])


        d['type_bond'] = np.where((d['take_bond'] == 1) | (d['need_bond'] == 1), 1, 0)
        d['type_bond'] = np.where(np.isnan(d['take_bond']) | np.isnan(d['need_bond']), np.nan, d['type_bond'])

        # own health
        d['take_own'] = np.where(d['reason_take'] == 1, 1, 0)
        d['take_own'] = np.where((d['reason_take'].isna()), np.nan, d['take_own'])
        d['take_own'] = np.where((np.isnan(d['take_own'])) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0, d['take_own'])

        # d['need_own'] = np.where((d['B6_1_CAT']==1) | (d['B6_2_CAT']==1),1,0)
        d['need_own'] = np.where(d['B6_1_CAT'] == 1, 1, 0)
        d['need_own'] = np.where(d['B6_1_CAT'].isna(), np.nan, d['need_own'])
        # multiple leaves - if one leave is Own the other NA, need_own is NA
        # d['need_own'] = np.where((d['B6_1_CAT'].isna()) & (d['B6_2_CAT']!=1), np.nan, d['need_own'])
        # d['need_own'] = np.where((d['B6_1_CAT']!=1) & (d['B6_2_CAT'].isna()), np.nan, d['need_own'])
        d['need_own'] = np.where(np.isnan(d['need_own']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0, d['need_own'])

        d['type_own'] = np.where((d['take_own'] == 1) | (d['need_own'] == 1), 1, 0)
        d['type_own'] = np.where(np.isnan(d['take_own']) | np.isnan(d['need_own']), np.nan, d['type_own'])

        # ill child
        d['take_illchild'] = np.where(d['reason_take'] == 11, 1, 0)
        d['take_illchild'] = np.where(d['reason_take'].isna(), np.nan, d['take_illchild'])
        d['take_illchild'] = np.where(np.isnan(d['take_illchild']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                                      d['take_illchild'])
        # d['need_illchild'] = np.where((d['B6_1_CAT']==11) | (d['B6_2_CAT']==11),1,0)
        d['need_illchild'] = np.where(d['B6_1_CAT'] == 11, 1, 0)
        d['need_illchild'] = np.where(np.isnan(d['B6_1_CAT']), np.nan, d['need_illchild'])

        # multiple leaves - if one leave is not Illchild the other NA, need_illchild is NA
        # d['need_illchild'] = np.where((d['B6_1_CAT'].isna()) & (d['B6_2_CAT']!=11), np.nan, d['need_illchild'])
        # d['need_illchild'] = np.where((d['B6_1_CAT']!=11) & (d['B6_2_CAT'].isna()), np.nan, d['need_illchild'])
        d['need_illchild'] = np.where(np.isnan(d['need_illchild']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                                      d['need_illchild'])

        d['type_illchild'] = np.where((d['take_illchild'] == 1) | (d['need_illchild'] == 1), 1, 0)
        d['type_illchild'] = np.where(np.isnan(d['take_illchild']) | np.isnan(d['need_illchild']), np.nan, d['type_illchild'])

        # ill spouse
        d['take_illspouse'] = np.where(d['reason_take'] == 12, 1, 0)
        d['take_illspouse'] = np.where(d['reason_take'].isna(), np.nan, d['take_illspouse'])
        d['take_illspouse'] = np.where(np.isnan(d['take_illspouse']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                                       d['take_illspouse'])
        d['take_illspouse'] = np.where(np.isnan(d['take_illspouse']) & ((d['nevermarried'] == 1) |
                                                                        (d['separated'] == 1) |
                                                                        (d['divorced'] == 1) |
                                                                        (d['widowed'] == 1)), 0, d['take_illspouse'])

        d['need_illspouse'] = np.where(d['B6_1_CAT'] == 12, 1, 0)
        d['need_illspouse'] = np.where(d['B6_1_CAT'].isna(), np.nan, d['need_illspouse'])
        d['need_illspouse'] = np.where(np.isnan(d['need_illspouse']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                                       d['need_illspouse'])
        d['need_illspouse'] = np.where(np.isnan(d['need_illspouse']) & ((d['nevermarried'] == 1) |
                                                                        (d['separated'] == 1) |
                                                                        (d['divorced'] == 1) |
                                                                        (d['widowed'] == 1)), 0, d['need_illspouse'])

        d['type_illspouse'] = np.where((d['take_illspouse'] == 1) | (d['need_illspouse'] == 1), 1, 0)
        d['type_illspouse'] = np.where(np.isnan(d['take_illspouse']) | np.isnan(d['need_illspouse']), np.nan,
                                       d['type_illspouse'])

        # ill parent
        d['take_illparent'] = np.where(d['reason_take'] == 13, 1, 0)
        d['take_illparent'] = np.where(d['reason_take'].isna(), np.nan, d['take_illparent'])
        d['take_illparent'] = np.where(np.isnan(d['take_illparent']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                                       d['take_illparent'])

        d['need_illparent'] = np.where(d['B6_1_CAT'] == 13, 1, 0)
        d['need_illparent'] = np.where(d['B6_1_CAT'].isna(), np.nan, d['need_illparent'])
        d['need_illparent'] = np.where(np.isnan(d['need_illparent']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                                       d['need_illparent'])

        d['type_illparent'] = np.where((d['take_illparent'] == 1) | (d['need_illparent'] == 1), 1, 0)
        d['type_illparent'] = np.where(np.isnan(d['take_illparent']) | np.isnan(d['need_illparent']), np.nan,
                                       d['type_illparent'])

        # overall taker/needer status, any of 6 types
        types = ['own','matdis','bond','illchild','illspouse','illparent']
        d['taker'] = d[['take_%s' % t for t in types]].apply(lambda x: max(x), axis=1)
        d['needer'] = d[['need_%s' % t for t in types]].apply(lambda x: max(x), axis=1)


        # number of reasons leaves taken - need to consolidate following
        # A5_1_CAT: loop 1 taker reason type
        # A5_2_CAT: loop 2 taker reason type
        # A4a_CAT: number of taker reasons

        # get total of take reasons(out of 6) inferred from A5_1_CAT and A5_2_CAT
        d = d.rename(columns={'A5_2_CAT_REV':'A5_2_CAT_rev'})
        for lp in ['1', '2']:
            d['take_any6_loop%s' % lp] = 0
            d['take_any6_loop%s' % lp] = np.where((d['A5_%s_CAT' % lp] == 1) |
                                                  (d['A5_%s_CAT' % lp] == 11) |
                                                  (d['A5_%s_CAT' % lp] == 12) |
                                                  (d['A5_%s_CAT' % lp] == 13) |
                                                  (d['A5_%s_CAT' % lp] == 21) |
                                                  (d['A5_%s_CAT_rev' % lp]==32 ), 1, 0)
        d['take_any6'] = d['take_any6_loop1'] + d['take_any6_loop2']
        # make sure num_leaves_taken is at least take_any6 above
        d['num_leaves_taken'] = d['A4a_CAT']
        d['num_leaves_taken'] = np.where(d['num_leaves_taken'].isna(), 0, d['num_leaves_taken'])
        d['num_leaves_taken'] = np.where(d['num_leaves_taken'] < d['take_any6'], d['take_any6'], d['num_leaves_taken'])


        # number of reasons leaves needed - need to consolidate following
        # B6_1_CAT: loop 1 needer reason type
        # B6_2_CAT: loop 2 needer reason type
        # B5_CAT: number of times needing leaves past 12m (best approximation of number of reasons in data)

        # get total of need reasons(out of 6) inferred from B6_1_CAT and B6_2_CAT
        for lp in ['1', '2']:
            d['need_any6_loop%s' % lp] = 0
            d['need_any6_loop%s' % lp] = np.where((d['B6_%s_CAT' % lp] == 1) |
                                                  (d['B6_%s_CAT' % lp] == 11) |
                                                  (d['B6_%s_CAT' % lp] == 12) |
                                                  (d['B6_%s_CAT' % lp] == 13) |
                                                  (d['B6_%s_CAT' % lp] == 21), 1, 0)
        d['need_any6'] = d['need_any6_loop1'] + d['need_any6_loop2']

        # make sure num_leaves_need is at least need_any6
        d['num_leaves_need'] = d['B5_CAT']
        d['num_leaves_need'] = np.where(d['num_leaves_need'].isna(), 0, d['num_leaves_need'])
        d['num_leaves_need'] = np.where(d['num_leaves_need'] < d['need_any6'], d['need_any6'], d['num_leaves_need'])

        # most recent leave length by leave type
        types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
        for t in types:
            d['length_%s' % t] = np.where(d['take_%s' % t] == 1, d['length'], 0)
            d['length_%s' % t] = np.where(d['take_%s' % t].isna(), np.nan, d['length_%s' % t])

        # most recent take/need leave type in 1 col

        d['take_type'] = np.nan
        for t in types:
            d.loc[(d['take_type'].isna()) & (d['take_%s' % t] == 1), 'take_type'] = t
        d['need_type'] = np.nan
        for t in types:
            d.loc[(d['need_type'].isna()) & (d['need_%s' % t] == 1), 'need_type'] = t

        # multiple leaver (taker/needer)
        d['multiple'] = np.nan
        d['multiple'] = np.where(d['LEAVE_CAT'] != 3, 0, d['multiple'])
        d['multiple'] = np.where((d['A4_CAT'].notna()) & (d['A4_CAT'] >= 2), 1, d['multiple'])
        d['multiple'] = np.where((d['B5b_CAT'].notna()) & (d['B5b_CAT'] >= 2), 1, d['multiple'])

        # proportion of pay received from employer (mid point of ranges provided in FMLA)
        d['prop_pay'] = np.where(d['A50'] == 1, 0.125, np.nan)
        d['prop_pay'] = np.where(d['A50'] == 2, 0.375, d['prop_pay'])
        d['prop_pay'] = np.where(d['A50'] == 3, 0.5, d['prop_pay'])
        d['prop_pay'] = np.where(d['A50'] == 4, 0.625, d['prop_pay'])
        d['prop_pay'] = np.where(d['A50'] == 5, 0.875, d['prop_pay'])
        d['prop_pay'] = np.where(d['A49'] == 1, 1, d['prop_pay'])
        d['prop_pay'] = np.where(d['A45'] == 2, 0, d['prop_pay'])

        # Benefits received as proportion of pay
        # baseline is employer-provided pay: starting at 0, will be imputed

        d['benefit_prop'] = 0
        # Leave Program Participation
        # baseline is absence of program, so this will start as a nonparticipant
        d['particip'] = 0
        # Cost to program as proportion of pay
        # baseline is 0
        d['cost_prop'] = 0

        # -------------
        # The rest of the code in this function creates a variable 'resp_len'
        # which will flag 0/1 for a worker that is likely to a more favourable leave program
        # by increasing leave length. It is used to help simulate counterfactual leave
        # for a program that increases wage replacement
        # -------------

        # Initiate
        d['resp_len'] = np.nan

        # LEAVE_CAT: employed only
        # EMPLOYED ONLY workers have no need and take no leave, would not respond anyway
        d.loc[d['LEAVE_CAT'] == 3, 'resp_len'] = 0

        # A55 asks if worker would take longer leave if paid?
        d.loc[(d['resp_len'].isna()) & (d['A55'] == 2), 'resp_len'] = 0
        d.loc[(d['resp_len'].isna()) & (d['A55'] == 1), 'resp_len'] = 1

        # The following variables indicate whether leave was cut short for financial issues
        # A23c: unable to afford unpaid leave due to leave taking
        # A53g: cut leave time short to cover lost wages
        # A62a: return to work because cannot afford more leaves
        # B15_1_CAT, B15_2_CAT: can't afford unpaid leave
        d.loc[(d['resp_len'].isna()) & ((d['A23c'] == 1) |
                                        (d['A53g'] == 1) |
                                        (d['A62a'] == 1) |
                                        (d['B15_1_CAT'] == 5) |
                                        (d['B15_2_CAT'] == 5)), 'resp_len'] = 1
        # Assign 0s to A23c, A53g, A62a among remaining workers who indicate reasons other than unaffordability
        # B15_1_CAT and B15_2_CAT only has one group (cod = 5) identified as constrained by unaffordability
        # These financially-constrained workers were assigned resp_len=1 above, all other cornered workers would not respond
        # Check reasons of no leave among rest: d[d['resp_len'].isna()].B15_1_CAT.value_counts().sort_index()
        # all reasons unsolved by replacement generosity
        d.loc[(d['resp_len'].isna()) & (d['A23c'] == 2), 'resp_len'] = 0
        d.loc[(d['resp_len'].isna()) & (d['A53g'] == 2), 'resp_len'] = 0
        d.loc[(d['resp_len'].isna()) & (d['A62a'] == 2), 'resp_len'] = 0
        d.loc[(d['resp_len'].isna()) & (d['B15_1_CAT'].notna()), 'resp_len'] = 0
        d.loc[(d['resp_len'].isna()) & (d['B15_2_CAT'].notna()), 'resp_len'] = 0

        # Assume all takers/needers with ongoing condition are 'cornered' and would respond with longer leaves
        # A10_1, A10_2: regular/ongoing condition, takers and dual
        # B11_1, B11_2: regular/ongoing condition, needers and dual
        d.loc[(d['resp_len'].isna()) &
              ((d['A10_1'] == 2) | (d['A10_1'] == 3) | (d['B11_1'] == 2) | (d['B11_1'] == 3)), 'resp_len'] = 1

        # Check LEAVE_CAT of rest: d[d['resp_len'].isna()]['LEAVE_CAT'].value_counts().sort_index()
        # 267 takers and 3 needers
        # with no evidence in data of need solvable / unsolvable by $, assume solvable to be conservative
        d.loc[d['resp_len'].isna(), 'resp_len'] = 1

        # Fill in multiple taker/needer reason types using info from rest of loops
        # fill in for multiple takers - first we need reason for non-recent 2nd leave (longest). This is in loop 1 if A20=2
        # if this reason is any6 and differs from most recent reason then fill in
        # separate New Child code into matdis and bond

        # For take_type2, fill in only if take_type (most recent type) is non-missing

        dctr = {1: 'own',
                11: 'illchild',
                12: 'illspouse',
                13: 'illparent',
                21: 'New Child'}
        d['take_type2'] = np.nan
        d.loc[(d['A20'] == 2)
              & (d['take_any6_loop1'] == 1) & (d['take_any6_loop2'] == 1)
              & (d['A5_1_CAT'] != d['A5_2_CAT']) & (d['take_type'].notna()), 'take_type2'] = \
            d.loc[(d['A20'] == 2)
                  & (d['take_any6_loop1'] == 1) & (d['take_any6_loop2'] == 1)
                  & (d['A5_1_CAT'] != d['A5_2_CAT']) & (d['take_type'].notna()), 'A5_1_CAT'].apply(lambda x: dctr[x])
        d.loc[((d['take_type2'] == 'New Child')
               & (d['A11_1'] == 1)
               & (d['GENDER_CAT'] == 2))
              | ((d['take_type2'] == 'New Child') & (d['A5_1_CAT_rev'] == 32)), 'take_type2'] = 'matdis'
        d.loc[(d['take_type2'] == 'New Child'), 'take_type2'] = 'bond'

        # fill in for multiple needers - non-recent 2nd leave is in need-loop 2
        # this reason is any6 and differs from most recent reason then fill in
        # separate New Child code into matdis and bond
        d['need_type2'] = np.nan
        d.loc[((d['need_any6_loop1'] == 1) & (d['need_any6_loop2'] == 1)
              & (d['B6_1_CAT'] != d['B6_2_CAT'])) & (d['need_type'].notna()), 'need_type2'] = \
            d.loc[((d['need_any6_loop1'] == 1) & (d['need_any6_loop2'] == 1)
                  & (d['B6_1_CAT'] != d['B6_2_CAT'])) & (d['need_type'].notna()), 'B6_1_CAT'].apply(lambda x: dctr[x])
        d.loc[((d['need_type2'] == 'New Child')
               & (d['B12_1'] == 1)
               & (d['GENDER_CAT'] == 2))
              | ((d['need_type2'] == 'New Child') & (d['B6_1_CAT_rev'] == 32)), 'need_type2'] = 'matdis'
        d.loc[(d['need_type2'] == 'New Child'), 'need_type2'] = 'bond'

        # Check why some obs have missing recent take_type
        #[check] print(d[(d['take_any6']==1) & (d['take_type'].isna())][['take_any6', 'take_type', 'A5_1_CAT', 'A5_2_CAT', 'A20']])
        # because they all have A20 = 2 and when searching for recent leave from loop 2, the type is 'other' or nan
        # for them we use A5_1_CAT as most recent leave taken, and breakdown New Child into matdis and bond using loop1 info
        d.loc[(d['take_any6']==1) & (d['take_type'].isna()), 'take_type'] = \
        d.loc[(d['take_any6']==1) & (d['take_type'].isna()), 'A5_1_CAT'].apply(lambda x: dctr[x])
        d.loc[((d['take_type'] == 'New Child')
               & (d['A11_1'] == 1)
               & (d['GENDER_CAT'] == 2)), 'take_type'] = 'matdis'
        d.loc[(d['take_type'] == 'New Child'), 'take_type'] = 'bond'
        # check
        #[check] print(d[(d['take_any6']==1) & (d['take_type'].isna())][['take_any6', 'take_type', 'A5_1_CAT', 'A5_2_CAT', 'A20']])

        # similarly, identified following needers
        #[check] print(d[(d['need_any6']==1) & (d['need_type'].isna())][['need_any6', 'need_type', 'B6_1_CAT', 'B6_2_CAT']])
        # similarly, force most recent leave type for them, by using B6_2_CAT
        d.loc[(d['need_any6']==1) & (d['need_type'].isna()), 'need_type'] = \
        d.loc[(d['need_any6']==1) & (d['need_type'].isna()), 'B6_2_CAT'].apply(lambda x: dctr[x])
        d.loc[((d['need_type'] == 'New Child')
               & (d['B12_1'] == 1)
               & (d['GENDER_CAT'] == 2)), 'need_type'] = 'matdis'
        d.loc[(d['need_type'] == 'New Child'), 'need_type'] = 'bond'
        # check
        #[check] print(d[(d['need_any6']==1) & (d['need_type'].isna())][['need_any6', 'need_type', 'B6_1_CAT', 'B6_2_CAT']])

        # Check: so far for take_any6 > 0, take_type and take_type2 should be all defined for num_leaves_taken = 1 or 2
        d[(d['take_any6']>0) & ((d['num_leaves_taken']==1) | (d['num_leaves_taken']==2))]['take_type'].isna().value_counts()
        d[(d['take_any6']>0) & (d['num_leaves_taken']==2)]['take_type2'].isna().value_counts()

        d[(d['need_any6']==1) & ((d['num_leaves_need']==1) | (d['num_leaves_need']==2))]['need_type'].isna().value_counts()

        # Fill in for rest of leave reasons for multiple leavers/needers using imputation
            # First estimate distribution

            # -------------
            # Conditional prob of other leave types of another leave if another leave exists, given a known leave type
            # These conditional probs will be applied recursively to simulate leave types for multiple leavers
            # e.g. given the recent take_type=matdis, nl =2, need to simulate 1 type from the other 5 types << NT know pr(type | matdis)
            # -------------
        dm = d[(d['take_any6'] > 1) & (d['take_type'].notna()) & (d['A20'] == 2) & (d['A5_1_CAT'].notna())]
        dcp = {}
        for t in types:
            nums = np.array([sum(dm[(dm['take_type'] == t) & (dm['take_type2'] == x)]['weight']) for x in types])
            nums = np.where(nums == 0, 10, nums)  # assign small weight (10 ppl) as total workers with take_type = t
                                                  # and take_type2 = x to avoid zero conditional probs, which may cause
                                                  # 'no-further-simulation' issue when recursively simulating
                                                  # multiple leave types
            ps = nums / sum(nums)
            dcp[t] = ps
        # Normalize to make sure probs are conditional on OTHER types, prob(next leave type = t | current type = t)=0 for all t
        for type, ps in dcp.items():
            i = types.index(type)
            ps[i] = 0
            ps = ps / ps.sum()
            dcp[type] = ps
            i += 1
        dict_dcp = dcp
        dcp = pd.DataFrame.from_dict(dcp, orient='index')
        dcp.columns = [t for t in types]
        dcp = dcp.sort_index()
        dcp = dcp.sort_index(axis=1)
        dcp.to_csv('./data/fmla_2012/conditional_probs_leavetypes.csv', index=True)
        otypes = list(dcp.columns) # types in alphabetical order

        # -------------
        # Unconditional prob of taking leaves if leave type is unknown
        # this will be used to simulate types of multi-leavers if missing types for all leaves
        # possible if multi-leavers and reported loop-1/2 types are out of the 6 types
        # -------------
        dict_dup = {}
        denom = 0
        for t in dcp.columns:
            num = d[d['take_%s' % t]==1]['weight'].sum()
            denom += num # build denom this way so that ratios sum up to 1
        ups = [] # unconditional probs of taken types in order of dcp.columns (alphabetical)
        for t in dcp.columns:
            num = d[d['take_%s' % t] == 1]['weight'].sum()
            dict_dup[t] = num / denom
            ups.append(num/denom)

        # Impute rest type info not in survey loops, up to num_leaves_taken/need

        # Before imputing leave types, consolidate num_leaves_taken/need - among 2 loops some reported types out of the 6 types
        # For these workers reduce num_leaves accordingly
        d['num_leaves_taken_adj'] = d['num_leaves_taken']
        d.loc[(d['num_leaves_taken']==1) & (d['take_type'].isna()), 'num_leaves_taken_adj'] = 0
        for n in range(2, 7):
            d.loc[(d['num_leaves_taken'] == n) & (d['take_type'].isna()), 'num_leaves_taken_adj'] = n-2  # if take_type is nan, then
            #  take_type2 by definition would be nan too, reduce num_leaves by 2
            d.loc[(d['num_leaves_taken'] == n) & (d['take_type'].notna()) & (d['take_type2'].isna()), 'num_leaves_taken_adj'] = n-1
        d['num_leaves_taken'] = d['num_leaves_taken_adj']
        # Similarly reduce for num_leaves_need
        d.loc[(d['num_leaves_need']==1) & (d['need_type'].isna()), 'num_leaves_taken'] = 0
        for n in range(2, 7):
            d.loc[(d['num_leaves_need'] == n) & (d['need_type'].isna()), 'num_leaves_need'] = n-2  # if need_type is nan, then
            #  need_type2 by definition would be nan too, reduce num_leaves by 2
            d.loc[(d['num_leaves_need'] == n) & (d['need_type'].notna()) & (d['need_type2'].isna()), 'num_leaves_need'] = n-1
        # Further cap num_leaves wrt logical restrictions
        # if male - excl. matdis
        # if nospouse (nevermarried, separated, divorced, widowed) - excl. illspouse
        # if nochildren - excl. bond

        d['max_num_leaves'] = 6
        d.loc[d['male']==1, 'max_num_leaves'] = d.loc[d['male']==1, 'max_num_leaves'] - 1
        d.loc[d['nospouse']==1, 'max_num_leaves'] = d.loc[d['nospouse']==1, 'max_num_leaves'] - 1
        d.loc[d['nochildren']==1, 'max_num_leaves'] = d.loc[d['nochildren']==1, 'max_num_leaves'] - 1
        d['num_leaves_taken'] = d[['num_leaves_taken', 'max_num_leaves']].apply(lambda x: min(x), axis=1)
        d['num_leaves_need'] = d[['num_leaves_need', 'max_num_leaves']].apply(lambda x: min(x), axis=1)



        # Impute leave types for multiple leaves - note that in general we have to do this for num_leaves = 1, 2, ... , 6
        # because the adjusted num_leaves = 1 can come from original num_leaves = 3 but with both reported loops having types
        # out of the 6 main types
        chars = ['male', 'nospouse', 'nochildren']
        # Impute for take types
        # Impute types for most recent take_type for num_leaves_taken = 1...6
        for nlt in range(1, 7):
            L = len(d.loc[(d['num_leaves_taken']==nlt) & (d['take_type'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].isna()), 'take_type'] = \
                    d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].isna()), chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_adj_ups(ups, x))], axis=1)
            else:
                pass

        # Impute types for 2nd most recent take_type2 for num_leaves_taken = 2...6
        for nlt in range(2, 7):
            L = len(d.loc[(d['num_leaves_taken']==nlt) & (d['take_type'].notna()) & (d['take_type2'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].notna()) & (d['take_type2'].isna()), 'take_type2'] = \
                    d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].notna()) & (d['take_type2'].isna()), ['take_type'] + chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_adj_cps(dcp, x[1:])[x[0]])], axis=1)
            else:
                pass

        # Impute types for 3rd most recent take_type3 for num_leaves_taken = 3...6
        # when impute need to discard candidate types that have been reported / imputed in exising leaves
        d['take_type3']=np.nan
        for nlt in range(3, 7):
            L = len(d.loc[(d['num_leaves_taken']==nlt) & (d['take_type2'].notna()) & (d['take_type3'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type2'].notna()) & (d['take_type3'].isna()), 'take_type3'] = \
                    d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type2'].notna()) & (d['take_type3'].isna()), ['take_type', 'take_type2'] + chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_marginal_probs(get_adj_cps(dcp, x[2:])[x[1]], [otypes.index(x[0]), otypes.index(x[1])]))], axis=1)
            else:
                pass

        # Impute types for 4th most recent take_type4 for num_leaves_taken = 4...6
        # when impute need to discard candidate types that have been reported / imputed in exising leaves
        d['take_type4']=np.nan
        for nlt in range(4, 7):
            L = len(d.loc[(d['num_leaves_taken']==nlt) & (d['take_type3'].notna()) & (d['take_type4'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type3'].notna()) & (d['take_type4'].isna()), 'take_type4'] = \
                    d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type3'].notna()) & (d['take_type4'].isna()), ['take_type', 'take_type2', 'take_type3']+chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_marginal_probs(get_adj_cps(dcp, x[3:])[x[2]], [otypes.index(x[k]) for k in range(3)]))], axis=1)
            else:
                pass

        # Impute types for 5th most recent take_type5 for num_leaves_taken = 5, 6
        # when impute need to discard candidate types that have been reported / imputed in exising leaves
        d['take_type5']=np.nan
        for nlt in range(5, 7):
            L = len(d.loc[(d['num_leaves_taken']==nlt) & (d['take_type4'].notna()) & (d['take_type5'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type4'].notna()) & (d['take_type5'].isna()), 'take_type5'] = \
                    d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type4'].notna()) & (d['take_type5'].isna()), ['take_type', 'take_type2', 'take_type3', 'take_type4']+chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_marginal_probs(get_adj_cps(dcp, x[:4])[x[3]], [otypes.index(x[k]) for k in range(4)]))], axis=1)
            else:
                pass

        # Impute types for last take_type6 for num_leaves_taken = 6
        # this is just the only left leave type not selected by previous simulation
        # no logical restriction needed if all 6 types exist
        d['take_type6']=np.nan
        L = len(d.loc[(d['num_leaves_taken']==6) & (d['take_type5'].notna()) & (d['take_type6'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_taken'] == 6) & (d['take_type5'].notna()) & (d['take_type6'].isna()), 'take_type6'] = \
            d.loc[(d['num_leaves_taken'] == 6) & (d['take_type5'].notna()) & (d['take_type6'].isna()), ['take_type', 'take_type2', 'take_type3', 'take_type4', 'take_type5']] \
            .apply(lambda x: list(set(otypes) - set(x))[0], axis=1)
        else:
            pass

        # Impute for need types
        # B6_1_CAT has enough obs so use need type-based ups as unconditional prob vector
        # B6_2_CAT has too few obs so keep using take type-based dcp as conditional prob matrix
        dict_dup = {}
        denom = 0
        for t in dcp.columns:
            num = d[d['need_%s' % t]==1]['weight'].sum()
            denom += num # build denom this way so that ratios sum up to 1
        ups = [] # unconditional probs of need types in order of dcp.columns (alphabetical)
        for t in dcp.columns:
            num = d[d['need_%s' % t] == 1]['weight'].sum()
            dict_dup[t] = num / denom
            ups.append(num/denom)

        # Impute types for most recent need_type for num_leaves_need = 1...6
        for nlt in range(1, 7):
            L = len(d.loc[(d['num_leaves_need']==nlt) & (d['need_type'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].isna()), 'need_type'] = \
                    d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].isna()), chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_adj_ups(ups, x))], axis=1)
            else:
                pass

        # Impute types for 2nd most recent need_type2 for num_leaves_taken = 2...6
        for nlt in range(2, 7):
            L = len(d.loc[(d['num_leaves_need']==nlt) & (d['need_type'].notna()) & (d['need_type2'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].notna()) & (d['need_type2'].isna()), 'need_type2'] = \
                    d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].notna()) & (d['need_type2'].isna()), ['need_type'] + chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_adj_cps(dcp, x[1:])[x[0]])], axis=1)
            else:
                pass

        # Impute types for 3rd most recent need_type3 for num_leaves_taken = 3...6
        # when impute need to discard candidate types that have been reported / imputed in existing leaves
        d['need_type3']=np.nan
        otypes = list(dcp.columns) # types in alphabetical order
        for nlt in range(3, 7):
            L = len(d.loc[(d['num_leaves_need']==nlt) & (d['need_type2'].notna()) & (d['need_type3'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_need'] == nlt) & (d['need_type2'].notna()) & (d['need_type3'].isna()), 'need_type3'] = \
                    d.loc[(d['num_leaves_need'] == nlt) & (d['need_type2'].notna()) & (d['need_type3'].isna()), ['need_type', 'need_type2']+chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_marginal_probs(get_adj_cps(dcp, x[2:])[x[1]], [otypes.index(x[0]), otypes.index(x[1])]))], axis=1)
            else:
                pass

        # Impute types for 4th most recent need_type4 for num_leaves_taken = 4...6
        # when impute need to discard candidate types that have been reported / imputed in existing leaves
        d['need_type4']=np.nan
        for nlt in range(4, 7):
            L = len(d.loc[(d['num_leaves_need']==nlt) & (d['need_type3'].notna()) & (d['need_type4'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_need'] == nlt) & (d['need_type3'].notna()) & (d['need_type4'].isna()), 'need_type4'] = \
                    d.loc[(d['num_leaves_need'] == nlt) & (d['need_type3'].notna()) & (d['need_type4'].isna()), ['need_type', 'need_type2', 'need_type3']+chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_marginal_probs(get_adj_cps(dcp, x[3:])[x[2]], [otypes.index(x[k]) for k in range(3)]))], axis=1)
            else:
                pass

        # Impute types for 5th most recent need_type5 for num_leaves_taken = 5, 6
        # when impute need to discard candidate types that have been reported / imputed in existing leaves
        d['need_type5']=np.nan
        for nlt in range(5, 7):
            L = len(d.loc[(d['num_leaves_need']==nlt) & (d['need_type4'].notna()) & (d['need_type5'].isna())])
            if L > 0:
                d.loc[(d['num_leaves_need'] == nlt) & (d['need_type4'].notna()) & (d['need_type5'].isna()), 'need_type5'] = \
                    d.loc[(d['num_leaves_need'] == nlt) & (d['need_type4'].notna()) & (d['need_type5'].isna()), ['need_type', 'need_type2', 'need_type3', 'need_type4']+chars].\
                        apply(lambda x: dcp.columns[simulate_wof(get_marginal_probs(get_adj_cps(dcp, x[4:])[x[3]], [otypes.index(x[k]) for k in range(4)]))], axis=1)
            else:
                pass

        # Impute types for last need_type6 for num_leaves_need = 6
        # this is just the only left leave type not selected by previous simulation
        # no logical restriction needed if all 6 types exist
        d['need_type6']=np.nan
        L = len(d.loc[(d['num_leaves_need']==6) & (d['need_type5'].notna()) & (d['need_type6'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_need'] == 6) & (d['need_type5'].notna()) & (d['need_type6'].isna()), 'need_type6'] = \
            d.loc[(d['num_leaves_need'] == 6) & (d['need_type5'].notna()) & (d['need_type6'].isna()), ['need_type', 'need_type2', 'need_type3', 'need_type4', 'need_type5']] \
            .apply(lambda x: list(set(otypes) - set(x))[0], axis=1)
        else:
            pass

        # Save data
        d.to_csv(self.fp_fmla_out, index=False)
        print('File saved: clean FMLA data file before CPS imputation.')
        return None

    def impute_fmla_cps(self, fp_cps_in, fp_cps_out):
        '''

        :param fp_cps: file path to original CPS data
        '''
        t0 = time()
        # Read in processed FMLA data and raw CPS data, and clean CPS data
        d = pd.read_csv(self.fp_fmla_out, low_memory=False)
        cps = pd.read_csv(fp_cps_in)

        cps['female'] = np.where(cps['a_sex'] == 2, 1, 0)
        cps['female'] = np.where(cps['a_sex'].isna(), np.nan, cps['female'])
        cps['black'] = np.where(cps['prdtrace'] == 2, 1, 0)
        cps['black'] = np.where(cps['prdtrace'].isna(), np.nan, cps['black'])
        cps['age'] = cps['a_age']
        cps['agesq'] = cps['age'] * cps['age']
        cps['BA'] = np.where(cps['a_hga'] == 43, 1, 0)
        cps['BA'] = np.where(cps['a_hga'].isna(), np.nan, cps['BA'])
        cps['GradSch'] = np.where((cps['a_hga'] <= 46) & (cps['a_hga'] >= 44), 1, 0)
        cps['GradSch'] = np.where(cps['a_hga'].isna(), np.nan, cps['GradSch'])
        for i in cps['a_mjind'].value_counts().sort_index().index:
            cps['ind_%s' % i] = np.where(cps['a_mjind'] == i, 1, 0)
            cps['ind_%s' % i] = np.where(cps['a_mjind'].isna(), np.nan, cps['ind_%s' % i])
        for i in cps['a_mjocc'].value_counts().sort_index().index:
            cps['occ_%s' % i] = np.where(cps['a_mjocc'] == i, 1, 0)
            cps['occ_%s' % i] = np.where(cps['a_mjocc'].isna(), np.nan, cps['occ_%s' % i])

        cps['hourly'] = np.where(cps['a_hrlywk'] == 1, 1, 0)
        cps['hourly'] = np.where((cps['a_hrlywk'] == 0) | (cps['a_hrlywk'].isna()), np.nan, cps['hourly'])
        cps['empsize'] = cps['noemp']
        cps['oneemp'] = np.where(cps['phmemprs'] == 1, 1, 0)
        cps['oneemp'] = np.where(cps['phmemprs'].isna(), np.nan, cps['oneemp'])

        cps = cps[['female', 'black', 'age', 'agesq', 'BA', 'GradSch'] +
                  ['ind_%s' % x for x in range(1, 14)] +
                  ['occ_%s' % x for x in range(1, 11)] +
                  ['hourly', 'empsize', 'oneemp', 'wkswork', 'marsupwt']]  # remove armed forces code from ind/occ
        cps = cps.dropna(how='any')
        '''
        do logit below:
        y: paid hourly, employer size, single employer last year, weeks worked (subject to FMLA weeks worked categories if any)
        x: chars - female, black, age, agesq, BA, GradSch, all occ codes, all ind codes

        '''

        # logit regressions
        X = cps[['female', 'black', 'age', 'agesq', 'BA', 'GradSch']]
        w = cps['marsupwt']
        Xd = fillna_binary(d[X.columns])
        # cps based hourly paid indicator
        y = cps['hourly']
        clf = sklearn.linear_model.LogisticRegression(solver='liblinear').fit(X, y, sample_weight=w)
        d['hourly_fmla'] = pd.Series(clf.predict(Xd))
        # one employer last year
        y = cps['oneemp']
        clf = sklearn.linear_model.LogisticRegression(solver='liblinear').fit(X, y, sample_weight=w)
        d['oneemp_fmla'] = pd.Series(clf.predict(Xd))
        # ordered logit - TODO: mord package not ideal for handling empsize, cannot impute all 6 levels, best is LogisticAT().
        # employer size
        y = cps['empsize']
        clf = mord.LogisticAT().fit(X, y)
        d['empsize_fmla'] = pd.Series(clf.predict(Xd))
        # regression
        # weeks worked
        y = cps['wkswork']
        clf = sklearn.linear_model.LinearRegression().fit(X, y, sample_weight=w)
        d['wkswork_fmla'] = pd.Series(clf.predict(Xd))

        # Save files
        d.to_csv(self.fp_fmla_out, index=False)
        cps.to_csv(fp_cps_out, index=False)
        progress = 'CPS cleaned and CPS variable imputation done for FMLA. Time elapsed = %s seconds' % (round(time() - t0, 2))
        print(progress)
        print('File saved: clean FMLA data file after CPS imputation.')
        return progress

    def get_length_distribution(self, fp_length_distribution_out):
        '''

        :param fp_length_distribution_out: file path to json outfile of leave length distribution estimated from FMLA
        '''
        # Read in cleaned FMLA data
        d = pd.read_csv(self.fp_fmla_out, low_memory=False)

        # A dictionary to store results
        dct = {}
        types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
        # For PFL states
        dct_PFL = {}
        for t in types:
            xps = d[(d['recStatePay'] == 1) & (d['take_type'] == t)][['weight', 'length']].groupby('length').sum()
            xps = (xps / xps.sum()).sort_index()
            xps.columns = ['']
            xps = [(i, v.values[0]) for i, v in xps.iterrows()]
            dct_PFL[t] = xps
        dct['PFL'] = dct_PFL
        # For non-PFL states
        dct_NPFL = {}
        for t in types:
            xps = d[(d['recStatePay'] == 0) & (d['take_type'] == t)][['weight', 'length']].groupby('length').sum()
            xps = (xps / xps.sum()).sort_index()
            xps.columns = ['']
            xps = [(i, v.values[0]) for i, v in xps.iterrows()]
            dct_NPFL[t] = xps
        dct['non-PFL'] = dct_NPFL

        # Save as json
        jsn = json.dumps(dct)
        f = open(fp_length_distribution_out, 'w')
        f.write(jsn)
        f.close()
        print('File saved: leave distribution estimated from FMLA data.')
        return None





