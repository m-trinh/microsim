import pandas as pd
import numpy as np
import os


def clean_fmla(file_path, save_result=True):
    # Read in FMLA data
    d = pd.read_csv(file_path)

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
    d['reason_take_rev'] = np.where((np.isnan(d['A20']) == False) & (d['A20'] == 2), d['A5_2_CAT_REV'],
                                    d['A5_1_CAT_rev'])

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
    d['doctor1'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & (d['LEAVE_CAT'] == 2), d['doctor_need'],
                            d['doctor_take'])
    d['doctor2'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 4)),
                            d['doctor_need'], d['doctor_take'])

    d['hospital1'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & (d['LEAVE_CAT'] == 2), d['hospital_need'],
                              d['hospital_take'])
    d['hospital2'] = np.where((np.isnan(d['LEAVE_CAT']) == False) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 4)),
                              d['hospital_need'], d['hospital_take'])

    d['doctor'] = d[['doctor_need', 'doctor_take']].apply(lambda x: max(x[0], x[1]), axis=1)
    d['hospital'] = d[['hospital_need', 'hospital_take']].apply(lambda x: max(x[0], x[1]), axis=1)

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

    # leave taken
    d['taker'] = np.where((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 4), 1, 0)
    d['needer'] = np.where((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 4), 1, 0)

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
    d['take_matdis'] = np.where(
        ((d['A5_1_CAT'] == 21) & (d['A11_1'] == 1) & (d['GENDER_CAT'] == 2)) & ((d['A20'] != 2) | (d['A20'].isna())) | (
                d['A5_1_CAT_rev'] == 32), 1, 0)
    d['take_matdis'] = np.where(np.isnan(d['take_matdis']), 0, d['take_matdis'])
    d['take_matdis'] = np.where((np.isnan(d['A5_1_CAT'])) & (np.isnan(d['A5_2_CAT'])), np.nan, d['take_matdis'])
    d['take_matdis'] = np.where(np.isnan(d['take_matdis']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                                d['take_matdis'])

    d['need_matdis'] = np.where(
        ((d['B6_1_CAT'] == 21) & (d['B12_1'] == 1) & (d['GENDER_CAT'] == 2)) | (d['B6_1_CAT_rev'] == 32), 1, 0)
    d['need_matdis'] = np.where(np.isnan(d['need_matdis']), 0, d['need_matdis'])
    d['need_matdis'] = np.where(np.isnan(d['B6_1_CAT']), np.nan, d['need_matdis'])
    d['need_matdis'] = np.where(np.isnan(d['need_matdis']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                                d['need_matdis'])

    d['type_matdis'] = np.where((d['take_matdis'] == 1) | (d['need_matdis'] == 1), 1, 0)
    d['type_matdis'] = np.where(np.isnan(d['take_matdis']) | np.isnan(d['need_matdis']), np.nan, d['type_matdis'])

    # new child/bond
    d['take_bond'] = np.where((d['A5_1_CAT'] == 21) & ((np.isnan(d['A11_1'])) | (d['GENDER_CAT'] == 1) | (
            (d['GENDER_CAT'] == 2) & (d['A11_1'] == 2) & (d['A5_1_CAT_rev'] != 32))) & (
                                      (d['A20'] != 2) | (d['A20'].isna())), 1, 0)
    d['take_bond'] = np.where(np.isnan(d['A5_1_CAT']), np.nan, d['take_bond'])
    d['take_bond'] = np.where(np.isnan(d['take_bond']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                              d['take_bond'])

    d['need_bond'] = np.where((d['B6_1_CAT'] == 21) & (np.isnan(d['B12_1']) | (d['GENDER_CAT'] == 1) | (
            (d['GENDER_CAT'] == 2) & (d['B12_1'] == 2) & (d['B6_1_CAT_rev'] != 32))), 1, 0)
    d['need_bond'] = np.where(np.isnan(d['B6_1_CAT']), np.nan, d['need_bond'])
    d['need_bond'] = np.where(np.isnan(d['need_bond']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                              d['need_bond'])

    d['type_bond'] = np.where((d['take_bond'] == 1) | (d['need_bond'] == 1), 1, 0)
    d['type_bond'] = np.where(np.isnan(d['take_bond']) | np.isnan(d['need_bond']), np.nan, d['type_bond'])

    # own health
    d['take_own'] = np.where(d['reason_take'] == 1, 1, 0)
    d['take_own'] = np.where((d['reason_take'].isna()), np.nan, d['take_own'])
    d['take_own'] = np.where((np.isnan(d['take_own'])) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                             d['take_own'])

    d['need_own'] = np.where((d['B6_1_CAT'] == 1) | (d['B6_2_CAT'] == 1), 1, 0)
    d['need_own'] = np.where((d['B6_1_CAT'].isna()) & (d['B6_2_CAT'].isna()), np.nan, d['need_own'])
    # multiple leaves - if one leave is Own the other NA, need_own is NA
    d['need_own'] = np.where((d['B6_1_CAT'].isna()) & (d['B6_2_CAT'] != 1), np.nan, d['need_own'])
    d['need_own'] = np.where((d['B6_1_CAT'] != 1) & (d['B6_2_CAT'].isna()), np.nan, d['need_own'])
    d['need_own'] = np.where(np.isnan(d['need_own']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                             d['need_own'])

    d['type_own'] = np.where((d['take_own'] == 1) | (d['need_own'] == 1), 1, 0)
    d['type_own'] = np.where(np.isnan(d['take_own']) | np.isnan(d['need_own']), np.nan, d['type_own'])

    # ill child
    d['take_illchild'] = np.where(d['reason_take'] == 11, 1, 0)
    d['take_illchild'] = np.where(d['reason_take'].isna(), np.nan, d['take_illchild'])
    d['take_illchild'] = np.where(np.isnan(d['take_illchild']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                                  d['take_illchild'])

    d['need_illchild'] = np.where((d['B6_1_CAT'] == 11) | (d['B6_2_CAT'] == 11), 1, 0)
    # multiple leaves - if one leave is not Illchild the other NA, need_illchild is NA
    d['need_illchild'] = np.where((d['B6_1_CAT'].isna()) & (d['B6_2_CAT'] != 11), np.nan, d['need_illchild'])
    d['need_illchild'] = np.where((d['B6_1_CAT'] != 11) & (d['B6_2_CAT'].isna()), np.nan, d['need_illchild'])
    d['need_illchild'] = np.where(np.isnan(d['need_illchild']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                                  d['need_illchild'])

    d['type_illchild'] = np.where((d['take_illchild'] == 1) | (d['need_illchild'] == 1), 1, 0)
    d['type_illchild'] = np.where(np.isnan(d['take_illchild']) | np.isnan(d['need_illchild']), np.nan,
                                  d['type_illchild'])

    # ill spouse
    d['take_illspouse'] = np.where(d['reason_take'] == 12, 1, 0)
    d['take_illspouse'] = np.where(d['reason_take'].isna(), np.nan, d['take_illspouse'])
    d['take_illspouse'] = np.where(np.isnan(d['take_illspouse']) & ((d['LEAVE_CAT'] == 2) | (d['LEAVE_CAT'] == 3)), 0,
                                   d['take_illspouse'])

    d['need_illspouse'] = np.where(d['B6_1_CAT'] == 12, 1, 0)
    d['need_illspouse'] = np.where(d['B6_1_CAT'].isna(), np.nan, d['need_illspouse'])
    d['need_illspouse'] = np.where(np.isnan(d['need_illspouse']) & ((d['LEAVE_CAT'] == 1) | (d['LEAVE_CAT'] == 3)), 0,
                                   d['need_illspouse'])

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

    # number of reasons leaves taken
    d['num_leaves_taken'] = d['A4a_CAT']
    d['num_leaves_taken'] = np.where(d['num_leaves_taken'].isna(), 0, d['num_leaves_taken'])

    # number of reasons leaves needed
    d['num_leaves_need'] = d['B5_CAT']
    d['num_leaves_need'] = np.where(d['num_leaves_need'].isna(), 0, d['num_leaves_need'])

    # leave length by leave type
    types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
    for t in types:
        d['length_%s' % t] = np.where(d['take_%s' % t] == 1, d['length'], 0)
        d['length_%s' % t] = np.where(d['take_%s' % t].isna(), np.nan, d['length_%s' % t])

    # most recent leave type in 1 col

    d['type_recent'] = np.nan
    d.loc[(d['type_recent'].isna()) & (d['reason_take'] == 1), 'type_recent'] = 'own'
    d.loc[(d['type_recent'].isna()) & (d['reason_take'] == 11), 'type_recent'] = 'illchild'
    d.loc[(d['type_recent'].isna()) & (d['reason_take'] == 12), 'type_recent'] = 'illspouse'
    d.loc[(d['type_recent'].isna()) & (d['reason_take'] == 13), 'type_recent'] = 'illparent'
    d.loc[(d['type_recent'].isna()) & (d['type_matdis'] == 1), 'type_recent'] = 'matdis'
    d.loc[(d['type_recent'].isna()) & (d['type_bond'] == 1), 'type_recent'] = 'bond'

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

    # Assume all takers/needers with ongoing condition are 'cornered' and would respond with longer leaves
    # A10_1, A10_2: regular/ongoing condition, takers and dual
    # B11_1, B11_2: regular/ongoing condition, needers and dual
    d.loc[(d['resp_len'].isna()) & (d['A10_1'] == 2) | (d['A10_1'] == 3)
          | (d['B11_1'] == 2) | (d['B11_1'] == 3), 'resp_len'] = 1

    # B15_1_CAT and B15_2_CAT only has one group (cod = 5) identified as constrained by unaffordability
    # These financially-constrained workers were assigned resp_len=1 above, all other cornered workers would not respond
    # Check reasons of no leave among rest: d[d['resp_len'].isna()].B15_1_CAT.value_counts().sort_index()
    # all reasons unsolved by replacement generosity
    d.loc[(d['resp_len'].isna()) & (d['B15_1_CAT'].notna()), 'resp_len'] = 0
    d.loc[(d['resp_len'].isna()) & (d['B15_2_CAT'].notna()), 'resp_len'] = 0

    # Check LEAVE_CAT of rest: d[d['resp_len'].isna()]['LEAVE_CAT'].value_counts().sort_index()
    # 267 takers and 3 needers
    # with no evidence in data of need solvable / unsolvable by $, assume solvable to be conservative
    d.loc[d['resp_len'].isna(), 'resp_len'] = 1

    # Save data
    if save_result:
        save_clean_file(d, file_path)

    return d


def clean_acs(household_file_path, person_file_path, save_result=True):
    # -------------------------- #
    # ACS Household File
    # -------------------------- #

    # Load data
    d_hh = pd.read_csv(household_file_path)

    # Create Variables
    d_hh["nochildren"] = pd.get_dummies(d_hh["FPARC"])[4]
    d_hh['faminc'] = d_hh['FINCP'] * d_hh[
        'ADJINC'] / 1042852 / 1000  # adjust to 2012 thousand dollars to conform with FMLA 2012 data
    d_hh.loc[(d_hh[
                  'faminc'] <= 0), 'faminc'] = 0.01 / 1000  # force non-positive income to be epsilon to get meaningful log-income
    d_hh["lnfaminc"] = np.log(d_hh["faminc"])

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
    for d in pd.read_csv(person_file_path, chunksize=chunk_size):

        # Merge with the household level variables
        d = pd.merge(d, d_hh[['SERIALNO', 'nochildren', 'faminc', 'lnfaminc', 'PARTNER', 'ndep_kid', 'ndep_old']],
                     on='SERIALNO')

        # Rename ACS variables to be consistent with FMLA data
        rename_dic = {'AGEP': 'age'}
        d.rename(columns=rename_dic, inplace=True)

        # duplicating age column for meshing with CPS estimate output
        d['a_age'] = d['age']

        # Create new ACS Variables
        d['married'] = pd.get_dummies(d["MAR"])[1]
        d["widowed"] = pd.get_dummies(d["MAR"])[2]
        d["divorced"] = pd.get_dummies(d["MAR"])[3]
        d["separated"] = pd.get_dummies(d["MAR"])[4]
        d["nevermarried"] = pd.get_dummies(d["MAR"])[5]
        # use PARTNER in household data to tease out unmarried partners
        d['partner'] = np.where((d['PARTNER'] == 1) | (d['PARTNER'] == 2) | (d['PARTNER'] == 3) | (d['PARTNER'] == 4),
                                1, 0)
        for m in ['married', 'widowed', 'divorced', 'separated', 'nevermarried']:
            d.loc[d['partner'] == 1, m] = 0

        d["male"] = pd.get_dummies(d["SEX"])[1]
        d["female"] = 1 - d["male"]
        d["agesq"] = d["age"] ** 2

        # Educational level
        d['sku'] = np.where(d['SCHL'].isna(), 0, d['SCHL'])
        d['ltHS'] = np.where(d['sku'] <= 11, 1, 0)
        d['someHS'] = np.where((d['sku'] >= 12) & (d['sku'] <= 15), 1, 0)
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
        d['wage12'] = d['WAGP'] * d['ADJINC'] / 1042852
        d['lnearn'] = np.nan
        d.loc[d['wage12'] > 0, 'lnearn'] = d.loc[d['wage12'] > 0, 'wage12'].apply(lambda x: np.log(x))

        # health insurance from employer
        d['hiemp'] = np.where(d['HINS1'] == 1, 1, 0)
        d['hiemp'] = np.where(d['HINS1'].isna(), np.nan, d['hiemp'])

        # Employment at government
        # missing = age<16, or NILF over 5 years, or never worked
        d['empgov_fed'] = np.where(d['COW'] == 5, 1, 0)
        d['empgov_fed'] = np.where(np.isnan(d['COW']), np.nan, d['empgov_fed'])
        d['empgov_st'] = np.where(d['COW'] == 4, 1, 0)
        d['empgov_st'] = np.where(np.isnan(d['COW']), np.nan, d['empgov_st'])
        d['empgov_loc'] = np.where(d['COW'] == 3, 1, 0)
        d['empgov_loc'] = np.where(np.isnan(d['COW']), np.nan, d['empgov_loc'])

        # Presence of children for females
        d['fem_cu6'] = np.where(d['PAOC'] == 1, 1, 0)
        d['fem_c617'] = np.where(d['PAOC'] == 2, 1, 0)
        d['fem_cu6and617'] = np.where(d['PAOC'] == 3, 1, 0)
        d['fem_nochild'] = np.where(d['PAOC'] == 4, 1, 0)
        for x in ['fem_cu6', 'fem_c617', 'fem_cu6and617', 'fem_nochild']:
            d.loc[d['PAOC'].isna(), x] = np.nan

        # Occupation

        # make numeric OCCP = OCCP10 if ACS 2011-2015, or OCCP12 if ACS 2012-2016
        if 'N.A.' in d['OCCP12'].value_counts().index:
            d.loc[d['OCCP12'] == 'N.A.', 'OCCP12'] = d.loc[d['OCCP12'] == 'N.A.', 'OCCP12'].apply(lambda x: np.nan)
        d.loc[d['OCCP12'].notna(), 'OCCP12'] = d.loc[d['OCCP12'].notna(), 'OCCP12'].apply(lambda x: int(x))

        if 'N.A.' in d['OCCP10'].value_counts().index:
            d.loc[d['OCCP10'] == 'N.A.', 'OCCP10'] = d.loc[d['OCCP10'] == 'N.A.', 'OCCP10'].apply(lambda x: np.nan)
        d.loc[d['OCCP10'].notna(), 'OCCP10'] = d.loc[d['OCCP10'].notna(), 'OCCP10'].apply(lambda x: int(x))

        d['OCCP'] = np.nan
        d['OCCP'] = np.where(d['OCCP12'].notna(), d['OCCP12'], d['OCCP'])
        d['OCCP'] = np.where((d['OCCP'].isna()) & (d['OCCP12'].isna()) & (d['OCCP10'].notna()), d['OCCP10'], d['OCCP'])

        d['occ_1'] = 0
        d['occ_2'] = 0
        d['occ_3'] = 0
        d['occ_4'] = 0
        d['occ_5'] = 0
        d['occ_6'] = 0
        d['occ_7'] = 0
        d['occ_8'] = 0
        d['occ_9'] = 0
        d['occ_10'] = 0
        d['maj_occ'] = 0
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

        # Industry
        d['ind_1'] = 0
        d['ind_2'] = 0
        d['ind_3'] = 0
        d['ind_4'] = 0
        d['ind_5'] = 0
        d['ind_6'] = 0
        d['ind_7'] = 0
        d['ind_8'] = 0
        d['ind_9'] = 0
        d['ind_10'] = 0
        d['ind_11'] = 0
        d['ind_12'] = 0
        d['ind_13'] = 0
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
        hr_est = pd.read_csv('estimates/CPS_paid_hrly.csv').set_index('var').to_dict()['est']
        d['prhourly'] = 0
        for dem in hr_est.keys():
            if dem != 'Intercept':
                d['prhourly'] += d[dem].fillna(0) * hr_est[dem]
        d['prhourly'] += hr_est['Intercept']
        d['prhourly'] = np.exp(d['prhourly']) / (1 + np.exp(d['prhourly']))
        d['rand'] = pd.Series(np.random.rand(d.shape[0]), index=d.index)
        d['hourly'] = np.where(d['prhourly'] > d['rand'], 1, 0)
        d = d.drop('rand', axis=1)

        # -------------------------- #
        # Save the resulting dataset
        # -------------------------- #
        cols = ['SERIALNO', 'PWGTP', 'ST',
                'employed', 'empgov_fed', 'empgov_st', 'empgov_loc',
                'wkhours', 'weeks_worked_cat', 'wage12', 'lnearn', 'hiemp',
                'a_age', 'age', 'agesq',
                'male', 'female',
                'nochildren', 'ndep_kid', 'ndep_old',
                'ltHS', 'someHS', 'HSgrad', 'someCol', 'BA', 'GradSch', 'noHSdegree', 'BAplus',
                'faminc', 'lnfaminc',
                'married', 'partner', 'separated', 'divorced', 'widowed', 'nevermarried',
                'asian', 'black', 'white', 'native', 'other', 'hisp',
                'fem_cu6', 'fem_cu6and617', 'fem_c617', 'fem_nochild',
                'ESR', 'COW'  # original ACS vars for restricting samples later
                ]
        cols += ['ind_%s' % x for x in range(1, 14)]
        cols += ['OCCP'] + ['occ_%s' % x for x in range(1, 11)]
        # cols += ['hourly'] # optional, move imputation of hourly to somewhere else?

        d_reduced = d[cols]
        dout = dout.append(d_reduced)
        print('ACS data cleaned for chunk %s of personal data...' % ichunk)
        ichunk += 1

    if save_result:
        save_clean_file(dout, person_file_path)

    # a wage only file for testing ABF
    # d2 = d_reduced[['wage12','PWGTP']]
    # d2.to_csv("./data/acs/ACS_cleaned_forsimulation_%s_wage.csv" % st, index=False, header=True)


def save_clean_file(df, file_path):
    file_name, _ = os.path.splitext(file_path)
    file_name += '_clean'
    df.to_csv(file_name + '.csv', index=False, header=True)
