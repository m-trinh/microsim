'''
main simulation engine
chris zhang 3/4/2019
'''
import pandas as pd
import numpy as np
import bisect
import json
from time import time
from _5a_aux_functions import get_columns, get_sim_col
import sklearn.linear_model, sklearn.naive_bayes, sklearn.neighbors, sklearn.tree, sklearn.ensemble, \
    sklearn.gaussian_process, sklearn.svm
import math
from datetime import datetime
import matplotlib.pyplot as plt
import os
import csv
from _1_clean_FMLA import DataCleanerFMLA
from _4_clean_ACS import DataCleanerACS
from Utils import format_chart
import random

class SimulationEngine:

    def __init__(self, st, yr, fps_in, fps_out, clf_name, prog_para, engine_type='Main', q=None):
        '''
        :param st: state name, 'ca', 'ma', etc.
        :param yr: end year of 5-year ACS
        :param fps_in: filepaths of infiles (FMLA, ACS h, ACS p, CPS)
        :param fps_out: filepaths of outfiles files (FMLA, CPS, ACS, length distribution, master ACS post sim)
        :param clf_name: classifier name
        :param prog_para: program parameters user specified
        '''

        self.st = st
        self.yr = yr
        self.fp_fmla_in = fps_in[0]
        self.fp_cps_in = fps_in[1]
        self.fp_acsh_in = fps_in[2] # directory only for ACS household file
        self.fp_acsp_in = fps_in[3] # directory only for ACS person file
        self.fp_fmla_out = fps_out[0]
        self.fp_cps_out = fps_out[1]
        self.fp_acs_out = fps_out[2] # directory only for cleaned ACS file
        self.fp_length_distribution_out = fps_out[3]
        self.clf_name = clf_name
        self.elig_wage12 = prog_para[0] # min annual wage
        self.elig_wkswork = prog_para[1] # min annual work weeks
        self.elig_yrhours = prog_para[2] # min annual work hours
        self.elig_empsize = prog_para[3] # min employer size - will be categorized as bin later
        self.rrp = prog_para[4] # proposed wage replacement ratio
        self.wkbene_cap = prog_para[5]
        self.d_maxwk = prog_para[6] # dict from types to max week of benefits
        self.d_takeup = prog_para[7] # dict from types to take up rates
        self.incl_empgov_fed = prog_para[8]
        self.incl_empgov_st = prog_para[9]
        self.incl_empgov_loc = prog_para[10]
        self.incl_empself = prog_para[11]
        self.sim_method = prog_para[12]
        self.needers_fully_participate = prog_para[13]
        self.state_of_work = prog_para[14]
        self.weight_factor = prog_para[15]
        self.dual_receivers_share = prog_para[16]

        # leave types
        self.types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']

        # a dict from clf_name to clf
        self.d_clf = {}
        self.d_clf['Logistic Regression'] = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto')
        self.d_clf['Ridge Classifier'] = sklearn.linear_model.RidgeClassifier()
        #self.d_clf['Stochastic Gradient Descent'] = sklearn.linear_model.SGDClassifier(loss='modified_huber', max_iter=1000, tol=0.001)
        self.d_clf['Naive Bayes'] = sklearn.naive_bayes.MultinomialNB()
        self.d_clf['Support Vector Machine'] = sklearn.svm.SVC(probability=True, gamma='auto')
        self.d_clf['Random Forest'] = sklearn.ensemble.RandomForestClassifier()
        self.d_clf['K Nearest Neighbor'] = sklearn.neighbors.KNeighborsClassifier()

        # out id for creating unique out folder to store all model outputs
        self.out_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.updates = []
        self.progress = 0
        self.figure = None
        self.engine_type = engine_type
        self.q = q
        if self.engine_type == 'main':
            self.output_directory = './output/output_%s' % self.out_id
        else:
            self.output_directory = './output/output_%s_%s' % (self.out_id, self.engine_type)

        # POW population weight multiplier
        # TODO: wrap this with Weight Factor in GUI, and apply final factor to get_cost()
        self.pow_pop_multiplier = 1.0217029934467345 # based on 2012-2016 ACS, see project acs_all


    def save_program_parameters(self):

        # create output folder
        os.makedirs(self.output_directory)

        # save meta file of program parameters
        para_labels = ['State', 'Year',
                       'Minimum Annual Wage','Minimum Annual Work Weeks','Minimum Annual Work Hours',
                       'Minimum Employer Size','Proposed Wage Replacement Ratio','Weekly Benefit Cap',
                       'Include Goverment Employees, Federal',
                       'Include Goverment Employees, State',
                       'Include Goverment Employees, Local',
                       'Include Self-employed',
                       'Simulation Method']
        para_labels_m = ['Maximum Week of Benefit Receiving',
                         'Take Up Rates'] # type-specific parameters

        para_values = [self.st.upper(),self.yr + 2000,
                       self.elig_wage12,self.elig_wkswork,self.elig_yrhours,self.elig_empsize,self.rrp,self.wkbene_cap,
                       self.incl_empgov_fed, self.incl_empgov_st,self.incl_empgov_loc,self.incl_empself, self.clf_name]
        para_values_m = [self.d_maxwk, self.d_takeup]

        d = pd.DataFrame(para_values, index=para_labels)
        dm = pd.DataFrame.from_dict(para_values_m)
        dm = dm.rename(index=dict(zip(dm.index, para_labels_m)))

        with open('%s/prog_para_%s.csv' % (self.output_directory, self.out_id), 'w', newline='') as f:
            writer = csv.writer(f)
            for idx, row in d.iterrows():
                writer.writerow((idx, row[0]))
            writer.writerow([])
            writer.writerow([''] + list(dm.columns))
            for idx, row in dm.iterrows():
                writer.writerow(([idx] + list(row.values)))
        return None

    def prepare_data(self):
        dcf = DataCleanerFMLA(self.fp_fmla_in, self.fp_fmla_out)
        dcf.clean_data()
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 10})
        self.__put_queue({'type': 'message', 'engine': self.engine_type,
                          'value': 'File saved: clean FMLA data file before CPS imputation.'})
        message = dcf.impute_fmla_cps(self.fp_cps_in, self.fp_cps_out)
        self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 20})
        self.__put_queue({'type': 'message', 'engine': self.engine_type,
                          'value': 'File saved: clean FMLA data file after CPS imputation.'})
        dcf.get_length_distribution(self.fp_length_distribution_out)
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 25})
        self.__put_queue({'type': 'message', 'engine': self.engine_type,
                          'value': 'File saved: leave distribution estimated from FMLA data.'})

        self.__put_queue({'type': 'message', 'engine': self.engine_type,
                          'value': 'Cleaning ACS data. State chosen = RI. Chunk size = 100000 ACS rows'})
        dca = DataCleanerACS(self.st, self.yr, self.fp_acsh_in, self.fp_acsp_in, self.fp_acs_out, self.state_of_work)
        dca.load_data()
        message = dca.clean_person_data()
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 60})
        self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
        return None

    def get_acs_simulated(self):
        tsim = time()

        # Read in cleaned ACS and FMLA data, and FMLA-based length distribution
        acs = pd.read_csv(self.fp_acs_out + 'ACS_cleaned_forsimulation_20%s_%s.csv' % (self.yr, self.st))
        pfl = 'non-PFL'  # status of PFL as of ACS sample period
        d = pd.read_csv(self.fp_fmla_out, low_memory=False)
        with open(self.fp_length_distribution_out) as f:
            flen = json.load(f)

        # Define classifier
        clf = self.d_clf[self.clf_name]

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

        # Conditional simulation - anypay, doctor, hospital for taker/needer sample
        acs['taker'] = [max(z) for z in acs[['take_%s' % t for t in self.types]].values]
        acs['needer'] = [max(z) for z in acs[['need_%s' % t for t in self.types]].values]
        X = d[(d['taker'] == 1) | (d['needer'] == 1)][col_Xs]
        w = d.loc[X.index][col_w]
        Xa = acs[(acs['taker'] == 1) | (acs['needer'] == 1)][X.columns]
        if len(Xa) == 0:
            print('Warning: Neither leave taker nor leave needer present in simulated ACS persons. '
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
            y = [D_prop[x] for x in d.loc[X.index]['prop_pay']]
            yhat = get_sim_col(X, y, w, Xa, clf)
            # prop_pay labels are from 1 to 6, get_sim_col() vectorization sum gives 0~5, increase label by 1
            yhat = pd.Series(data=yhat.values + 1, index=yhat.index, name='prop_pay')
            acs = acs.join(yhat)
            acs.loc[acs['prop_pay'].notna(), 'prop_pay'] = [d_prop[x] for x in
                                                            acs.loc[acs['prop_pay'].notna(), 'prop_pay']]

        # Draw status-quo leave length for each type
        # Without-program lengths - draw from FMLA-based distribution (pfl indicator = 0)
        # note: here, cumsum/bisect is 20% faster than np/choice.
        # But when simulate_wof applied as lambda to df, np/multinomial is 5X faster!
        t0 = time()
        for t in self.types:
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
        for t in self.types:
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

        # sample restriction
        acs = acs.drop(acs[(acs['taker'] == 0) & (acs['needer'] == 0)].index)

        if not self.incl_empgov_fed:
            acs = acs.drop(acs[acs['empgov_fed'] == 1].index)
        if not self.incl_empgov_st:
            acs = acs.drop(acs[acs['empgov_st'] == 1].index)
        if not self.incl_empgov_loc:
            acs = acs.drop(acs[acs['empgov_loc'] == 1].index)
        if not self.incl_empself:
            acs = acs.drop(acs[(acs['COW'] == 6) | (acs['COW'] == 7)].index)

        # program eligibility
        acs['elig_prog'] = 0

        elig_empsizebin = 0
        if 1 <= self.elig_empsize < 10:
            elig_empsizebin = 1
        elif 10 <= self.elig_empsize <= 49:
            elig_empsizebin = 2
        elif 50 <= self.elig_empsize <= 99:
            elig_empsizebin = 3
        elif 100 <= self.elig_empsize <= 499:
            elig_empsizebin = 4
        elif 500 <= self.elig_empsize <= 999:
            elig_empsizebin = 5
        elif self.elig_empsize >= 1000:
            elig_empsizebin = 6

        acs.loc[(acs['wage12'] >= self.elig_wage12) &
                (acs['wkswork'] >= self.elig_wkswork) &
                (acs['wkswork'] * acs['wkhours'] >= self.elig_yrhours) &
                (acs['empsize'] >= elig_empsizebin), 'elig_prog'] = 1

        # keep only eligible population
        acs = acs.drop(acs[acs['elig_prog'] != 1].index)

        # Given fraction of dual receiver x, simulate dual/single receiver status
        # With state program:
        # if anypay = 0, must be single receiver
        # let %(anypay=0) = a, %single-receiver specified must satisfy (1-x) >= a, i.e. x <= (1-a)
        s_no_emp_pay = acs[acs['anypay'] == 0]['PWGTP'].sum() / acs['PWGTP'].sum()
        x = self.dual_receivers_share  # specified share of double receiver
        x = min(1 - s_no_emp_pay, x)  # cap x at (1 - %(anypay=0))
        # simulate double receiver status
        # we need x/(1-a) share of double receiver from (1-a) of all eligible workers who have anypay=1
        acs['z'] = [random.random() for x in range(len(acs))]
        acs['dual_receiver'] = (acs['z'] < (x / (1 - s_no_emp_pay))).astype(int)
        acs.loc[acs['anypay'] == 0, 'dual_receiver'] = 0
        # treat each ACS row equally and check if post-sim weighted share = x
        # using even a small state RI shows close to equality
        s_dual_receiver = acs[acs['dual_receiver'] == 1]['PWGTP'].sum() / acs['PWGTP'].sum()
        s_dual_receiver = round(s_dual_receiver, 2)
        print('Specified share of dual-receiver = %s. Post-sim weighted share = %s' % (x, s_dual_receiver))

        # Simulate counterfactual leave lengths (cf-len) for dual receivers
        # Given cf-len, get cp-len
        # With program, effective rr =min(rre+rrp, 1), assuming responsiveness diminishes if full replacement attainable
        for t in self.types:
            acs['cfl_%s' % t] = np.nan
            ## Get cf-len for dual receivers
            # use [(rre, sql), (1, mnl)] to interpolate cfl at rre+rrp, regardless rre+rrp<1 or not
            acs.loc[acs['dual_receiver'] == 1, 'cfl_%s' % t] = \
                acs.loc[acs['dual_receiver'] == 1, 'len_%s' % t] + (acs.loc[
                                                                          acs['dual_receiver'] == 1, 'mnl_%s' % t] -
                                                                      acs.loc[
                                                                          acs['dual_receiver'] == 1, 'len_%s' % t]) \
                                                                     * self.rrp / (1 - acs.loc[
                    acs['dual_receiver'] == 1, 'prop_pay'])
            # if rre+rrp>=1, set cfl = mnl
            acs.loc[(acs['dual_receiver'] == 1) & (acs['prop_pay'] >= 1-self.rrp), 'cfl_%s' % t] = \
                acs.loc[(acs['dual_receiver'] == 1) & (acs['prop_pay'] >= 1-self.rrp), 'mnl_%s' % t]
            ## Get covered-by-program leave lengths (cp-len) for dual receivers
            # allocate cf-len between employer and state according to rre/rrp ratio
            acs.loc[acs['dual_receiver'] == 1, 'cpl_%s' % t] = acs.loc[acs['dual_receiver'] == 1, 'cfl_%s' % t] * \
                                                                 self.rrp / (
                self.rrp + acs.loc[acs['dual_receiver'] == 1, 'prop_pay'])


        # Simulate cf-len for single receivers
        # Given cf-len, get cp-len
        for t in self.types:
            # single receiver, rrp>rre. Assume will use state program benefit to replace employer benefit
            acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'cfl_%s' % t] = \
                acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'len_%s' % t] + \
                (acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'mnl_%s' % t] -
                 acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'len_%s' % t]) * \
                (self.rrp - acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'prop_pay']) / \
                (1 - acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'prop_pay'])
            # single receiver, rrp<=rre. Assume will not use any state program benefit
            # so still using same employer benefit as status-quo, thus cf-len = sq-len
            acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] >= self.rrp), 'cfl_%s' % t] = \
                acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] >= self.rrp), 'len_%s' % t]
            # Get covered-by-program leave lengths (cp-len) for single receivers
            # if rrp>rre, cp-len = cf-len
            acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'cpl_%s' % t] = \
                acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] < self.rrp), 'cfl_%s' % t]
            # if rrp<=rre, cp-len = 0
            acs.loc[(acs['dual_receiver'] == 0) & (acs['prop_pay'] >= self.rrp), 'cpl_%s' % t] = 0
            # set cp-len = 0 if missing
            acs.loc[acs['cpl_%s' % t].isna(), 'cpl_%s' % t] = 0

        # Apply cap of coverage period (in weeks) to cpl_type (in days) for each leave type
        for t in self.types:
            acs.loc[acs['cpl_%s' % t] >= 0, 'cpl_%s' % t] = [min(x, 5 * self.d_maxwk[t]) for x in
                                                             acs.loc[acs['cpl_%s' % t] >= 0, 'cpl_%s' % t]]
        # Save ACS data after finishing simulation
        acs.to_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id), index=False)
        message = 'Leaves simulated for 5-year ACS 20%s-20%s in state %s. Time needed = %s seconds' % \
                  ((self.yr-4), self.yr, self.st.upper(), round(time()-tsim, 0))
        print(message)
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 95})
        self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
        return acs

    def get_adjusted_weight(self, takeup_factor, acs, col_w='PWGTP'):
        ## Adjust weights per user input
        ## w = acs weight col
        ## takeup_factor = type-specific takeup factor
        ## acs = post-sim acs
        ## col_w = ACS weight col to use, default is PWGTP, could be rep-w cols like PWGTP1...80, etc.

        # get raw weight
        w = acs[col_w]
        # takeup factor - multiplier on ACS row (incl. non-participants) to reach user-specified takeup rates
        w = w * takeup_factor
        # needers_fully_participate - if True, then apply selective multiplier=1 on ACS rows with needer = 1
        if self.needers_fully_participate:
            w = [x[1] if x[0] == 1 else x[1] * takeup_factor for x in acs[['needer', col_w]].values]
            w = np.array(w)
        # state_of_work - multiplier to account for 'missing workers' with missing POW data in ACS
        if self.state_of_work:
            w = w * self.pow_pop_multiplier
        # weight_factor - multiplier to account for user-specified weight factor (e.g. pop growth)
        w = w * self.weight_factor
        return w


    def get_cost(self):
        # read simulated ACS
        acs = pd.read_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id))
        # apply take up rates and weekly benefit cap, and compute total cost, 6 types
        costs = {}
        for t in self.types:
            # v = capped weekly benefit of leave type
            v = [min(x, self.wkbene_cap) for x in
                 ((acs['cpl_%s' % t] / 5) * (acs['wage12'] / acs['wkswork'] * self.rrp))]
            # w = population that take up benefit of leave type
            # d_takeup[t] is 'official' takeup rate = pop take / pop eligible
            # so pop take = total pop * official take up
            # takeup normalization factor = pop take / pop of ACS rows where take up occurs for leave type
            takeup_factor = acs['PWGTP'].sum() * self.d_takeup[t] / acs[acs['cpl_%s' % t] > 0]['PWGTP'].sum()
            # get adjusted weight
            w = self.get_adjusted_weight(takeup_factor, acs)

            # # apply takeup factor to weights
            # # if needers_fully_participate = 1 is chosen, for needers takeup factor = 1
            # w = acs['PWGTP'] * takeup_factor
            # if self.needers_fully_participate:
            #     w = [x[1] if x[0]==1 else x[1]*takeup_factor for x in acs[['needer', 'PWGTP']].values]
            #     w = np.array(w)
            # # apply pow_pop_multiplier if state_of_work is True
            # if self.state_of_work:
            #     w = w * self.pow_pop_multiplier
            # # apply weight factor
            # w = w * self.weight_factor

            # get program cost for leave type t - sumprod of capped benefit and adjusted weight for each ACS row
            costs[t] = (v * w).sum()
        costs['total'] = sum(list(costs.values()))

        # compute standard error using replication weights, then compute confidence interval
        sesq = dict(zip(costs.keys(), [0]*len(costs.keys())))
        for wt in ['PWGTP%s' % x for x in range(1, 81)]:
            costs_rep = {}
            for t in self.types:
                v = [min(x, self.wkbene_cap) for x in
                     ((acs['cpl_%s' % t] / 5) * (acs['wage12'] / acs['wkswork'] * self.rrp))]
                takeup_factor = acs[wt].sum() * self.d_takeup[t] / acs[acs['cpl_%s' % t] > 0][wt].sum()
                # get adjusted weight
                w = self.get_adjusted_weight(takeup_factor, acs, col_w=wt)
                #w = acs[wt] * takeup_factor
                costs_rep[t] = (v * w).sum()
            costs_rep['total'] = sum(list(costs_rep.values()))
            for k in costs_rep.keys():
                sesq[k] += 4 / 80 * (costs[k] - costs_rep[k]) ** 2

        for k, v in sesq.items():
            sesq[k] = v**0.5
        ci = {}
        for k, v in sesq.items():
            ci[k] = (costs[k] - 1.96*sesq[k], costs[k] + 1.96*sesq[k])

        # Save output
        out_costs = pd.DataFrame.from_dict(costs, orient='index')
        out_costs = out_costs.reset_index()
        out_costs.columns = ['type', 'cost']

        out_ci = pd.DataFrame.from_dict(ci, orient='index')
        out_ci = out_ci.reset_index()
        out_ci.columns = ['type', 'ci_lower', 'ci_upper']

        out = pd.merge(out_costs, out_ci, how='left', on='type')

        d_tix = {'own':1, 'matdis':2, 'bond':3, 'illchild':4, 'illspouse':5, 'illparent':6, 'total':7}
        out['tix'] = out['type'].apply(lambda x: d_tix[x])
        out = out.sort_values(by='tix')
        del out['tix']

        out.to_csv('%s/program_cost_%s_%s.csv' % (self.output_directory, self.st, self.out_id), index=False)

        message = 'Output saved. Total cost = $%s million 2012 dollars' % (round(costs['total']/1000000, 1))
        print(message)
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 100})
        self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
        return out  # df of leave type specific costs and total cost, along with ci's

    def create_chart(self, out):
        # Plot costs and ci
        total_cost = round(list(out.loc[out['type'] == 'total', 'cost'])[0] / 10 ** 6, 1)
        spread = round((list(out.loc[out['type'] == 'total', 'ci_upper'])[0] -
                        list(out.loc[out['type'] == 'total', 'ci_lower'])[0]) / 10 ** 6, 1)
        title = 'State: %s. Total Benefits Cost = $%s million (\u00B1%s).' % (self.st.upper(), total_cost, spread)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ind = np.arange(6)
        ys = out[:-1]['cost'] / 10 ** 6
        es = 0.5 * (out[:-1]['ci_upper'] - out[:-1]['ci_lower']) / 10 ** 6
        width = 0.5
        ax.bar(ind, ys, width, yerr=es, align='center', capsize=5, color='#1aff8c', ecolor='white')
        ax.set_ylabel('$ millions')
        ax.set_xticks(ind)
        ax.set_xticklabels(('Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent'))
        ax.yaxis.grid(False)
        format_chart(fig, ax, title)

        plt.savefig('%s/total_cost_%s_%s_%s' % (self.output_directory, self.yr, self.st, self.out_id),
                    facecolor='#333333', edgecolor='white')
        self.figure = fig
        return fig

    def run(self):
        self.save_program_parameters()
        self.prepare_data()
        self.get_acs_simulated()
        self.get_cost()

    def __put_queue(self, obj):
        if self.q is not None:
            self.q.put(obj)

    def get_progress(self):
        result = self.progress, self.updates
        self.updates = []
        return result

    def get_results(self):
        return pd.read_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id))

    def get_cost_df(self):
        return pd.read_csv('%s/program_cost_%s_%s.csv' % (self.output_directory, self.st, self.out_id))

    def get_population_analysis_results(self):
        # read in simulated acs, this is just df returned from get_acs_simulated()
        d = pd.read_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id))
        # total covered-by-program length
        d['cpl'] = [sum(x) for x in d[['cpl_%s' % t for t in self.types]].values]
        # keep needed vars for population analysis plots
        columns = ['PWGTP', 'cpl', 'female', 'age', 'wage12', 'nochildren', 'asian', 'black', 'white', 'native',
                   'other', 'hisp']
        d = d[columns]
        return d

## Other factors
# Leave prob factors, 6 types - TODO: code in wof in get_sim_col(), bound phat by max = 1

### test
#
# st = 'nj'
# yr = 16
# fp_fmla_in = './data/fmla_2012/fmla_2012_employee_restrict_puf.csv'
# fp_fmla_out = './data/fmla_2012/fmla_clean_2012.csv'
# fp_cps_in = './data/cps/CPS2014extract.csv'
# fp_cps_out = './data/cps/cps_for_acs_sim.csv'
# fp_length_distribution_out = './data/fmla_2012/length_distributions.json'
#
# fp_acsh_in = 'C:/workfiles/Microsimulation/git/large_data_files/'
# fp_acsp_in = 'C:/workfiles/Microsimulation/git/large_data_files/'
# fp_acs_out = './data/acs/'
#
# fps_in = [fp_fmla_in, fp_cps_in, fp_acsh_in, fp_acsp_in]
# fps_out = [fp_fmla_out, fp_cps_out, fp_acs_out, fp_length_distribution_out]
# clf_name = 'Logistic Regression'
#
# prog_para = [3440, 20, 1, 1, 0.67, 650]
# types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
# d_maxwk = dict(zip(types, 6*np.ones(6)))
# d_takeup = dict(zip(types, 1*np.ones(6)))
# prog_para.append(d_maxwk)
# prog_para.append(d_takeup)
# prog_para += [False, False] # empgov, empself
#
# se = SimulationEngine(st, yr, fps_in, fps_out, clf_name, prog_para)
# se.save_program_parameters()
# se.prepare_data()
# se.get_acs_simulated()
# se.get_cost()