'''
main simulation engine
chris zhang 12/4/2019
'''
# CHANGES 11/14/2019
# Made dual receiver as proportionate parameter of those individuals with >0 prop_pay.
# Removed imputation of doctor/hospital variables
# Removed multiple leave types imputation
# Temporary - removed link between cp-len and generosity. Now set cp-len = mn-len for all types,
# and dual receiver status becomes irrelevant - may add this link back later
# Modified take up rate to randomly draw rows until target take up population is created
# Used common 'denom' ACS data for takeup flags - eligible workers after excl. per wage12/wkweek/wksworked/empsize
# Used take up rates = official case load data / eligible worker pop. This is indep from sim,
# skipping the hard-to-estimate taker/needer pops
# Finding - logit sim fewest takers/needers, ridge most, rest methods in between.
# but these do not cause huge diff in cost est, given the small empirical takeup ratios
# logit does not sim enough takers/needers to reach required takeup in RI. Other methods okay
# Applied logic control - age range of taker/needer of matdis, bond (set max to 50)

# TODO: validate MLs within FMLA, check diff MLs performance on predicting number of weeks (pmts) for RI/NJ/CA data
# TODO: adopt needers_fully_part - really necessary?? Not a empirical possibility
# TODO: make a note in doc about 1.02 POW factor if user tries to create pop est / needers_full_part override

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
from Utils import format_chart, check_dependency


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
        self.fp_length_distribution_out = fps_out[3] # fp to length distributions in days estimated from restricted FMLA
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
        self.clone_factor = prog_para[15] # integer multiplier to scale up sample (while shrink weight) for granular sim
        self.dual_receivers_share = prog_para[16] # share of company/state benefit receivers among anypay = 1
        self.random_seed = prog_para[17]
        print('Random seed:', self.random_seed)
        self.random_state = np.random.RandomState(self.random_seed)

        # leave types
        self.types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']

        # a dict from clf_name to clf
        self.d_clf = {}
        self.d_clf['Logistic Regression'] = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto', random_state=self.random_state)
        self.d_clf['Ridge Classifier'] = sklearn.linear_model.RidgeClassifier(random_state=self.random_state)
        #self.d_clf['Stochastic Gradient Descent'] = sklearn.linear_model.SGDClassifier(loss='modified_huber', max_iter=1000, tol=0.001)
        self.d_clf['Naive Bayes'] = sklearn.naive_bayes.MultinomialNB()
        self.d_clf['Support Vector Machine'] = sklearn.svm.SVC(probability=True, gamma='auto', random_state=self.random_state)
        self.d_clf['Random Forest'] = sklearn.ensemble.RandomForestClassifier(random_state=self.random_state)
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
        self.pow_pop_multiplier = 1.0217029934467345 # based on 2012-2016 ACS, see project acs_all

    def save_program_parameters(self):
        # create output folder
        os.makedirs(self.output_directory)

        # save meta file of program parameters
        para_labels = ['State', 'Year', 'Place of Work',
                       'Minimum Annual Wage','Minimum Annual Work Weeks','Minimum Annual Work Hours',
                       'Minimum Employer Size','Proposed Wage Replacement Ratio','Weekly Benefit Cap',
                       'Include Goverment Employees, Federal',
                       'Include Goverment Employees, State',
                       'Include Goverment Employees, Local',
                       'Include Self-employed',
                       'Simulation Method',
                       'Share of Dual Receivers',
                       'Clone Factor',
                       'Random Seed']
        para_labels_m = ['Maximum Week of Benefit Receiving',
                         'Take Up Rates'] # type-specific parameters

        para_values = [self.st.upper(),self.yr + 2000, self.state_of_work,
                       self.elig_wage12,self.elig_wkswork,self.elig_yrhours,self.elig_empsize,self.rrp,self.wkbene_cap,
                       self.incl_empgov_fed, self.incl_empgov_st,self.incl_empgov_loc,self.incl_empself,
                       self.clf_name,
                       self.dual_receivers_share,
                       self.clone_factor, self.random_seed]
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
        dcf = DataCleanerFMLA(self.fp_fmla_in, self.fp_fmla_out, self.random_state)
        dcf.clean_data()
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 10})
        self.__put_queue({'type': 'message', 'engine': self.engine_type,
                          'value': 'File saved: clean FMLA data file before CPS imputation.'})
        message = dcf.impute_fmla_cps(self.fp_cps_in, self.fp_cps_out)
        self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 20})
        self.__put_queue({'type': 'message', 'engine': self.engine_type,
                          'value': 'File saved: clean FMLA data file after CPS imputation.'})
        # dcf.get_length_distribution(self.fp_length_distribution_out)
        # self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 25})
        # self.__put_queue({'type': 'message', 'engine': self.engine_type,
        #                   'value': 'File saved: leave distribution estimated from FMLA data.'})

        self.__put_queue({'type': 'message', 'engine': self.engine_type,
                          'value': 'Cleaning ACS data. State chosen = RI. Chunk size = 100000 ACS rows'})
        dca = DataCleanerACS(self.st, self.yr, self.fp_acsh_in, self.fp_acsp_in, self.fp_acs_out, self.state_of_work, self.random_state)
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

        # Sample restriction - reduce to eligible workers (all elig criteria indep from simulation below)

        # drop government workers if desired
        if not self.incl_empgov_fed:
            acs = acs.drop(acs[acs['empgov_fed'] == 1].index)
        if not self.incl_empgov_st:
            acs = acs.drop(acs[acs['empgov_st'] == 1].index)
        if not self.incl_empgov_loc:
            acs = acs.drop(acs[acs['empgov_loc'] == 1].index)
        if not self.incl_empself:
            acs = acs.drop(acs[(acs['COW'] == 6) | (acs['COW'] == 7)].index)
        # check other program eligibility
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
        # drop ineligible workers (based on wage/work/empsize)
        acs = acs.drop(acs[acs['elig_prog'] != 1].index)

        # Expand ACS if clone factor > 1
        # shrink all weights by factor
        if self.clone_factor>1:
            for wt in ['PWGTP'] + ['PWGTP' + str(x) for x in range(1, 81)]:
                acs[wt] = acs[wt]/self.clone_factor
            # then expand acs by factor
            acs = pd.concat([acs]*self.clone_factor)

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
            acs[c] = get_sim_col(X, y, w, Xa, clf, self.random_state)
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
        acs['taker'] = [max(z) for z in acs[['take_%s' % t for t in self.types]].values]
        acs['needer'] = [max(z) for z in acs[['need_%s' % t for t in self.types]].values]
        X = d[(d['taker'] == 1) | (d['needer'] == 1)][col_Xs]
        w = d.loc[X.index][col_w]
        Xa = acs[(acs['taker'] == 1) | (acs['needer'] == 1)][X.columns]
        if len(Xa) == 0:
            print('Warning: Neither leave taker nor leave needer present in simulated ACS persons. '
                  'Simulation gives degenerate scenario of zero leaves for all workers.')
        else:
            for c in ['anypay']:
                y = d.loc[X.index][c]
                simcol_indexed = get_sim_col(X, y, w, Xa, clf, self.random_state)
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
            yhat = get_sim_col(X, y, w, Xa, clf, self.random_state)
            # prop_pay labels are from 1 to 6, get_sim_col() vectorization sum gives 0~5, increase label by 1
            yhat = pd.Series(data=[x + 1 for x in yhat], index=Xa.index, name='prop_pay')
            acs = acs.join(yhat)
            acs.loc[acs['prop_pay'].notna(), 'prop_pay'] = [d_prop[x] for x in
                                                            acs.loc[acs['prop_pay'].notna(), 'prop_pay']]

        # Sample restriction - reduce to simulated takers/needers, append dropped rows later before saving post-sim acs
        acs_neither_taker_needer = acs[(acs['taker'] == 0) & (acs['needer'] == 0)]
        acs = acs.drop(acs_neither_taker_needer.index)

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
                lens.append(flen[pfl][t][bisect.bisect(cs, self.random_state.random_sample())][0])
            acs.loc[acs['take_%s' % t] == 1, 'len_%s' % t] = np.array(lens)
            # print('mean = %s' % acs['len_%s' % t].mean())
        # print('te: sq length sim = %s' % (time()-t0))

        # Max needed lengths (mnl) - draw from simulated without-program length distribution
        # conditional on max length >= without-program length
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
                    vxs = self.random_state.choice(dct_vw[x]['len_%s' % t], mx, p=dct_vw[x]['PWGTP'] / dct_vw[x]['PWGTP'].sum())
                    acs.loc[(acs['resp_len'] == 1) & (acs['len_%s' % t] == x), 'mnl_%s' % t] = vxs
                else:
                    acs.loc[(acs['resp_len'] == 1) & (acs['len_%s' % t] == x), 'mnl_%s' % t] = int(x * 1.25)
                    # print('mean = %s. MNL sim done for type %s. telapse = %s' % (acs['mnl_%s' % t].mean(), t, (time()-t0)))

        # logic control of mnl
        acs.loc[acs['male'] == 1, 'mnl_matdis'] = 0
        acs.loc[(acs['nevermarried'] == 1) | (acs['divorced'] == 1), 'mnl_illspouse'] = 0
        acs.loc[acs['nochildren'] == 1, 'mnl_bond'] = 0
        acs.loc[acs['nochildren'] == 1, 'mnl_matdis'] = 0

        # check if sum of mnl hits max = 52*5 = 260. If so, use max=260 to distribute prop to mnl of 6 types
        acs['mnl_all'] = [x.sum() for x in acs[['mnl_%s' % x for x in self.types]].values]
        for t in self.types:
            acs.loc[acs['mnl_all']>260, 'mnl_%s' % t] = [int(x) for x in acs.loc[acs['mnl_all']>260, 'mnl_%s' % t] / \
                                                     acs.loc[acs['mnl_all']>260, 'mnl_all'] * 260]
        # the mnl-capped workers would must have sq-len no larger than mn-len
        for t in self.types:
            acs.loc[acs['mnl_all']>260, 'len_%s' % t] = acs.loc[acs['mnl_all']>260, 'mnl_%s' % t]

        ## If do following, then ignores link between generosity (rrp) and cf-len, cp-len
        # # set covered-by-program leave lengths (cp-len) as maximum needed leave lengths (mn-len) for each type
        # for t in self.types:
        #     acs['cpl_%s' % t] = acs['mnl_%s' % t]

        # Given fraction of dual receiver x among anypay=1, simulate dual/single receiver status among anypay=1
        acs['dual_receiver'] = 0
        ws = acs[acs['anypay']==1]['PWGTP'] # weights of anypay=1
        acs.loc[acs['anypay']==1, 'dual_receiver'] = get_weighted_draws(ws, self.dual_receivers_share, self.random_state)
        # check if target pop is achieved among anypay=1
        s_dual_receiver = acs[(acs['anypay']==1) & (acs['dual_receiver'] == 1)]['PWGTP'].sum() / acs[acs['anypay']==1]['PWGTP'].sum()
        s_dual_receiver = round(s_dual_receiver, 2)
        print('Specified share of dual-receiver = %s. Post-sim weighted share = %s' % (self.dual_receivers_share, s_dual_receiver))

        # Simulate counterfactual leave lengths (cf-len) for dual receivers
        # Given cf-len, get cp-len
        # With program, effective rr =min(rre+rrp, 1), assuming responsiveness diminishes if full replacement attainable
        for t in self.types:
            acs['cfl_%s' % t] = np.nan
            ## Get cf-len for dual receivers among anypay=1
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
            ## Get covered-by-program leave lengths (cp-len) for dual receivers among anypay=1
            # allocate cf-len between employer and state according to rre/rrp ratio
            acs.loc[acs['dual_receiver'] == 1, 'cpl_%s' % t] = acs.loc[acs['dual_receiver'] == 1, 'cfl_%s' % t] * \
                                                                 self.rrp / (
                self.rrp + acs.loc[acs['dual_receiver'] == 1, 'prop_pay'])


        # Simulate cf-len for single receivers among anypay=1
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

        # Simulate cf-len for anypay=0 workers
        # Given cf-len, get cp-len
        for t in self.types:
            # anypay=0 workers extends leave by (mnl-sql)*rrp
            acs.loc[(acs['anypay'] == 0), 'cfl_%s' % t] = \
                acs.loc[(acs['anypay'] == 0), 'len_%s' % t] + \
                (acs.loc[(acs['anypay'] == 0), 'mnl_%s' % t] - acs.loc[(acs['anypay'] == 0), 'len_%s' % t]) * self.rrp
            # anypay=0 workers' cp-len is just cf-len, as they don't have company benefits
            acs.loc[(acs['anypay']==0), 'cpl_%s' % t] = acs.loc[(acs['anypay']==0), 'cfl_%s' % t]

        # Apply cap of coverage period (in weeks) to cpl_type (in days) for each leave type
        for t in self.types:
            acs.loc[acs['cpl_%s' % t] >= 0, 'cpl_%s' % t] = [min(x, 5 * self.d_maxwk[t]) for x in
                                                             acs.loc[acs['cpl_%s' % t] >= 0, 'cpl_%s' % t]]
        # get acs with takeup flags for each leave type
        acs = self.get_acs_with_takeup_flags(acs, acs_neither_taker_needer, 'PWGTP')

        # get benefit received for each worker
        acs_taker_needer = acs[(acs['taker']==1) | (acs['needer']==1)]
        acs_neither_taker_needer = acs.drop(acs_taker_needer.index)
        # apply take up flag and weekly benefit cap, and annual benefit for each worker, 6 types
        for t in self.types:
            # v = capped weekly benefit of leave type
            v = [min(x, self.wkbene_cap) for x in
                 ((acs_taker_needer['cpl_%s' % t] / 5) * (acs_taker_needer['wage12'] / acs_taker_needer['wkswork'] * self.rrp))]
            # get annual benefit for leave type t - sumprod of capped benefit, and takeup flag for each ACS row
            acs_taker_needer['annual_benefit_%s' % t] = (v * acs_taker_needer['cpl_%s' % t] / 5 * acs_taker_needer['takeup_%s' % t])
        # append acs_neither_taker_needer
        acs = acs_taker_needer.append(acs_neither_taker_needer, sort=True)

        # Save ACS data after finishing simulation
        acs.to_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id), index=False)
        message = 'Leaves simulated for 5-year ACS 20%s-20%s in state %s. Time needed = %s seconds. ' \
                  'Total worker pop in post-sim ACS = %s' % \
                  ((self.yr-4), self.yr, self.st.upper(), round(time()-tsim, 0), acs['PWGTP'].sum())
        print(message)
        self.__put_queue({'type': 'progress', 'engine': self.engine_type, 'value': 95})
        self.__put_queue({'type': 'message', 'engine': self.engine_type, 'value': message})
        return acs

    def get_acs_with_takeup_flags(self, acs_taker_needer, acs_neither_taker_needer, col_w):
        # get 0/1 takeup flag using post-sim acs with only takers/needers
        # col_w = weight column, PWGTP for main, or PWGTPx for x-th rep weight in ACS data

        # We first append acs_neither_taker_needer back to post-sim acs, so we'll work with a common population
        acs = acs_taker_needer.append(acs_neither_taker_needer, sort=True) # for new cols not in acs_neither_taker_needer, will create nan
        # drop takeup flag cols if any
        for c in ['takeup_%s' % x for x in self.types]:
            if c in acs.columns:
                del acs[c]

        # Then perform a weighted random draw using user-specified take up rate until target pop is reached
        for t in self.types:
            # cap user-specified take up for type t by max possible takeup = s_positive_cpl, in pop per sim results
            s_positive_cpl = acs[acs['cpl_%s' % t] > 0][col_w].sum() / acs[col_w].sum()
            # display warning for unable to reach target pop from simulated positive cpl_type pop
            if col_w=='PWGTP':
                if self.d_takeup[t] > s_positive_cpl:
                    print('Warning: User-specified take up for type -%s- is capped '
                          'by maximum possible take up rate (share of positive covered-by-program length) '
                          'based on simulation results, at %s.' % (t, s_positive_cpl))
            takeup = min(s_positive_cpl, self.d_takeup[t])
            p_draw = takeup / s_positive_cpl  # need to draw w/ prob=p_draw from cpl>0 subpop, to get desired takeup
            #print('p_draw for type -%s- = %s' % (t, p_draw))
            # get take up indicator for type t - weighted random draw from cpl_type>0 until target is reached
            acs['takeup_%s' % t] = 0
            draws = get_weighted_draws(acs[acs['cpl_%s' % t] > 0][col_w], p_draw, self.random_state)
            #print('draws = %s' % draws)
            acs.loc[acs['cpl_%s' % t] > 0, 'takeup_%s' % t] \
                = draws

            # for main weight, check if target pop is achieved among eligible ACS persons
            if col_w=='PWGTP':
                s_takeup = acs[acs['takeup_%s' % t] == 1][col_w].sum() / acs[col_w].sum()
                s_takeup = round(s_takeup, 4)
                print('Specified takeup for type %s = %s. '
                      'Effective takeup = %s. '
                      'Post-sim weighted share = %s' % (t, self.d_takeup[t], takeup, s_takeup))
            # return ACS with all eligible workers (regardless of taker/needer status), with takeup_type flags sim'ed
        return acs

    def get_cost(self):
        # read simulated ACS, and reduce to takers/needers
        acs = pd.read_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id))
        acs_taker_needer = acs[(acs['taker']==1) | (acs['needer']==1)]
        acs_neither_taker_needer = acs.drop(acs_taker_needer.index)

        # apply take up flag and weekly benefit cap, and compute total cost, 6 types
        costs = {}
        for t in self.types:
            # v = capped weekly benefit of leave type
            v = [min(x, self.wkbene_cap) for x in
                 ((acs_taker_needer['cpl_%s' % t] / 5) * (acs_taker_needer['wage12'] / acs_taker_needer['wkswork'] * self.rrp))]
            # inflate weight for missing POW
            w = acs_taker_needer['PWGTP'] * self.pow_pop_multiplier

            # get program cost for leave type t - sumprod of capped benefit, weight, and takeup flag for each ACS row
            costs[t] = (v * acs_taker_needer['cpl_%s' % t] / 5 * w * acs_taker_needer['takeup_%s' % t]).sum()
        costs['total'] = sum(list(costs.values()))

        # compute standard error using replication weights, then compute confidence interval
        sesq = dict(zip(costs.keys(), [0]*len(costs.keys())))
        for wt in ['PWGTP%s' % x for x in range(1, 81)]:
            # initialize costs dict for current rep weight
            costs_rep = {}
            # get takeup_type flags for acs under current rep weight
            acs = self.get_acs_with_takeup_flags(acs_taker_needer, acs_neither_taker_needer, wt)
            acs_taker_needer = acs[(acs['taker'] == 1) | (acs['needer'] == 1)]

            for t in self.types:
                v = [min(x, self.wkbene_cap) for x in
                     ((acs_taker_needer['cpl_%s' % t] / 5) * (acs_taker_needer['wage12'] / acs_taker_needer['wkswork'] * self.rrp))]
                # inflate weight for missing POW
                w = acs_taker_needer[wt] * self.pow_pop_multiplier

                # get program cost for leave type t - sumprod of capped benefit, weight, and takeup flag for each ACS row
                costs_rep[t] = (v * acs_taker_needer['cpl_%s' % t] / 5 * w * acs_taker_needer['takeup_%s' % t]).sum()
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
        t0 = time()
        try:
            self.check_dependencies()
            self.save_program_parameters()
            self.prepare_data()
            self.get_acs_simulated()
            self.get_cost()
        except Exception as e:
            self.__put_queue({'type': 'error', 'engine': self.engine_type, 'value': e})
            raise
        print('Total runtime = %s seconds.' % (round(time()-t0, 0)))

    def check_dependencies(self):
        dependency_versions = {
            'matplotlib': '2.2.3',
            'mord': '0.5',
            'pandas': '0.23.0',
            'sklearn': '0.20.1'
        }

        for dependency, version in dependency_versions.items():
            if not check_dependency(dependency, version):
                self.__put_queue({'type': 'warning', 'engine': self.engine_type,
                                  'value': (dependency, version)})

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
        # restrict to taker/needer only (workers with neither status have cpl_type = nan)
        # d = d[(d['taker']==1) | (d['needer']==1)]

        # restrict to workers who take up the program
        d['takeup_any'] = [int(x.sum()>0) for x in d[['takeup_%s' % x for x in self.types]].values]
        d = d[d['takeup_any']==1]

        # make sure cpl_type is non-missing
        for t in self.types:
            d.loc[d['cpl_%s' % t].isna(),  'cpl_%s' % t] = 0

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