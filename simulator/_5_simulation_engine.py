'''
Simulate leave taking behavior of ACS sample, using FMLA data

to do:
1. Need to think more about imputation to fill in missing values before kNN
2. Need to impute coveligd, more from ACM model
3. Need to compute counterfactual 'length' of leave for FMLA samples under new program

Chris Zhang 9/13/2018
'''

# -------------
# Housekeeping
# -------------

import pandas as pd
import numpy as np
import json
import collections
import os.path
from ._5a_aux_functions import simulate_wof, get_marginal_probs, get_dps_lowerBounded
from time import time
from datetime import datetime
import sklearn.linear_model, sklearn.naive_bayes, sklearn.neighbors, sklearn.tree, sklearn.ensemble, \
    sklearn.gaussian_process, sklearn.svm

class SimulationEngine():


    def __init__(self, st, fp_acs, fp_fmla, fp_out, clf_name, rr1, hrs, d_max_wk, max_wk_replace, empgov=False, empself=False):
        '''

        :param st: state name, 'ca', 'ma', etc.
        :param fp_acs: file path to cleaned up state-level ACS data
        :param fp_fmla: file path to cleaned up FMLA data
        :param fp_out: file path to save output
        :param rr1: replacement rate under new program
        :param hrs: min annual work hours for eligibility
        :param d_max_wk: a dict from leave type to max number of weeks to receive benefits
        :param max_wk_replace: max weekly benefits
        :param empgov: eligibility of government workers, default is False
        :param empself: eligibility of self-employed workers, default is False

        Returns: None
        '''


        self.st = st
        self.fp_acs = fp_acs
        self.fp_fmla = fp_fmla
        self.fp_out = fp_out
        self.clf_name = clf_name
        self.rr1 = rr1
        self.hrs = hrs
        self.d_max_wk = d_max_wk
        self.max_wk_replace = max_wk_replace
        self.empgov = empgov
        self.empself = empself
        
        # a dict from varname style leave types to user-readable leave types
        self.d_type = {}
        self.d_type['own'] = 'Own Health'
        self.d_type['matdis'] = 'Maternity'
        self.d_type['bond'] = 'New Child'
        self.d_type['illchild'] = 'Ill Child'
        self.d_type['illspouse'] = 'Ill Spouse'
        self.d_type['illparent'] = 'Ill Parent'

        # a dict from clf_name to clf
        self.d_clf = {}
        self.d_clf['Logistic Regression'] = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto')
        self.d_clf['Ridge Classifier'] = sklearn.linear_model.RidgeClassifier()
        self.d_clf['Stochastic Gradient Descent'] = sklearn.linear_model.SGDClassifier(loss='modified_huber', max_iter=1000, tol=0.001)
        self.d_clf['Naive Bayes'] = sklearn.naive_bayes.MultinomialNB()
        self.d_clf['Support Vector Machine'] = sklearn.svm.SVC(probability=True)
        self.d_clf['Random Forest'] = sklearn.ensemble.RandomForestClassifier()
        # self.d_clf['K Nearest Neighbor'] = sklearn.neighbors.KNeighborsClassifier()

        # get clf
        if self.clf_name not in ['K Nearest Neighbor']:
            self.clf = self.d_clf[self.clf_name]

        # calibration factor alpha
        self.alpha = 1
        if self.clf_name == 'Logistic Regression':
            self.alpha = 8.671
        elif self.clf_name == 'Ridge Classifier':
            self.alpha = 1.529
        elif self.clf_name == 'Naive Bayes':
            self.alpha = 5.837
        elif self.clf_name == 'Random Forest':
            self.alpha = 3.686
        elif self.clf_name == 'K Nearest Neighbor':
            self.alpha = 3.293
        elif self.clf_name == 'Support Vector Machine':
            self.alpha = 2.485

    def get_simulator_params(self):
        Xs = ['age', 'agesq', 'male', 'noHSdegree',
                  'BAplus', 'empgov_fed', 'empgov_st', 'empgov_loc',
                  'lnfaminc', 'black', 'asian', 'hisp', 'other',
                  'ndep_kid', 'ndep_old', 'nevermarried', 'partner',
                  'widowed', 'divorced', 'separated']
        ys = ['taker', 'needer']
        ys += ['resp_len']
        w = ['weight']
        k = 2


        return (Xs, ys, w, k)

    def get_aux_data_fp(self):
        '''
        get some auxiliary data filepaths such as marginal probs of numleaves estimated from FMLA
        :return:
        '''

        fp_mp_numleaves = './data/fmla_2012/marginal_probs_numleaves.csv'
        fp_cp_leavetypes = './data/fmla_2012/conditional_probs_leavetypes.csv'
        fp_pdf_lengths = './data/fmla_2012/length_distributions.json'
        return(fp_mp_numleaves, fp_cp_leavetypes, fp_pdf_lengths)

    def fit_model(self, xtr, ytr, ws):
        '''
        fit model using a simulation method, with some configurations tailored to the method
        :param xtr: training data, x
        :param ytr: training data, y
        :param ws: weights

        '''
        if self.clf_name in ['Logistic Regression',
                             'Random Forest',
                             'Support Vector Machine',
                             'Ridge Classifier',
                             'Stochastic Gradient Descent',
                             'Naive Bayes']:
            self.clf.fit(xtr, ytr, sample_weight = ws)
        if self.clf_name in ['K Nearest Neighbor']:
            ws = np.array([ws]) # make sure weights[:, i] can be called in package code classification.py
            f = lambda x: ws
            self.clf = sklearn.neighbors.KNeighborsClassifier(weights=f)
            self.clf.fit(xtr, ytr)
        return None

    def get_pred_probs(self, xts):
        '''
        get predicted probabilities of all classes for the post-fit self.clf
        :param xts: testing/prediction dataset
        :return: array of list of probs
        '''

        if self.clf_name in ['Logistic Regression',
                             'Random Forest',
                             'Stochastic Gradient Descent',
                             'Support Vector Machine',
                             'Naive Bayes',
                             'K Nearest Neighbor']:
            phat = self.clf.predict_proba(xts)
        if self.clf_name in ['Ridge Classifier']:
            d = self.clf.decision_function(xts)  # distance to hyperplane
            # case of binary classification, d is np.array([...])
            if d.ndim==1:
                phat = np.exp(d) / (1 + np.exp(d))  # list of pr(yhat = 1)
                phat = np.array([[(1 - x), x] for x in phat])
            # case of multiclass problem (n class >=3), di is np.array([[...]])
            elif d.ndim==2:
                phat = np.exp(d) / np.array([[x] * len(d[0]) for x in np.exp(d).sum(axis=1)])
        return phat

    def get_acs_simulated(self):
        t0 = time()

        # -------------
        # Load Data
        # -------------

        # Read in the ACS data for the specified state
        acs = pd.read_csv(self.fp_acs)

        # Read in cleaned-up FMLA data
        fmla = pd.read_csv(self.fp_fmla)
        fmla.index.name = 'empid'

        # Read auxiliary data
        fp_mp_numleaves, fp_cp_leavetypes, fp_pdf_lengths = self.get_aux_data_fp()
        dmp = pd.read_csv(fp_mp_numleaves, index_col=0)
        dcp = pd.read_csv(fp_cp_leavetypes, index_col=0)
        with open(fp_pdf_lengths) as f:
            dlen = json.load(f)
        dlen = {k : {int(k): v for k, v in dlen_t.items()} for k, dlen_t in dlen.items()}
        dlen = {k: collections.OrderedDict(sorted(dlen_t.items(), key=lambda x: x[0])) for k, dlen_t in dlen.items()}

        # -------------
        # Restrict ACS sample
        # -------------

        # Restrict based on employment status and age
        acs = acs[(acs.age>=18)]
        acs = acs.drop(index=acs[acs.ESR==4].index) # armed forces at work
        acs = acs.drop(index=acs[acs.ESR==5].index) # armed forces with job but not at work
        acs = acs.drop(index=acs[acs.ESR==6].index) # NILF
        acs = acs.drop(index=acs[acs.COW==8].index) # working without pay in family biz
        acs = acs.drop(index=acs[acs.COW==9].index) # unemployed for 5+ years, or never worked
        acs.index.name = 'acswid'

        # -------------
        # Read in Parameters
        # -------------

        # Algorithm parameter
        Xs, ys, w, k = self.get_simulator_params()

        # -------------
        # Train FMLA data, use ACS to predict Taker/Needer statuses, and financially-constrained status (resp_len)
        # -------------

        # Randomly sample a portion of FMLA data to train model
        ixs_tr = np.random.choice(fmla.index, int(0.8*len(fmla)), replace=False)
        fmla_xtr = fmla.loc[ixs_tr][Xs]
        fmla_ytr = fmla.loc[ixs_tr][ys]
        wght = fmla.loc[ixs_tr][w]
        wght = [x[0] for x in np.array(wght)]

        # ACS columns for simulation
        acs_x = acs[Xs]

        # Fill in missing values
        fmla_xtr = fmla_xtr.fillna(fmla_xtr.mean())
        fmla_ytr = fmla_ytr.fillna(0) # fill with label 0 for now
        acs_x = acs_x.fillna(acs_x.mean())
        print('Input data read. Now simulating...')
        print('Simulation method: %s\n' % self.clf_name)

        # Fit the training data and predict taker/needer/resp_len for ACS

        for c in fmla_ytr.columns:
            self.fit_model(fmla_xtr, fmla_ytr[c],wght)
            phat = self.get_pred_probs(acs_x)
            phat = pd.DataFrame(phat).set_index(acs.index)
            phat.columns = ['p_%s' % int(x) for x in self.clf.classes_]
            # apply wheel of fortune
            phat[c] = phat[phat.columns].apply(lambda x: simulate_wof(x), axis=1)
            acs = acs.join(phat[c])

        # make sure resp_len = 1 only for takers or needers
        acs.loc[(acs['taker']==0) & (acs['needer']==0), 'resp_len'] = 0

        # -------------
        # Simulate most recent leave type for takers/needers/dual
        # -------------
        # If A20 = . or 1, then loop 1 (longest leave) is the only leave, and most recent one
        # If A20 = 2, then loop 1 (longest leave) is not the most recent, but loop 2 is
        # Relevant var in FMLA is 'reason_take'
        # retrict sample to taker/needer/dual with valid reason
        ytr = fmla.loc[ixs_tr]
        ytr = ytr[(ytr['LEAVE_CAT']!=3) & (ytr['type_recent'].notna())]['type_recent']
        xtr = fmla_xtr.loc[ytr.index] # use indices of ytr to pick xtr rows
        wght = fmla.loc[ytr.index][w]
        wght = [x[0] for x in np.array(wght)]
        self.fit_model(xtr, ytr, wght)
            # identify acs ids with predicted taker/needer status
            # then predict on these acs ppl, fill other rows as na
        acswids_fit = acs[(acs['taker']==1) | (acs['needer']==1)].index

        # TODO: check SGD prediction of taker/needer after intermediate imputation of work weeks etc.
        # check if taker/needer = 1 would give any subsample under SGD, if empty subsample then can't continue
        if self.clf_name in ['Stochastic Gradient Descent']:
            print(acs['taker'].value_counts())
            print(acs['needer'].value_counts())

        phat = self.get_pred_probs(acs_x.loc[acswids_fit])
        phat = pd.DataFrame(phat).set_index(acswids_fit)
        phat.columns = ['p_%s' % x for x in self.clf.classes_]
            # apply wheel of fortune
            # TODO: force matdis to females only
        phat['type_ix'] = phat[phat.columns].apply(lambda x: simulate_wof(x), axis=1)
        phat['type_recent'] = phat['type_ix'].apply(lambda x: self.clf.classes_[int(x)])
        phat = phat['type_recent']
        acs = acs.join(phat)

        # -------------
        # Simulate status of multiple leave
        # -------------
        ytr = fmla.loc[ixs_tr]
        ytr = ytr[ytr['LEAVE_CAT']!=3]
        ytr['multiple'] = 0
        ytr['multiple'] = np.where(ytr['A20']==2, 1, ytr['multiple'])
        ytr = ytr['multiple']
        xtr = fmla_xtr.loc[ytr.index]
        wght = fmla.loc[ytr.index][w]
        wght = [x[0] for x in np.array(wght)]
        self.fit_model(xtr, ytr, wght)
        acswids_fit = acs[(acs['taker']==1) | (acs['needer']==1)].index
        phat = self.get_pred_probs(acs_x.loc[acswids_fit])
        phat = pd.DataFrame(phat).set_index(acswids_fit)
        phat.columns = ['p_%s' % x for x in self.clf.classes_]
            # apply wheel of fortune
        phat['multiple'] = phat[phat.columns].apply(lambda x: simulate_wof(x), axis=1)
        acs = acs.join(phat['multiple'])

        # -------------
        # Simulate number of leaves for workers with multiple leaves
        # -------------
        dmp['ps'] = dmp[dmp.columns].apply(lambda x: (x[0], x[1], x[2], x[3], x[4]), axis=1)
        dmp = dmp['ps']
        dmp = dmp.to_dict()
        acs_nl = acs[['taker','needer','type_recent','multiple']][(acs['type_recent'].notna()) & (acs['multiple']==1)]

        acs_nl['status'] = 'taker'
        acs_nl['status'] = np.where((acs_nl['taker']==0) & (acs_nl['needer']==1), 'needer', acs_nl['status'])
        acs_nl['status'] = acs_nl['status'] + '_' + acs_nl['type_recent']
        acs_nl['ps'] = acs_nl['status'].apply(lambda x: dmp[x])
            # apply wheel of fortune, make sure add 2 so that ix 0 points to min numleaves = 2
        acs_nl['num_leaves'] = acs_nl['ps'].apply(lambda x: 2 + simulate_wof(x))
        acs_nl = acs_nl.drop(columns=['ps'])

        # -------------
        # Simulate leave types for multiple leavers
        # -------------
        # Get index of type_recent
        otypes = list(dcp.columns) # ordered types, alphabetically
        acs_nl['type_ix_1'] = acs_nl['type_recent'].apply(lambda x: otypes.index(x))

        # Loop through numleaves = 2 to 5
        for nl in range(2, 6):
            acs_nl['ps_nxt'] = np.nan
            acs_nl.loc[acs_nl['num_leaves'] >= nl, 'ps_nxt'] = acs_nl.loc[acs_nl['num_leaves']>=nl, 'type_recent'].apply(lambda x: list(dcp.loc[x]))
            acs_nl.loc[acs_nl['num_leaves'] >= nl, 'ps_nxt'] = acs_nl.loc[acs_nl['num_leaves']>=nl,['ps_nxt'] + ['type_ix_%s' % xx for xx in range(1, nl)]].apply(lambda x: get_marginal_probs(x[0], [int(xx) for xx in x[1:]]), axis=1)
            acs_nl['type_ix_%s' % nl] = np.nan
            acs_nl.loc[acs_nl['num_leaves'] >= nl, 'type_ix_%s' % nl] = acs_nl.loc[acs_nl['num_leaves']>=nl, 'ps_nxt'].apply(lambda x: simulate_wof(x))
            acs_nl['type_%s' % nl] = np.nan
            acs_nl.loc[acs_nl['num_leaves'] >= nl,'type_%s'  % nl] = acs_nl.loc[acs_nl['num_leaves'] >= nl,'type_ix_%s' % nl].apply(lambda x: otypes[int(x)])
            acs_nl = acs_nl.drop(columns=['ps_nxt'])
        # For num_leaves == 6, leave type is the only one left
        acs_nl['type_6'] = np.nan
        acs_nl.loc[acs_nl['num_leaves'] == 6,'type_6'] = acs_nl.loc[acs_nl['num_leaves'] == 6, ['type_recent'] + ['type_%s' % xx for xx in range(2, 6)]].apply(lambda x: list(set(otypes) - set(x))[0], axis=1)

        # Merge finished multi-leaver simulation results to working acs
        acs_nl = acs_nl[['type_%s' % x for x in range(2, 7)]]
        acs = acs.join(acs_nl)
        acs = acs.rename(columns={'type_recent': 'type_1'})
        # -------------
        # Simulate status-quo leave lengths
        # for each type where leave occurs, draw from FMLA distribution of status-quo lengths
        # from FMLA data cleaning, var 'length' is most recent leave length
        # we use this length var to build distribution of length for each type
        # -------------
        for tix in range(1, 7):
            acs['length_%s' % tix] = np.nan
            acs.loc[acs['type_%s' % tix].notna(), 'length_%s' % tix] = \
            acs.loc[acs['type_%s' % tix].notna(), 'type_%s' % tix].apply(lambda x: list(dlen[x].keys())[simulate_wof(list(dlen[x].values()))])

        # -------------
        # Simulate counterfactual leave lengths
        # (length that would satisfy all needs, no more corners, i.e. under most generous program)
        # If resp_len = 0, counterfactual = status-quo
        # If resp_len = 1 (thus taker/needer), counterfactual = draw of length of type t from resp_len = 0 acs pool
        # subject to all lengths in pool > current length
        # -------------

        # get distribution of lengths from resp_len = 0 acs pool, by type,
        # use most recent leave for all multiple leaves (most accurate for pdf est)
        dleni = {}
        for t in otypes:
            dkvs = acs[(acs['resp_len'] == 0) & (acs['length_1'] > 0) & (acs['type_1'] == t)].groupby('length_1').agg({'PWGTP':'sum'})
            ks = list(dkvs.sort_index().index)
            vs = list(dkvs.sort_index()['PWGTP'])
            vs = np.array(vs)
            vs = vs/vs.sum()
            dleni_t = dict(zip(ks, vs))
            dleni[t] = dleni_t

        dleni = {k: collections.OrderedDict(sorted(dleni_t.items(), key=lambda x: x[0])) for k, dleni_t in dleni.items()}

        # simulating resp_length_1 through 6
        for tix in range(1, 7):
            acs['resp_length_%s' % tix] = np.nan
            acs.loc[(acs['resp_len']==0) & (acs['length_%s' % tix].notna()), 'resp_length_%s' % tix] = acs.loc[(acs['resp_len']==0) & (acs['length_%s' % tix].notna()), 'length_%s' % tix]
            acs.loc[(acs['resp_len']==1) & (acs['length_%s' % tix].notna()), 'resp_length_%s' % tix] = \
            acs.loc[(acs['resp_len']==1) & (acs['length_%s' % tix].notna()), ['type_%s' % tix, 'length_%s' % tix]].apply(lambda x: list(get_dps_lowerBounded(dleni[x[0]], x[1]).keys())[simulate_wof(list(get_dps_lowerBounded(dleni[x[0]], x[1]).values()))], axis=1)

        # -------------
        # Simulate any pay (would be) received from employer, for all takers/needers
        # -------------
        ytr = fmla.loc[ixs_tr]
        ytr = ytr[ytr['anypay'].notna()]['anypay']
        xtr = fmla_xtr.loc[ytr.index]
        wght = fmla.loc[ytr.index][w]
        wght = [x[0] for x in np.array(wght)]
        self.fit_model(xtr, ytr, wght)
        acswids_fit = acs[(acs['taker']==1) | (acs['needer']==1)].index
        phat = self.get_pred_probs(acs_x.loc[acswids_fit])
        phat = pd.DataFrame(phat).set_index(acswids_fit)
        phat.columns = ['p_%s' % x for x in self.clf.classes_]
            # apply wheel of fortune
        phat['anypay'] = phat[phat.columns].apply(lambda x: simulate_wof(x), axis=1)
        acs = acs.join(phat['anypay'])

        # -------------
        # Simulate amount of pay received, for all takers/needers with anypay = 1
        # -------------
        ytr = fmla.loc[ixs_tr]
        # make numeric proportion of pay received as string, for multinomial classification
        ytr = ytr[(ytr['prop_pay'].notna()) & (ytr['prop_pay']>0)][['prop_pay']]['prop_pay'].apply(lambda x: str(x))
        xtr = fmla_xtr.loc[ytr.index]
        wght = fmla.loc[ytr.index][w]
        wght = [x[0] for x in np.array(wght)]
        self.fit_model(xtr, ytr, wght)
        acswids_fit = acs[((acs['taker']==1) | (acs['needer']==1)) & (acs['anypay']==1)].index
        phat = self.get_pred_probs(acs_x.loc[acswids_fit])
        phat = pd.DataFrame(phat).set_index(acswids_fit)
        phat.columns = ['p_%s' % x for x in self.clf.classes_]
        # apply wheel of fortune
        phat['prop_pay_ix'] = phat[phat.columns].apply(lambda x: simulate_wof(x), axis=1)
        phat['prop_pay'] = phat['prop_pay_ix'].apply(lambda x: float(self.clf.classes_[int(x)]))
        phat = phat['prop_pay']
        acs = acs.join(phat)
        # set prop_pay = 0 for anypay = 0
        acs.loc[(acs['anypay'].notna()) & (acs['anypay']==0), 'prop_pay'] = 0

        # -------------
        # Simulate statuses of doctor visit, hospitalization
        # -------------
        for c in ['doctor', 'hospital']:
            ytr = fmla.loc[ixs_tr]
            ytr = ytr[ytr[c].notna()][c]
            xtr = fmla_xtr.loc[ytr.index]
            wght = fmla.loc[ytr.index][w]
            wght = [x[0] for x in np.array(wght)]
            self.fit_model(xtr, ytr, wght)
            acswids_fit = acs[(acs['taker']==1) | (acs['needer']==1)].index
            phat = self.get_pred_probs(acs_x.loc[acswids_fit])
            phat = pd.DataFrame(phat).set_index(acswids_fit)
            phat.columns = ['p_%s' % x for x in self.clf.classes_]
            # apply wheel of fortune
            phat[c] = phat[phat.columns].apply(lambda x: simulate_wof(x), axis=1)
            acs = acs.join(phat[c])
        # make sure hospital=1 implies doctor=1
        acs.loc[(acs['hospital'] == 1) & (acs['doctor'] == 0), 'doctor'] = 1

        #########################
        # TODO: Impute eligibility vars: wks worked, thus weekly wage. Then hourly, employer size, single emp last yr
        #########################

        # Compute program eligibility
        acs['elig_prog'] = 0

        # set eligibility for all gov workers if empgov == True
        if self.empgov:
            acs.loc[acs['empgov_fed']==1, 'elig_prog'] = 1
            acs.loc[acs['empgov_st']==1, 'elig_prog'] = 1
            acs.loc[acs['empgov_loc']==1, 'elig_prog'] = 1
        # set eligibility for self-employed workers if empself == True
        if self.empself:
            acs.loc[acs['COW']==6, 'elig_prog'] = 1 # self-employed, not incorporated
            acs.loc[acs['COW']==7, 'elig_prog'] = 1 # self-employed, incoporated

        # set to 1 for all workers with annual hours >= hrs
        acs['yrhours'] = acs['wkhours'] * acs['weeks_worked_cat']
        acs.loc[acs['yrhours']>=self.hrs, 'elig_prog'] = 1
        del acs['yrhours']

        # set to 0 for all leavers without either doctor visit or hospitalization
        acs.loc[(acs['doctor']==0) & (acs['hospital']==0), 'elig_prog'] = 0

        # check wm-eligibility against FMLA
        elig_wm_acs = sum(acs['elig_prog']*acs['PWGTP']) / sum(acs['PWGTP'])
        elig_wm_fmla = (fmla['coveligd']*fmla['freq_weight']).sum() / fmla[fmla.coveligd.isna()==False]['freq_weight'].sum()
        x = elig_wm_acs/elig_wm_fmla
        print('Estimated mean eligibility in ACS = %s times of mean eligibility in FMLA data' % round(x, 3))

        # Daily wage (not used for eligibility but can be computed once wks worked is imputed)
        # daily wage used for computing program cost later
        acs['wage1d'] = acs['wage12'] / acs['weeks_worked_cat'] / 5
        acs.loc[acs['wage12']==0, 'wage1d'] = 0 # this handles wage12=0 workers who have missing weeks worked
        t1 = time()
        print('Simulation done for all ACS workers in state %s. Time elapsed = %s seconds\n' % (self.st.upper(), round((t1-t0))))
        return acs

    def get_sq_paras(self):
        '''
        :return: status-quo program parameters, now set as an average replacement rate
        '''
        if self.st == 'ca':
            return 0.55
        else:
            return 0

    def get_cost(self, acs, d_takeups, calibration=False, counterfactual=True):
        '''

        :param acs: acs with all leave vars simulated (sourced) from FMLA data
        :param d_takeups: a dict from leave type to takeup rates
        :param rr1: proposed replacement rate
        :param calibration: default is False. If true then alpha = 1 where alpha is calibration factor to account for
                            adjustment in eligibility, employer payment, etc.
        :param counterfactual: if False then will compute cost using status-quo 'length' in acs instead of 'resp_length'
        :return:
        '''

        print('Computing program costs...\n')
        t0 = time()

        # leave types
        types = self.d_type.keys()

        # set parameters for Status-quo / Counterfactual scenarios
        rr0 = self.get_sq_paras()
        if counterfactual:
            rr = self.rr1
            cols_length = ['resp_length_%s' % x for x in range(1, 7)]
            status = 'Counterfactual'
        else:
            rr = rr0
            cols_length = ['length_%s' % x for x in range(1, 7)]
            status = 'Status-quo'
        # set rr0=status-quo replacement rate, so that (rr-rr0)/(1-rr0)=share of rr increase out of total possible increase
        # Adjust full responsive leave length to replacement rate
        # assume full responsive leave length would result under rr = 100%
        for tix in range(1, 7):
            acs.loc[acs['resp_length_%s' % tix].notna(), 'resp_length_%s' % tix] = \
            acs.loc[acs['resp_length_%s' % tix].notna(), ['length_%s' % tix,'resp_length_%s' % tix]] \
                .apply(lambda x: x[0] + (x[1]-x[0])*(self.rr1-rr0)/(1-rr0), axis=1)

        # Compute total program cost
            # compute leave length in work weeks
        acs[['len_wks_%s' % x for x in range(1, 7)]] = acs[cols_length] / 5  # length in work weeks
            # apply leave length cap of each type
        for x in range(1, 7):
            acs.loc[acs['len_wks_%s' % x].notna(), 'len_wks_%s' % x] = acs.loc[acs['len_wks_%s' % x].notna(), ['len_wks_%s' % x, 'type_%s' % x]].apply(lambda x: min(x[0], self.d_max_wk[x[1]]), axis=1)
            # compute total leave length after capping each type
            # Note: no need to cap total or type-specific leave length to a year (52 weeks), as annual total cost would
            # need to include those 'overflow' cost arising from unfinished long leaves from last year
        acs[['len_wks_%s' % x for x in range(1, 7)]] = acs[['len_wks_%s' % x for x in range(1, 7)]].fillna(0)
        acs['len_wks'] = acs[['len_wks_%s' % x for x in range(1, 7)]].apply(lambda x: sum(x), axis=1)
            # compute weekly replacement in dollars for all acs rows
        acs.loc[acs['len_wks'].notna(), 'replace1w'] =  \
        acs.loc[acs['len_wks'].notna(), ['wage1d', 'prop_pay']].apply(lambda x: x[0]*min(rr, (1-x[1]))*5, axis=1)

        acs['replace1w'] = acs['replace1w'].apply(lambda x: min(x, self.max_wk_replace))
            # Compute total cost, alpha is calibration factor
            # up to here, cost_all = total cost if takeup = 1 for all types

            # Determine calibration factor alpha and compute costs of all leaves taken by each individual
        alpha =  1 # set to 1, no adjustment under calibration
        if calibration==False:
            alpha = self.alpha
        acs['cost_all'] = alpha * acs['len_wks'] * acs['replace1w'] * acs['PWGTP'] * acs['elig_prog']

        # Create columns of len_wks_[type]
        for t in self.d_type.keys():
            # any leave is [type] among up to 6 leaves?
            acs['anyleave_%s' % t] = acs[['type_%s' % tix for tix in range(1, 7)]].apply(lambda x: t in [z for z in x], axis=1)
                # if leave [type] exists, find its type index as in type_1, ..., type_6
            acs.loc[acs['anyleave_%s' % t]==True, 'tix_%s' % t] =  \
            acs.loc[acs['anyleave_%s' % t]==True, ['type_%s' % x for x in range(1, 7)]].apply(lambda x: 1 + [z for z in x].index(t), axis=1)
            del acs['anyleave_%s' % t]
            # use the identified type index, set len_wks_[type] = len_wks_[tix]
            acs['len_wks_%s' % t] = 0
            for tix in acs['tix_%s' % t].value_counts().index:
                acs.loc[acs['tix_%s' % t]==tix, 'len_wks_%s' % t] = acs.loc[acs['tix_%s' % t]==tix, 'len_wks_%s' % int(tix)]
            del acs['tix_%s' % t]
            # Create columns of share of len_wks by types
            acs.loc[(acs['len_wks'].notna()) & (acs['len_wks']>0), 'slen_%s' % t] = \
            acs.loc[(acs['len_wks'].notna()) & (acs['len_wks'] > 0), ['len_wks', 'len_wks_%s' % t]].apply(lambda x: x[1]/x[0], axis=1)

        TCs = []  # initiate an output list of total program costs
        for t in self.d_type.keys():
            acs['cost_%s' % t] = acs['cost_all'] * acs['slen_%s' % t] # cost under share of length and full takeup
            TCs.append(round(acs['cost_%s' % t].sum()/10**6, 3)*d_takeups[t]) # type-level takeup handled here
        TCs = [sum(TCs)] + TCs
        TCs = pd.Series(TCs, index= ['Total'] + [self.d_type[type] for type in types])
        TCs_out = TCs[:] # keep a copy of TCs in pd.Series format as output

            # Make a pd.Series of user input
        user_input = pd.Series([])
        user_input['State'] = self.st.upper()
        if counterfactual:
            user_input['Presence of New Program'] = 'Yes'
            user_input['Replacement Rate'] = rr
        else:
            user_input['Presence of New Program'] = 'No'
            user_input['Eligibility Requirement, Minimum Hours Worked'] = self.hrs
            user_input['Replacement Rate'] = rr
        for type in types:
            user_input['Maximum Weeks to Receive Benefit, %s' % self.d_type[type]] = self.d_max_wk[type]
        user_input['Weekly Benefit Cap'] = self.max_wk_replace
        for type in types:
            user_input['Takeup Rate, %s' % self.d_type[type]] = d_takeups[type]
            # Save a output file with user input and total cost combined
        if self.empgov:
            user_input['Government Employees Included'] = 'Yes'
        else:
            user_input['Government Employees Included'] = 'No'
        if self.empself:
            user_input['Self-employed Workers Included'] = 'Yes'
        else:
            user_input['Self-employed Workers Included'] = 'No'

        user_input['Simulation Method'] = self.clf_name
        user_input = user_input.reset_index()
        user_input.columns = ['User Input', 'Value']

        TCs = TCs.reset_index() # move type from index of series TCs to a col
        TCs.columns = ['Leave Type', 'Program Cost, $ Millions']
        TCs['tix'] = 0
        TCs.loc[TCs['Leave Type']=='Own Health', 'tix'] = 1
        TCs.loc[TCs['Leave Type']=='Maternity', 'tix'] = 2
        TCs.loc[TCs['Leave Type']=='New Child', 'tix'] = 3
        TCs.loc[TCs['Leave Type']=='Ill Child', 'tix'] = 4
        TCs.loc[TCs['Leave Type']=='Ill Spouse', 'tix'] = 5
        TCs.loc[TCs['Leave Type']=='Ill Parent', 'tix'] = 6
        TCs = TCs.sort_values(by='tix')
        TCs = TCs.reset_index()
        TCs = TCs.drop(columns=['tix', 'index'])
        out = user_input.join(TCs)
        now = datetime.now()
        out_id = now.strftime('%Y%m%d%H%M%S')
        out.to_csv(self.fp_out + '/program_cost_%s_%s.csv' % (self.st, out_id), index=False)

        print('Simulation Results:')
        print('State = %s, Replacement rate = %s, Status = %s, Total Program Cost = $%s million' % (self.st, rr, status, round(TCs_out['Total'], 1)))
        print('Simulation output saved to %s' % self.fp_out)
        t1 = time()
        print('Time elapsed for computing program costs = %s seconds' % (round(t1-t0)))
        return TCs_out