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


class SimulationEngine:

    def __init__(self, st, yr, fps_in, fps_out, clf_name, prog_para, engine_type='main'):
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
        self.fp_acsh_in = fps_in[2] # directory only for raw ACS household file, excl. csv file ext
        self.fp_acsp_in = fps_in[3] # directory only for raw ACS person file, excl. csv file ext
        self.fp_fmla_out = fps_out[0]
        self.fp_cps_out = fps_out[1]
        self.fp_acs_out = fps_out[2] # directory only for cleaned ACS file, excl. csv file ext
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
        self.type = engine_type
        if self.type == 'main':
            self.output_directory = './output/output_%s' % self.out_id
        else:
            self.output_directory = './output/output_%s_%s' % (self.out_id, self.type)

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
        self.updates.append('File saved: clean FMLA data file before CPS imputation.')
        message = dcf.impute_fmla_cps(self.fp_cps_in, self.fp_cps_out)
        self.updates.append(message)
        self.updates.append('File saved: clean FMLA data file after CPS imputation.')
        dcf.get_length_distribution(self.fp_length_distribution_out)
        self.updates.append('File saved: leave distribution estimated from FMLA data.')
        self.progress = 25

        self.updates.append('Cleaning ACS data. State chosen = RI. Chunk size = 100000 ACS rows')
        dca = DataCleanerACS(self.st, self.yr, self.fp_acsh_in, self.fp_acsp_in, self.fp_acs_out)
        dca.load_data()
        message = dca.clean_person_data()
        self.updates.append(message)
        self.progress = 60
        return None

    def get_acs_simulated(self):

        # Set up timer
        tsim = time()

        # Read in cleaned ACS and FMLA data, and FMLA-based length distribution
        acs = pd.read_csv(self.fp_acs_out + 'ACS_cleaned_forsimulation_20%s_%s.csv' % (self.yr, self.st))
        pfl = 'non-PFL' # status of PFL as of ACS sample period
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
            #print('Simulation of col %s done. Time elapsed = %s' % (c, (time()-tt)))
        #print('6+6+1 simulated. Time elapsed = %s' % (time()-t0))

        # Post-simluation logic control
        acs.loc[acs['male']==1, 'take_matdis']=0
        acs.loc[acs['male']==1, 'need_matdis']=0
        acs.loc[(acs['nevermarried']==1) | (acs['divorced']==1), 'take_illspouse'] = 0
        acs.loc[(acs['nevermarried']==1) | (acs['divorced']==1), 'need_illspouse'] = 0
        acs.loc[acs['nochildren']==1, 'take_bond'] = 0
        acs.loc[acs['nochildren']==1, 'need_bond'] = 0
        acs.loc[acs['nochildren']==1, 'take_matdis'] = 0
        acs.loc[acs['nochildren']==1, 'need_matdis'] = 0

        # Conditional simulation - anypay, doctor, hospital for taker/needer sample
        acs['taker'] = acs[['take_%s' % t for t in self.types]].apply(lambda x: max(x), axis=1)
        acs['needer'] = acs[['need_%s' % t for t in self.types]].apply(lambda x: max(x), axis=1)

        X = d[(d['taker']==1) | (d['needer']==1)][col_Xs]
        w = d.loc[X.index][col_w]
        Xa = acs[(acs['taker']==1) | (acs['needer']==1)][X.columns]
        if len(Xa)==0:
            pass
        else:
            for c in ['anypay', 'doctor', 'hospital']:
                y = d.loc[X.index][c]
                acs = acs.join(get_sim_col(X, y, w, Xa, clf))
            # Post-simluation logic control
            acs.loc[acs['hospital']==1, 'doctor'] = 1

        # Conditional simulation - prop_pay for anypay=1 sample
        X = d[(d['anypay']==1) & (d['prop_pay'].notna())][col_Xs]
        w = d.loc[X.index][col_w]
        Xa = acs[acs['anypay']==1][X.columns]
        # a dict from prop_pay int category to numerical prop_pay value
        # int category used for phat 'p_0', etc. in get_sim_col
        v = d.prop_pay.value_counts().sort_index().index
        k = range(len(v))
        d_prop = dict(zip(k, v))
        D_prop = dict(zip(v, k))

        if len(Xa)==0:
            pass
        else:
            y = d.loc[X.index]['prop_pay'].apply(lambda x: D_prop[x])
            yhat = get_sim_col(X, y, w, Xa, clf)
            # prop_pay labels are from 1 to 6, get_sim_col() vectorization sum gives 0~5, increase label by 1
            yhat = pd.Series(data=yhat.values + 1, index=yhat.index, name='prop_pay')
            acs = acs.join(yhat)
            acs.loc[acs['prop_pay'].notna(), 'prop_pay'] = acs.loc[acs['prop_pay'].notna(), 'prop_pay'].apply(lambda x: d_prop[x])

        # Draw leave length for each type
        # Without-program lengths - draw from FMLA-based distribution (pfl indicator = 0)
        # note: here, cumsum/bisect is 20% faster than np/choice.
        # But when simulate_wof applied as lambda to df, np/multinomial is 5X faster!
        t0 = time()
        for t in self.types:
            acs['len_%s' % t] = 0
            n_lensim = len(acs.loc[acs['take_%s' % t]==1]) # number of acs workers who need length simulation
            #print(n_lensim)
            ps = [x[1] for x in flen[pfl][t]] # prob vector of length of type t
            cs = np.cumsum(ps)
            lens = [] # initiate list of lengths
            for i in range(n_lensim):
                lens.append(flen[pfl][t][bisect.bisect(cs, np.random.random())][0])
            acs.loc[acs['take_%s' % t]==1, 'len_%s' % t] = np.array(lens)
            #print('mean = %s' % acs['len_%s' % t].mean())
        #print('te: sq length sim = %s' % (time()-t0))

        # Max needed lengths (mnl) - draw from simulated without-program length distribution
        # conditional on max length >= without-program length
        T0 = time()
        for t in self.types:
            t0 = time()
            acs['mnl_%s' % t] = 0
            # resp_len = 0 workers' mnl = sq length
            acs.loc[acs['resp_len']==0, 'mnl_%s' % t] = acs.loc[acs['resp_len']==0, 'len_%s' % t]
            # resp_len = 1 workers' mnl draw from length distribution conditional on new length > sq length
            dct_vw = {} # dict from sq length to possible greater length value, and associated weight of worker who provides the length
            x_max = acs['len_%s' % t].max()
            for x in acs['len_%s' % t].value_counts().index:
                if x<x_max:
                    dct_vw[x] = acs[(acs['len_%s' % t] > x)][['len_%s' % t, 'PWGTP']].groupby(by='len_%s' % t)['PWGTP'].sum().reset_index()
                    mx = len(acs[(acs['resp_len'] == 1) & (acs['len_%s' % t] == x)])
                    vxs = np.random.choice(dct_vw[x]['len_%s' % t], mx, p=dct_vw[x]['PWGTP'] / dct_vw[x]['PWGTP'].sum())
                    acs.loc[(acs['resp_len']==1) & (acs['len_%s' % t]==x), 'mnl_%s' % t] = vxs
                else:
                    acs.loc[(acs['resp_len']==1) & (acs['len_%s' % t]==x), 'mnl_%s' % t] = x*1.25
            #print('mean = %s. MNL sim done for type %s. telapse = %s' % (acs['mnl_%s' % t].mean(), t, (time()-t0)))

        # logic control of mnl
        acs.loc[acs['male']==1, 'mnl_matdis']=0
        acs.loc[(acs['nevermarried']==1) | (acs['divorced']==1), 'mnl_illspouse'] = 0
        acs.loc[acs['nochildren']==1, 'mnl_bond'] = 0
        acs.loc[acs['nochildren']==1, 'mnl_matdis'] = 0

        #print('All MNL sim done. TElapsed = %s' % (time()-T0))

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
        acs = acs.drop(acs[(acs['taker']==0) & (acs['needer']==0)].index)

        if not self.incl_empgov_fed:
            acs = acs.drop(acs[acs['empgov_fed']==1].index)
        if not self.incl_empgov_st:
            acs = acs.drop(acs[acs['empgov_st']==1].index)
        if not self.incl_empgov_loc:
            acs = acs.drop(acs[acs['empgov_loc']==1].index)
        if not self.incl_empself:
            acs = acs.drop(acs[(acs['COW']==6) | (acs['COW']==7)].index)

        # program eligibility - TODO: port to GUI input, program eligibility determinants
        acs['elig_prog'] = 0

        elig_empsizebin = 0
        if 1<= self.elig_empsize < 10:
            elig_empsizebin = 1
        elif 10 <= self.elig_empsize <=49:
            elig_empsizebin = 2
        elif 50 <= self.elig_empsize <=99:
            elig_empsizebin = 3
        elif 100 <= self.elig_empsize <=499:
            elig_empsizebin = 4
        elif 500 <= self.elig_empsize <=999:
            elig_empsizebin = 5
        elif self.elig_empsize >=1000:
            elig_empsizebin = 6

        acs.loc[(acs['wage12']>=self.elig_wage12) &
                (acs['wkswork']>=self.elig_wkswork) &
                (acs['wkswork']*acs['wkhours']>=self.elig_yrhours) &
                (acs['empsize']>=elig_empsizebin), 'elig_prog'] = 1

        acs = acs.drop(acs[acs['elig_prog']!=1].index)

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
                acs['len_%s' % t] + (acs['mnl_%s' % t] - acs['len_%s' % t]) * (self.rrp - acs['prop_pay']) / (1 - acs['prop_pay'])
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
        message = 'Leaves simulated for 5-year ACS 20%s-20%s in state %s. Time needed = %s seconds' % ((self.yr-4), self.yr, self.st.upper(), round(time()-tsim, 0))
        print(message)
        self.updates.append(message)
        self.progress = 95
        return acs

    def get_cost(self):
        # read simulated ACS
        acs = pd.read_csv('%s/acs_sim_%s.csv' % (self.output_directory, self.out_id))
        # apply take up rates and weekly benefit cap, and compute total cost, 6 types
        costs = {}
        for t in self.types:
            v = ((acs['cpl_%s' % t]/5) * (acs['wage12'] / acs['wkswork'] * self.rrp)).apply(lambda x: min(x, self.wkbene_cap))
            w = acs['PWGTP']*self.d_takeup[t]
            costs[t] = (v*w).sum()
        costs['total'] = sum(list(costs.values()))
        # compute standard error using replication weights, then compute confidence interval
        sesq = dict(zip(costs.keys(), [0]*len(costs.keys())))
        for wt in ['PWGTP%s' % x for x in range(1, 81)]:
            costs1 = {}
            for t in self.types:
                v = (acs['cpl_%s' % t]/5) * (acs['wage12'] / acs['wkswork'] * self.rrp).apply(lambda x: min(x, self.wkbene_cap))
                w = acs[wt]*self.d_takeup[t]
                costs1[t] = (v*w).sum()
            costs1['total'] = sum(list(costs1.values()))
            for k in costs1.keys():
                sesq[k] += 4/80 * (costs[k] - costs1[k])**2
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
        self.updates.append(message)
        self.progress = 100
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
        d['cpl'] = d[['cpl_%s' % t for t in self.types]].apply(lambda x: sum(x), axis=1)
        # keep needed vars for population analysis plots
        d = d[['PWGTP', 'cpl', 'female', 'age']]
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