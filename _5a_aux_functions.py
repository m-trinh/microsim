'''
Auxiliary functions called in class SimulationEngine()

Chris Zhang 10/30/2018
'''

import numpy as np
import pandas as pd
import sklearn.preprocessing, sklearn.linear_model, sklearn.naive_bayes, sklearn.neighbors, sklearn.tree, sklearn.ensemble, \
    sklearn.gaussian_process, sklearn.svm
pd.options.mode.chained_assignment = None

# a function to get columns
def get_columns():
    # original CZ's xvars
    # Xs = ['age', 'agesq', 'male', 'noHSdegree',
    #       'BAplus', 'empgov_fed', 'empgov_st', 'empgov_loc',
    #       'lnfaminc', 'black', 'asian', 'hisp', 'other',
    #       'ndep_kid', 'ndep_old', 'nevermarried', 'partner',
    #       'widowed', 'divorced', 'separated']

    # using LP's xvars
    Xs = ['widowed', 'divorced', 'separated', 'nevermarried',
          'female', 'age','agesq',
          'ltHS', 'someCol', 'BA', 'GradSch',
          'black', 'other', 'asian','native',
          'hisp','nochildren','faminc','coveligd']

    ys = ['take_own', 'take_matdis', 'take_bond', 'take_illchild', 'take_illspouse', 'take_illparent']
    ys += ['need_own', 'need_matdis', 'need_bond', 'need_illchild', 'need_illspouse', 'need_illparent']
    ys += ['resp_len']
    w = 'weight'

    return (Xs, ys, w)

# a function to fill in missing values for binary variables, impute mean-preserving 0/1
def fillna_binary(df, random_state):
    '''
    df: df of pd.Series which are binary cols with missing values
    return: df of pd.Series with missing values filled in as mean-preserving 0/1s
    '''
    _df = pd.DataFrame([])
    for c in df.columns:
        v = df[c]
        nmiss = 0
        try:
            nmiss = v.isna().value_counts()[True]
            draws = random_state.binomial(1, v.mean(), nmiss)
            draws = pd.Series(draws, index=v[v.isna()].index)
            v = v.combine(draws, lambda x0, x1: x0 if not np.isnan(x0) else x1)
        except KeyError:
            pass
        _df[c] = v
    return _df

def get_bool_num_cols(df):
    bool_cols = [c for c in df.columns
                 if df[c].dropna().value_counts().index.isin([0, 1]).all()]
    num_cols = list(set(df.columns) - set(bool_cols))
    return (bool_cols, num_cols)

def fillna_df(df, random_state):
    '''
    df: df with cols either binary or decimal
    return: df missing values filled in as mean-preserving 0/1s for binary, and mean for decimal columns
    '''
    bool_cols, num_cols = get_bool_num_cols(df)
    df[bool_cols] = fillna_binary(df[bool_cols], random_state)
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for c in ['ndep_kid', 'ndep_old']:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: round(x, 0))
        else:
            pass
    return df

# Adjust unconditional prob vector and conditional prob matrix wrt logical restrictions at worker level
def get_adj_ups(ups, chars):
    '''

    :param ups: unconditional probability list before any adjustment, ordered as otypes
    :param chars: worker char list - mmust be [male, nospouse, nochildren]
    :return: adjusted ups
    '''
    ups_adj = ups[:] # copy of list
    # if male - excl. matdis
    if chars[0]==1:
        ups_adj[4] = 0
        ups_adj = [p / sum(ups_adj) for p in ups_adj]
    # if nospouse - excl. illspouse
    if chars[1]==1:
        ups_adj[3] = 0
        ups_adj = [p / sum(ups_adj) for p in ups_adj]
    # if nochildren - excl. bond
    if chars[2]==1:
        ups_adj[0] = 0
        ups_adj = [p / sum(ups_adj) for p in ups_adj]
    return ups_adj

def get_adj_cps(dcp, chars):
    '''

    :param dcp: df of conditional prob, ordered as otypes in both dimensions
    :param chars: worker char list - must be [male, nospouse, nochildren]
    :return: dict based on adjusted dcp
    '''
    # ordered leave types list
    otypes = ['bond','illchild','illparent','illspouse','matdis','own']

    dcp_adj = dcp[:] # copy of dcp df
    # if male - excl. matdis
    if chars[0]==1:
        dcp_adj['matdis'] = 0
        dcp_adj['rowsum'] = dcp_adj.sum(axis=1)
        for t in otypes:
            dcp_adj[t] = dcp_adj[t] / dcp_adj['rowsum']
        del dcp_adj['rowsum']
    # if nospouse - excl. illspouse
    if chars[1]==1:
        dcp_adj['illspouse'] = 0
        dcp_adj['rowsum'] = dcp_adj.sum(axis=1)
        for t in otypes:
            dcp_adj[t] = dcp_adj[t] / dcp_adj['rowsum']
        del dcp_adj['rowsum']
    # if nochildren - excl. bond
    if chars[2]==1:
        dcp_adj['bond'] = 0
        dcp_adj['rowsum'] = dcp_adj.sum(axis=1)
        for t in otypes:
            dcp_adj[t] = dcp_adj[t] / dcp_adj['rowsum']
        del dcp_adj['rowsum']

    dict_dcp_adj = {}
    for t in dcp.columns:
        dict_dcp_adj[t] = list(dcp_adj.loc[t, ])
    return dict_dcp_adj

# a function to get weighted quantile
def get_wquantile(v, w, q):
    '''

    :param v: pd Series of values
    :param w: pd Series of weights in same length as v
    :param q: quantile, between [0,1]
    :return: weighted quantile of v
    '''
    v = v.reset_index(drop=True)
    w = w.reset_index(drop=True)
    vw = pd.DataFrame(v).join(w)
    vname, wname = v.name, w.name
    vw[wname] = vw[wname] / vw[wname].sum()
    vw = pd.DataFrame(vw.groupby(by=vname)[wname].sum()).sort_index()
    vw['cw'] = np.nan
    vw = vw.reset_index()
    for i, row in vw.iterrows():
        if i == 0:
            vw.loc[i, 'cw'] = vw.loc[i, wname]
        else:
            vw.loc[i, 'cw'] = vw.loc[i, wname] + vw.loc[i - 1, 'cw']
    del vw[wname]
    big_cws = [cw for cw in list(vw['cw']) if cw >= q]
    cw_q = min(big_cws)
    v_q = vw.loc[vw['cw'] == cw_q, vname].values[0]

    return v_q

# a function to draw 0/1 from weighted population with prob = p, until target proportion is reached
def get_weighted_draws(ws, p, random_state):
    # ws = pop weight vector
    # p = probability of draw

    # get target population to receive 1s
    target = sum(ws)*p
    # shuffle weights and get cumsums
    ws = np.array(list(enumerate(ws)))
    random_state.shuffle(ws)
    cumsums = np.cumsum([x[1] for x in ws])
    # max index in shuffled indexed weight vector
    # must have equal sign! otherwise if target = 1 then will never reach it, and will receive 0 for all elements
    max_idx_in_shuffled_ws = np.argmax(cumsums >= target)
    # weights receive 1 up to above max index, and receive 0 otherwise
    ws = [x + [1] if ix <= max_idx_in_shuffled_ws else x + [0] for ix, x in enumerate(ws.tolist())]
    # restore original order of weights
    ws = sorted(ws, key=lambda x: x[0])
    # now draws are in same order as original weight vector, return draws only
    draws = [x[2] for x in ws]

    return draws

# a function to get predicted probabilies of classifiers
def get_pred_probs(clf, xts):
    '''
    get predicted probabilities of all classes for the post-fit classfier
    :param clf: the actual classifier post-fit
    :param xts: testing/prediction dataset
    :return: array of list of probs
    '''
    if isinstance(clf, sklearn.linear_model.LogisticRegression) \
        or isinstance(clf, sklearn.ensemble.RandomForestClassifier) \
        or isinstance(clf, sklearn.linear_model.SGDClassifier) \
        or isinstance(clf, sklearn.svm.SVC) \
        or isinstance(clf, sklearn.naive_bayes.BernoulliNB) \
        or isinstance(clf, sklearn.naive_bayes.MultinomialNB) \
        or isinstance(clf, sklearn.neighbors.KNeighborsClassifier):
        phat = clf.predict_proba(xts)
    elif isinstance(clf, sklearn.linear_model.RidgeClassifier):
        d = clf.decision_function(xts)  # distance to hyperplane
        # case of binary classification, d is np.array([...])
        if d.ndim == 1:
            phat = np.exp(d) / (1 + np.exp(d))  # list of pr(yhat = 1)
            phat = np.array([[(1 - x), x] for x in phat])
        # case of multiclass problem (n class >=3), d is np.array([[...]])
        elif d.ndim == 2:
            phat = np.exp(d) / np.array([[x] * len(d[0]) for x in np.exp(d).sum(axis=1)])
    return phat

# a function to simulate from wheel of fortune (e.g. simulate leave type from discrete distribution of 6 types)
def simulate_wof(ps, random_state):
    '''

    :param ps: a list of discrete probabilities, must sum up to 1
    :return: the index of group where the draw indicates
    '''
    # Option 1: np/choice - 1-call code, get index directly, no searching of element 1 involved
    # ix = np.random.choice(len(ps), p=ps)
    # Option 2: np/multinomial - fastest
    ix = list(random_state.multinomial(1, ps, 1)[0]).index(1)
    # Option 3: below also works but much slower esp in pandas lambda (5X+ time)
    # cs = np.cumsum(ps) # cumulative prob vector
    # ix = bisect.bisect(cs, np.random.random())
    return ix

def get_sim_col(X, y, w, Xa, clf, random_state):
    '''

    :param X: training data predictors
    :param y: training data outcomes
    :param w: training data weights
    :param Xa: test / target data predictors
    :param clf: classifier instance
    :return:
    '''
    # Data preparing - fill in nan
    X = fillna_df(X, random_state)
    y = fillna_df(pd.DataFrame(y), random_state)
    y = y[y.columns[0]]
    Xa = fillna_df(Xa, random_state)

    # if clf = 'random draw'
    if clf == 'random draw':
        simcol = [y.iloc[z] for z in random_state.choice(len(y), len(Xa))]
        return simcol
    else:
        # Data preparing - categorization for Naive Bayes
        if isinstance(clf, sklearn.naive_bayes.MultinomialNB):
            # Cateogrize integer variables (ndep_kid, ndep_old) into 0, 1, 2+ groups
            # Categorize decimal variables into binary columns of tercile groups
            num_cols = get_bool_num_cols(X)[1]
            for c in num_cols:
                if c in ['ndep_kid', 'ndep_old']:
                    for df in [X, Xa]:
                        df['%s_0' % c] = (df[c] == 0).astype(int)
                        df['%s_1' % c] = (df[c] == 1).astype(int)
                        df['%s_2' % c] = (df[c] >= 2).astype(int)
                        del df[c]
                else:
                    wq1, wq2 = get_wquantile(X[c], w, 1/3), get_wquantile(X[c], w, 2/3)
                    for df in [X, Xa]:
                        df['%s_ter1' % c] = (df[c] < wq1).astype(int)
                        df['%s_ter2' % c] = ((df[c] >= wq1) & (df[c] < wq2)).astype(int)
                        df['%s_ter3' % c] = (df[c] >= wq2).astype(int)
                        del df[c]
        else:
            # Data preparing - standardization, (x-mu)/sigma
            X = pd.DataFrame(sklearn.preprocessing.scale(X), columns=X.columns)
            Xa = pd.DataFrame(sklearn.preprocessing.scale(Xa), columns=Xa.columns)

        # Fit model
        # Weight config for kNN is specified in clf input before fit. For all other clf weight is specified during fit
        if isinstance(clf, sklearn.neighbors.KNeighborsClassifier):
            f = lambda x: np.array([w]) # make sure weights[:, i] can be called in package code classification.py
            clf = clf.__class__(weights=f)
            clf = clf.fit(X, y)
        else:
            clf = clf.fit(X, y, sample_weight=w)

        # Make prediction
        # get cumulative distribution using phat vector, then draw unif(0,1) to see which segment it falls into
        phat = get_pred_probs(clf, Xa)
        #print('phat top 30 rows = %s ' % phat[:30])
        s = phat.cumsum(axis=1)
        r = random_state.rand(phat.shape[0])
        simcol = pd.Series((r > s.transpose()).transpose().sum(axis=1))
        simcol = list(simcol)
        return simcol

# a function to get marginal probability vector, i.e. change values of probability list to 0 for given indices, and normalize
def get_marginal_probs(ps, ixs):
    mps = ps.copy() # make sure ps is not mutated
    for ix in ixs:
        mps[ix] = 0
    return [p/sum(mps) for p in mps]

# a function to compute spread given a set of weights
def get_mean_spread(df, vcol, wcol, rpw_cols, how='sum'):
    '''

    :param df: master df
    :param vcol: label of col of interest
    :param wcol: main weight col label (for mean)
    :param rpw_cols: replication weight col labels
    :param how: 'sum' or 'mean', the way to compute statistic for vcol
    :return: mean, spread (half length of c.i.) at 95%
    '''
    # mean
    if how == 'sum':
        mean = (df[vcol] * df[wcol]).sum()
    elif how == 'mean':
        mean = (df[vcol] * df[wcol] / df[df[vcol].notna()][wcol].sum()).sum()
    else:
        mean = np.nan
    # standard error square
    sesq = 0
    for wt in rpw_cols:
        if how == 'sum':
            sesq += 4 / 80 * ((df[vcol] * df[wt]).sum() - mean) ** 2
        elif how == 'mean':
            sesq += 4 / 80 * ((df[df[vcol].notna()][vcol] * df[df[vcol].notna()][wt] / df[df[vcol].notna()][wt].sum()).sum() - mean) ** 2
        else:
            sesq = np.nan
    spread = sesq ** 0.5 * 1.96
    return mean, spread

# a function to simulate leave types for multiple leavers (optional in fmla data cleaning code)
def get_multiple_leave_vars(d, types, random_state):
    '''

    :param d: FMLA dataset
    :param types: leave types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
    :return: d with multiple leaver variables generated
    '''

    # number of reasons leaves taken - need to consolidate following
    # A5_1_CAT: loop 1 taker reason type
    # A5_2_CAT: loop 2 taker reason type
    # A4a_CAT: number of taker reasons

    # get total of take reasons(out of 6) inferred from A5_1_CAT and A5_2_CAT
    d = d.rename(columns={'A5_2_CAT_REV': 'A5_2_CAT_rev'})
    for lp in ['1', '2']:
        d['take_any6_loop%s' % lp] = 0
        d['take_any6_loop%s' % lp] = np.where((d['A5_%s_CAT' % lp] == 1) |
                                              (d['A5_%s_CAT' % lp] == 11) |
                                              (d['A5_%s_CAT' % lp] == 12) |
                                              (d['A5_%s_CAT' % lp] == 13) |
                                              (d['A5_%s_CAT' % lp] == 21) |
                                              (d['A5_%s_CAT_rev' % lp] == 32), 1, 0)
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
    # [check] print(d[(d['take_any6']==1) & (d['take_type'].isna())][['take_any6', 'take_type', 'A5_1_CAT', 'A5_2_CAT', 'A20']])
    # because they all have A20 = 2 and when searching for recent leave from loop 2, the type is 'other' or nan
    # for them we use A5_1_CAT as most recent leave taken, and breakdown New Child into matdis and bond using loop1 info
    d.loc[(d['take_any6'] == 1) & (d['take_type'].isna()), 'take_type'] = \
        d.loc[(d['take_any6'] == 1) & (d['take_type'].isna()), 'A5_1_CAT'].apply(lambda x: dctr[x])
    d.loc[((d['take_type'] == 'New Child')
           & (d['A11_1'] == 1)
           & (d['GENDER_CAT'] == 2)), 'take_type'] = 'matdis'
    d.loc[(d['take_type'] == 'New Child'), 'take_type'] = 'bond'
    # check
    # [check] print(d[(d['take_any6']==1) & (d['take_type'].isna())][['take_any6', 'take_type', 'A5_1_CAT', 'A5_2_CAT', 'A20']])

    # similarly, identified following needers
    # [check] print(d[(d['need_any6']==1) & (d['need_type'].isna())][['need_any6', 'need_type', 'B6_1_CAT', 'B6_2_CAT']])
    # similarly, force most recent leave type for them, by using B6_2_CAT
    d.loc[(d['need_any6'] == 1) & (d['need_type'].isna()), 'need_type'] = \
        d.loc[(d['need_any6'] == 1) & (d['need_type'].isna()), 'B6_2_CAT'].apply(lambda x: dctr[x])
    d.loc[((d['need_type'] == 'New Child')
           & (d['B12_1'] == 1)
           & (d['GENDER_CAT'] == 2)), 'need_type'] = 'matdis'
    d.loc[(d['need_type'] == 'New Child'), 'need_type'] = 'bond'
    # check
    # [check] print(d[(d['need_any6']==1) & (d['need_type'].isna())][['need_any6', 'need_type', 'B6_1_CAT', 'B6_2_CAT']])

    # Check: so far for take_any6 > 0, take_type and take_type2 should be all defined for num_leaves_taken = 1 or 2
    d[(d['take_any6'] > 0) & ((d['num_leaves_taken'] == 1) | (d['num_leaves_taken'] == 2))][
        'take_type'].isna().value_counts()
    d[(d['take_any6'] > 0) & (d['num_leaves_taken'] == 2)]['take_type2'].isna().value_counts()

    d[(d['need_any6'] == 1) & ((d['num_leaves_need'] == 1) | (d['num_leaves_need'] == 2))][
        'need_type'].isna().value_counts()

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
    otypes = list(dcp.columns)  # types in alphabetical order

    # -------------
    # Unconditional prob of taking leaves if leave type is unknown
    # this will be used to simulate types of multi-leavers if missing types for all leaves
    # possible if multi-leavers and reported loop-1/2 types are out of the 6 types
    # -------------
    dict_dup = {}
    denom = 0
    for t in dcp.columns:
        num = d[d['take_%s' % t] == 1]['weight'].sum()
        denom += num  # build denom this way so that ratios sum up to 1
    ups = []  # unconditional probs of taken types in order of dcp.columns (alphabetical)
    for t in dcp.columns:
        num = d[d['take_%s' % t] == 1]['weight'].sum()
        dict_dup[t] = num / denom
        ups.append(num / denom)

    # Impute rest type info not in survey loops, up to num_leaves_taken/need

    # Before imputing leave types, consolidate num_leaves_taken/need - among 2 loops some reported types out of the 6 types
    # For these workers reduce num_leaves accordingly
    d['num_leaves_taken_adj'] = d['num_leaves_taken']
    d.loc[(d['num_leaves_taken'] == 1) & (d['take_type'].isna()), 'num_leaves_taken_adj'] = 0
    for n in range(2, 7):
        d.loc[(d['num_leaves_taken'] == n) & (
        d['take_type'].isna()), 'num_leaves_taken_adj'] = n - 2  # if take_type is nan, then
        #  take_type2 by definition would be nan too, reduce num_leaves by 2
        d.loc[(d['num_leaves_taken'] == n) & (d['take_type'].notna()) & (
        d['take_type2'].isna()), 'num_leaves_taken_adj'] = n - 1
    d['num_leaves_taken'] = d['num_leaves_taken_adj']
    # Similarly reduce for num_leaves_need
    d.loc[(d['num_leaves_need'] == 1) & (d['need_type'].isna()), 'num_leaves_taken'] = 0
    for n in range(2, 7):
        d.loc[(d['num_leaves_need'] == n) & (
        d['need_type'].isna()), 'num_leaves_need'] = n - 2  # if need_type is nan, then
        #  need_type2 by definition would be nan too, reduce num_leaves by 2
        d.loc[(d['num_leaves_need'] == n) & (d['need_type'].notna()) & (
        d['need_type2'].isna()), 'num_leaves_need'] = n - 1
    # Further cap num_leaves wrt logical restrictions
    # if male - excl. matdis
    # if nospouse (nevermarried, separated, divorced, widowed) - excl. illspouse
    # if nochildren - excl. bond

    d['max_num_leaves'] = 6
    d.loc[d['male'] == 1, 'max_num_leaves'] = d.loc[d['male'] == 1, 'max_num_leaves'] - 1
    d.loc[d['nospouse'] == 1, 'max_num_leaves'] = d.loc[d['nospouse'] == 1, 'max_num_leaves'] - 1
    d.loc[d['nochildren'] == 1, 'max_num_leaves'] = d.loc[d['nochildren'] == 1, 'max_num_leaves'] - 1
    d['num_leaves_taken'] = d[['num_leaves_taken', 'max_num_leaves']].apply(lambda x: min(x), axis=1)
    d['num_leaves_need'] = d[['num_leaves_need', 'max_num_leaves']].apply(lambda x: min(x), axis=1)

    # Impute leave types for multiple leaves - note that in general we have to do this for num_leaves = 1, 2, ... , 6
    # because the adjusted num_leaves = 1 can come from original num_leaves = 3 but with both reported loops having types
    # out of the 6 main types
    chars = ['male', 'nospouse', 'nochildren']
    # Impute for take types
    # Impute types for most recent take_type for num_leaves_taken = 1...6
    for nlt in range(1, 7):
        L = len(d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].isna()), 'take_type'] = \
                d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].isna()), chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(get_adj_ups(ups, x), random_state)], axis=1)
        else:
            pass

    # Impute types for 2nd most recent take_type2 for num_leaves_taken = 2...6
    for nlt in range(2, 7):
        L = len(d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].notna()) & (d['take_type2'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].notna()) & (d['take_type2'].isna()), 'take_type2'] = \
                d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type'].notna()) & (d['take_type2'].isna()), [
                    'take_type'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(get_adj_cps(dcp, x[1:])[x[0]], random_state)], axis=1)
        else:
            pass

    # Impute types for 3rd most recent take_type3 for num_leaves_taken = 3...6
    # when impute need to discard candidate types that have been reported / imputed in exising leaves
    d['take_type3'] = np.nan
    for nlt in range(3, 7):
        L = len(d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type2'].notna()) & (d['take_type3'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type2'].notna()) & (d['take_type3'].isna()), 'take_type3'] = \
                d.loc[
                    (d['num_leaves_taken'] == nlt) & (d['take_type2'].notna()) & (d['take_type3'].isna()), ['take_type',
                                                                                                            'take_type2'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(
                    get_marginal_probs(get_adj_cps(dcp, x[2:])[x[1]], [otypes.index(x[0]), otypes.index(x[1])]), random_state)],
                          axis=1)
        else:
            pass

    # Impute types for 4th most recent take_type4 for num_leaves_taken = 4...6
    # when impute need to discard candidate types that have been reported / imputed in exising leaves
    d['take_type4'] = np.nan
    for nlt in range(4, 7):
        L = len(d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type3'].notna()) & (d['take_type4'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type3'].notna()) & (d['take_type4'].isna()), 'take_type4'] = \
                d.loc[
                    (d['num_leaves_taken'] == nlt) & (d['take_type3'].notna()) & (d['take_type4'].isna()), ['take_type',
                                                                                                            'take_type2',
                                                                                                            'take_type3'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(
                    get_marginal_probs(get_adj_cps(dcp, x[3:])[x[2]], [otypes.index(x[k]) for k in range(3)]), random_state)], axis=1)
        else:
            pass

    # Impute types for 5th most recent take_type5 for num_leaves_taken = 5, 6
    # when impute need to discard candidate types that have been reported / imputed in exising leaves
    d['take_type5'] = np.nan
    for nlt in range(5, 7):
        L = len(d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type4'].notna()) & (d['take_type5'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_taken'] == nlt) & (d['take_type4'].notna()) & (d['take_type5'].isna()), 'take_type5'] = \
                d.loc[
                    (d['num_leaves_taken'] == nlt) & (d['take_type4'].notna()) & (d['take_type5'].isna()), ['take_type',
                                                                                                            'take_type2',
                                                                                                            'take_type3',
                                                                                                            'take_type4'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(
                    get_marginal_probs(get_adj_cps(dcp, x[:4])[x[3]], [otypes.index(x[k]) for k in range(4)]), random_state)], axis=1)
        else:
            pass

    # Impute types for last take_type6 for num_leaves_taken = 6
    # this is just the only left leave type not selected by previous simulation
    # no logical restriction needed if all 6 types exist
    d['take_type6'] = np.nan
    L = len(d.loc[(d['num_leaves_taken'] == 6) & (d['take_type5'].notna()) & (d['take_type6'].isna())])
    if L > 0:
        d.loc[(d['num_leaves_taken'] == 6) & (d['take_type5'].notna()) & (d['take_type6'].isna()), 'take_type6'] = \
            d.loc[(d['num_leaves_taken'] == 6) & (d['take_type5'].notna()) & (d['take_type6'].isna()), ['take_type',
                                                                                                        'take_type2',
                                                                                                        'take_type3',
                                                                                                        'take_type4',
                                                                                                        'take_type5']] \
                .apply(lambda x: list(set(otypes) - set(x))[0], axis=1)
    else:
        pass

    # Impute for need types
    # B6_1_CAT has enough obs so use need type-based ups as unconditional prob vector
    # B6_2_CAT has too few obs so keep using take type-based dcp as conditional prob matrix
    dict_dup = {}
    denom = 0
    for t in dcp.columns:
        num = d[d['need_%s' % t] == 1]['weight'].sum()
        denom += num  # build denom this way so that ratios sum up to 1
    ups = []  # unconditional probs of need types in order of dcp.columns (alphabetical)
    for t in dcp.columns:
        num = d[d['need_%s' % t] == 1]['weight'].sum()
        dict_dup[t] = num / denom
        ups.append(num / denom)

    # Impute types for most recent need_type for num_leaves_need = 1...6
    for nlt in range(1, 7):
        L = len(d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].isna()), 'need_type'] = \
                d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].isna()), chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(get_adj_ups(ups, x), random_state)], axis=1)
        else:
            pass

    # Impute types for 2nd most recent need_type2 for num_leaves_taken = 2...6
    for nlt in range(2, 7):
        L = len(d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].notna()) & (d['need_type2'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].notna()) & (d['need_type2'].isna()), 'need_type2'] = \
                d.loc[(d['num_leaves_need'] == nlt) & (d['need_type'].notna()) & (d['need_type2'].isna()), [
                    'need_type'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(get_adj_cps(dcp, x[1:])[x[0]], random_state)], axis=1)
        else:
            pass

    # Impute types for 3rd most recent need_type3 for num_leaves_taken = 3...6
    # when impute need to discard candidate types that have been reported / imputed in existing leaves
    d['need_type3'] = np.nan
    otypes = list(dcp.columns)  # types in alphabetical order
    for nlt in range(3, 7):
        L = len(d.loc[(d['num_leaves_need'] == nlt) & (d['need_type2'].notna()) & (d['need_type3'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_need'] == nlt) & (d['need_type2'].notna()) & (d['need_type3'].isna()), 'need_type3'] = \
                d.loc[
                    (d['num_leaves_need'] == nlt) & (d['need_type2'].notna()) & (d['need_type3'].isna()), ['need_type',
                                                                                                           'need_type2'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(
                    get_marginal_probs(get_adj_cps(dcp, x[2:])[x[1]], [otypes.index(x[0]), otypes.index(x[1])]), random_state)],
                          axis=1)
        else:
            pass

    # Impute types for 4th most recent need_type4 for num_leaves_taken = 4...6
    # when impute need to discard candidate types that have been reported / imputed in existing leaves
    d['need_type4'] = np.nan
    for nlt in range(4, 7):
        L = len(d.loc[(d['num_leaves_need'] == nlt) & (d['need_type3'].notna()) & (d['need_type4'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_need'] == nlt) & (d['need_type3'].notna()) & (d['need_type4'].isna()), 'need_type4'] = \
                d.loc[
                    (d['num_leaves_need'] == nlt) & (d['need_type3'].notna()) & (d['need_type4'].isna()), ['need_type',
                                                                                                           'need_type2',
                                                                                                           'need_type3'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(
                    get_marginal_probs(get_adj_cps(dcp, x[3:])[x[2]], [otypes.index(x[k]) for k in range(3)]), random_state)], axis=1)
        else:
            pass

    # Impute types for 5th most recent need_type5 for num_leaves_taken = 5, 6
    # when impute need to discard candidate types that have been reported / imputed in existing leaves
    d['need_type5'] = np.nan
    for nlt in range(5, 7):
        L = len(d.loc[(d['num_leaves_need'] == nlt) & (d['need_type4'].notna()) & (d['need_type5'].isna())])
        if L > 0:
            d.loc[(d['num_leaves_need'] == nlt) & (d['need_type4'].notna()) & (d['need_type5'].isna()), 'need_type5'] = \
                d.loc[
                    (d['num_leaves_need'] == nlt) & (d['need_type4'].notna()) & (d['need_type5'].isna()), ['need_type',
                                                                                                           'need_type2',
                                                                                                           'need_type3',
                                                                                                           'need_type4'] + chars]. \
                    apply(lambda x: dcp.columns[simulate_wof(
                    get_marginal_probs(get_adj_cps(dcp, x[4:])[x[3]], [otypes.index(x[k]) for k in range(4)]), random_state)], axis=1)
        else:
            pass

    # Impute types for last need_type6 for num_leaves_need = 6
    # this is just the only left leave type not selected by previous simulation
    # no logical restriction needed if all 6 types exist
    d['need_type6'] = np.nan
    L = len(d.loc[(d['num_leaves_need'] == 6) & (d['need_type5'].notna()) & (d['need_type6'].isna())])
    if L > 0:
        d.loc[(d['num_leaves_need'] == 6) & (d['need_type5'].notna()) & (d['need_type6'].isna()), 'need_type6'] = \
            d.loc[(d['num_leaves_need'] == 6) & (d['need_type5'].notna()) & (d['need_type6'].isna()), ['need_type',
                                                                                                       'need_type2',
                                                                                                       'need_type3',
                                                                                                       'need_type4',
                                                                                                       'need_type5']] \
                .apply(lambda x: list(set(otypes) - set(x))[0], axis=1)
    else:
        pass

    # most recent leave length by leave type
    types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']
    for t in types:
        d['length_%s' % t] = np.where(d['take_%s' % t] == 1, d['length'], 0)
        d['length_%s' % t] = np.where(d['take_%s' % t].isna(), np.nan, d['length_%s' % t])

    return d
