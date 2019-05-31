'''
Auxiliary functions called in class SimulationEngine()

Chris Zhang 10/30/2018
'''

import numpy as np
import pandas as pd
import sklearn.linear_model, sklearn.naive_bayes, sklearn.neighbors, sklearn.tree, sklearn.ensemble, \
    sklearn.gaussian_process, sklearn.svm
pd.options.mode.chained_assignment = None

# a function to get columns
def get_columns():
    Xs = ['age', 'agesq', 'male', 'noHSdegree',
          'BAplus', 'empgov_fed', 'empgov_st', 'empgov_loc',
          'lnfaminc', 'black', 'asian', 'hisp', 'other',
          'ndep_kid', 'ndep_old', 'nevermarried', 'partner',
          'widowed', 'divorced', 'separated']
    ys = ['take_own', 'take_matdis', 'take_bond', 'take_illchild', 'take_illspouse', 'take_illparent']
    ys += ['need_own', 'need_matdis', 'need_bond', 'need_illchild', 'need_illspouse', 'need_illparent']
    ys += ['resp_len']
    w = 'weight'

    return (Xs, ys, w)

# a function to fill in missing values for binary variables, impute mean-preserving 0/1
def fillna_binary(df):
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
            draws = np.random.binomial(1, v.mean(), nmiss)
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

def fillna_df(df):
    '''
    df: df with cols either binary or decimal
    return: df missing values filled in as mean-preserving 0/1s for binary, and mean for decimal columns
    '''
    bool_cols, num_cols = get_bool_num_cols(df)
    df[bool_cols] = fillna_binary(df[bool_cols])
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

# a function to get predicted probabilies of classifiers
def get_pred_probs(clf, xts):
    '''
    get predicted probabilities of all classes for the post-fit classfier
    :param clf: the actual classifier post-fit
    :param xts: testing/prediction dataset
    :return: array of list of probs
    '''
    if isinstance(clf, sklearn.linear_model.logistic.LogisticRegression) \
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
def simulate_wof(ps):
    '''

    :param ps: a list of discrete probabilities, must sum up to 1
    :return: the index of group where the draw indicates
    '''
    # Option 1: np/choice - 1-call code, get index directly, no searching of element 1 involved
    # ix = np.random.choice(len(ps), p=ps)
    # Option 2: np/multinomial - fastest
    ix = list(np.random.multinomial(1, ps, 1)[0]).index(1)
    # Option 3: below also works but much slower esp in pandas lambda (5X+ time)
    # cs = np.cumsum(ps) # cumulative prob vector
    # ix = bisect.bisect(cs, np.random.random())
    return ix

def get_sim_col(X, y, w, Xa, clf):
    '''

    :param X: training data predictors
    :param y: training data outcomes
    :param w: training data weights
    :param Xa: test / target data predictors
    :param clf: classifier instance
    :return:
    '''
    # Data preparing - fill in nan
    X = fillna_df(X)
    y = fillna_df(pd.DataFrame(y))
    y = y[y.columns[0]]
    Xa = fillna_df(Xa)

    # if clf = 'random draw'
    if clf=='random draw':
        yhat = [y.iloc[z] for z in np.random.choice(len(y), len(Xa))]
        simcol = pd.Series(yhat, index=Xa.index)
        simcol.name = y.name
        return simcol
    else:
        # Data preparing - categorization for Naive Bayes
        if isinstance(clf, sklearn.naive_bayes.BernoulliNB):
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
            pass

        # Fit model
        # Weight config for kNN is specified in clf input before fit. For all other clf weight is specified during fit
        if isinstance(clf, sklearn.neighbors.KNeighborsClassifier):
            f = lambda x: np.array([w]) # make sure weights[:, i] can be called in package code classification.py
            clf = clf.__class__(weights=f)
            clf = clf.fit(X, y)
        else:
            clf = clf.fit(X, y, sample_weight=w)

        # Make prediction
        if isinstance(clf, sklearn.linear_model.LinearRegression):
            # simple OLS, get prediction directly
            simcol = pd.Series(clf.predict(Xa), index=Xa.index)
        else:
            # probabilistic outcomes - get predicted probs, convert to df, assign target sample index for merging, assign col names
            phat = get_pred_probs(clf, Xa)
            phat = pd.DataFrame(phat).set_index(Xa.index)
            phat.columns = ['p_%s' % int(x) for x in clf.classes_]
            # Option 1: full vectorization
            phat = get_pred_probs(clf, Xa)
            s = phat.cumsum(axis=1)
            r = np.random.rand(phat.shape[0])
            simcol = pd.Series((r > s.transpose()).transpose().sum(axis=1), index=Xa.index)
            # Option 2: 5X slower than Option 1
            # simcol = phat[phat.columns].apply(lambda x: simulate_wof(x), axis=1)
            ## Option 3: 20X slower than Option 1
            # simcol = []
            # for i in range(len(phat)):
            #     simcol.append(simulate_wof(phat.iloc[i]))
            # simcol = pd.Series(simcol)
        simcol.name = y.name
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