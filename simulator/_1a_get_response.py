'''
get response of leave taking in presence of new program

To do: need a hyperparameter alpha to capture effect of program generosity (replacement rate) on response length

can use California data (with current replacement > 0) to calibrate

Chris Zhang 10/1/2018
'''

# -------------
# Housekeeping
# -------------

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from time import time

# -------------
# Function to create an indicator variable of those that would change leave behaviour
# in presence of a new program.
# -------------

def set_data(df):
    '''
    This function adds a column to fmla data that indicates whether leave behavior
    would change
    input:  dataframe of cleaned fmla data
    output: fmla datafram with a new column 'resp_len' which flags whether a worker would respond to new program by taking longer leaves
    '''
     
    # Set index to uniquely identify workers
    df = df.set_index('empid')

    # fillna for current leave length (this still leaves some NA's. For eg. LEAVE_CAT=2,3)
    df['length'] = df[['LEAVE_CAT', 'length']].groupby('LEAVE_CAT').transform(lambda x: x.fillna(x.mean()))

    # -------------
    # The rest of the code in this function creates a variable 'resp_len' 
    # which will flag 0/1 for a worker that is likely to a more favourable leave program
    # by increasing leave length. It is used to help simulate counterfactual leave
    # for a program that increases wage replacement
    # -------------

    # Initiate
    df['resp_len'] = np.nan
        
    # LEAVE_CAT: employed only
    # EMPLOYED ONLY workers have no need and take no leave, would not respond anyway
    df.loc[df['LEAVE_CAT']==3, 'resp_len'] = 0

    # A55 asks if worker would take longer leave if paid?
    df.loc[(df['resp_len'].isna()) & (df['A55']==2), 'resp_len'] = 0
    df.loc[(df['resp_len'].isna()) & (df['A55']==1), 'resp_len'] = 1
    
    # The following variables indicate whether leave was cut short for financial issues
    # A23c: unable to afford unpaid leave due to leave taking
    # A53g: cut leave time short to cover lost wages
    # A62a: return to work because cannot afford more leaves
    # B15_1_CAT, B15_2_CAT: can't afford unpaid leave
    df.loc[(df['resp_len'].isna()) & ((df['A23c']==1) |
                               (df['A53g']==1) |
                               (df['A62a']==1) |
                               (df['B15_1_CAT']==5) |
                               (df['B15_2_CAT']==5)), 'resp_len'] = 1
    
    # Assume all takers/needers with ongoing condition are 'cornered' and would respond with longer leaves
    # A10_1, A10_2: regular/ongoing condition, takers and dual
    # B11_1, B11_2: regular/ongoing condition, needers and dual
    df.loc[(df['resp_len'].isna()) & (df['A10_1']==2) | (df['A10_1']==3)
           | (df['B11_1']==2) | (df['B11_1']==3), 'resp_len'] = 1
    
    # B15_1_CAT and B15_2_CAT only has one group (cod = 5) identified as constrained by unaffordability
    # These financially-constrained workers were assigned resp_len=1 above, all other cornered workers would not respond
    # Check reasons of no leave among rest: df[df['resp_len'].isna()].B15_1_CAT.value_counts().sort_index()
    # all reasons unsolved by replacement generosity
    df.loc[(df['resp_len'].isna()) & (df['B15_1_CAT'].notna()), 'resp_len'] = 0
    df.loc[(df['resp_len'].isna()) & (df['B15_2_CAT'].notna()), 'resp_len'] = 0
    
    # Check LEAVE_CAT of rest: df[df['resp_len'].isna()]['LEAVE_CAT'].value_counts().sort_index()
    # 267 takers and 3 needers
    # with no evidence in data of need solvable / unsolvable by $, assume solvable to be conservative
    df.loc[df['resp_len'].isna(), 'resp_len'] = 1

    return df

# -------------
# Function to get weighted mean using v-w columns
# [done - d-pop weighted] Chris, we should discuss whether using means in this way is the correct way. I have no idea.
# [done - answer is No] Also, can this be done in the scikit-learn class?
# -------------

def get_wm_col(df, vws):
    """
    Function to create weighted mean of a variable based on the number of nearest neighbors.
    """

    # Initialize
    num,  denom = pd.Series(np.zeros(len(df))), pd.Series(np.zeros(len(df)))

    # Loop over list of var-weight tuples [('v0','w0'),...]
    for v, w in vws:  
        num += df[v] * df[w]
        denom += df[w]
    
    return(num / denom)


# a function to fillna using knn
def fillna_knn(k, d0, d1, df, ixlabel, cols, v, w, vout):
    '''

    :param k: k in knn
    :param d0: df with v known
    :param d1: df with v missing
    :param df: master df used to fill in missing values in d0[cols] and d1[cols]
    :param ixlabel: label of index of d0, d1, and df for consistent merging
    :param cols: cols used to determine nearest neighbors (should not contain v)
    :param v: label of variable to be filled in d0 and d1
    :param w: label of weight col in d0
    :param vout: label of weighted v col in d0 after fillna
    :return: d0 and d1 appended together, with v filled in
    '''

    locs0, locs1 = d0[cols], d1[cols] # location of workers in pool and workers whose v will be imputed

    # fill in missing values
    for c in cols:
        locs0.loc[locs0[c].isna(), c] = df[c].mean() # Hautahi, I still get SettingWithCopyWarning with .loc used
        locs1.loc[locs1[c].isna(), c] = df[c].mean()
        # run kNN regression
    '''
    # [done] KNeighborsRegressor not working - can't specify weight = [callable], as 'distance' option only gives resulting
    # knn distances but not indices of neighbors, so cannot identify population weight of neighbors and configure
    # a callable as weight

    v0 = pd.DataFrame(d0[v]) # values of v of workers in pool
    neigh = KNeighborsRegressor(n_neighbors=k, weights='distance')
    neigh.fit(locs0, v0)
    v1 = pd.DataFrame(neigh.predict(locs1)) # column of predicted v1
    v1 = v1.set_index(locs1.index) # set index as empid to be consistent as locs1
    v1.columns = ['length']

    # Append locs0, v0, locs1, v1
    locs = locs0.append(locs1)
    vs = v0.append(v1)
    locs[v] = vs
    '''
    nbrs = NearestNeighbors(n_neighbors=k).fit(locs0)
    distances, indices = nbrs.kneighbors(locs1) # indices=order in locs0, not index of d0
        # translate indices from knn to index of d0 for merging
    Nnids = []
    for row in range(len(locs1)):
        nnids = []
        for kk in range(k):
            i = indices[row][kk]
            nnids.append(locs0[i:(i + 1)].index[0])  # these are the k NNs (empid) identified
        Nnids.append(nnids)

        # make a df of nns for empids with missing length
    dict_nn = dict(zip(locs1.index, Nnids))
    dnn = pd.DataFrame.from_dict(dict_nn, orient='index')
    dnn.index.rename(ixlabel, inplace=True)
    dnnCols = []
    for kk in range(k):
        dnnCols.append('nn%s' % kk)
    dnn.columns = dnnCols

        # use ixlabel of nns as key to merge with df[[ixlabel, v, w]]
    d1nn = df.join(dnn, how='right') # wkr dataset with nns, responsive, with length drawn from entire pool
    for kk in range(k):
        dfnn = df[:][[v, w]]
        dfnn['nn%s' % kk] = df.index
        dfnn.columns = ['resp%s' % kk, 'wp%s' % kk, 'nn%s' % kk] # wp = population weight in FMLA
        d1nn = pd.merge(d1nn, dfnn, how='left', on='nn%s' % kk)

        # set up a df of inverse distances, and merge with d1nn
    invDist = pd.DataFrame(1/(distances + 10**(-20))) # avoid 1/0 = inf weight
    invDist = invDist.set_index(d1nn.index)
    invDist.columns = ['invd%s' % kk for kk in range(k)]
    d1nn = d1nn.join(invDist)

        # compute distance-population weights
    for kk in range(k):
        d1nn['w%s' % kk] = d1nn['wp%s' % kk] * d1nn['invd%s' % kk]

        # in resulting d1nn, compute weighted length using nns' empid
    vws = []
    for kk in range(k):
        vws += [('resp%s' % kk, 'w%s' % kk)]

    d1nn[vout] = get_wm_col(d1nn, vws)
    d1nn = d1nn.drop(columns=['nn%s' % kk for kk in range(k)] + ['wp%s' % kk for kk in range(k)] +
                                             [x[0] for x in vws] + [x[1] for x in vws])
                                            # x[0] picks resp<kk>, x[1] picks w<kk>
    return d1nn

# -------------
# Get the counterfactual length of leave that would be taken by those
# who were deemed responsive to the new leave schedule
# -------------

def get_resp_length(df, params): # [done] Chris, want to consider creating pool in this function?
    
    # Unpack parameters
    k, cols, ixlabel, v, w, vout = params

    # For workers predicted to have no response to a leave program (interior solns), set resp_length = length
    wkrs_noResp = df[df['resp_len']==0]
    wkrs_noResp['resp_length'] = wkrs_noResp['length']
    
    # up to here, some interior-soln workers did not report length, they are:
        # (i) Employed only, we set resp_length = 0.
    wkrs_noResp.loc[wkrs_noResp['LEAVE_CAT'] == 3, 'resp_length'] = 0
        # (ii) Needers (not financially-constrained) thus w/o length,
        # we also set resp_length = 0 as need won't be addressed by program generosity
    wkrs_noResp.loc[wkrs_noResp['LEAVE_CAT'] == 2, 'resp_length'] = 0


    # For responsive workers with known length, estimate length from pool of workers with greater length

    # Pool of workers with 'length' not limited by replacement.
    pool = df[(df['resp_len']==0) & (df.length>0)] # pool size = 272 workers

    # Initialize
    rwkrs_0nn, wkrs_rnn_len  = [], pd.DataFrame([])  # list of resp wkrs with no nns, e.g. those with max leave length in entire sample
    
    # Loop over all responsive workers with known length
    resp_wkrs = df[(df['resp_len']==1) & (df['length'].notna())]
    i = -1
    for wid, row in resp_wkrs.iterrows():  # [done] Chris, it's probably quicker and cleaner to use the .iterrows() method here
        i += 1

        # Get length of leave of worker
        l = row['length']
        
        # Take dataframe of individuals where the length of their leave is greater than this worker
        # and who were not forced to take shorter leave
        d0 = pool[pool['length'] > l]
        
        # Check whether there are enough relevant people to run knn
        if len(d0) >=k:

            # use kNN to estimate conditional length
            d1 = pd.DataFrame(row).transpose()
            d1nn = fillna_knn(k, d0, d1, df, ixlabel, cols, v, w, vout)
            print('---- d1nn filled in for empid = %s, row %s-th done' % (wid, i))
            
            wkrs_rnn_len = wkrs_rnn_len.append(d1nn)
        
        else: # if not enough (less than k) leave takers with greater length in poolw
            
            rwkrs_0nn.append(wid)
            print('---->>> empid = %s has large leave length, insufficient eligible neighbors in pool, empid recorded to rwkrs_0nn' % wid)

    # For responsive workers with long length thus 0nn, get response using heuristics
    wkrs_r0nn_len = df.loc[rwkrs_0nn]
    wkrs_r0nn_len['resp_length'] = wkrs_r0nn_len['length'] * 1.5 # 1.5x for those w/o nns (long current length)

    # For responsive workers with missing length, draw length from entire pool
    d0 = pool
    d1 = df[(df['resp_len']==1) & (df['length'].isna())]
    wkrs_rnn_noLen = fillna_knn(k, d0, d1, df, ixlabel, cols, v, w, vout)

    # Append 4 resulting dfs:
    wkrs_noResp['source'] = 'wkrs_noResp'
    wkrs_rnn_len['source'] = 'wkrs_rnn_len'
    wkrs_r0nn_len['source'] = 'wkrs_r0nn_len'
    wkrs_rnn_noLen['source'] = 'wkrs_rnn_noLen'
    ds = [wkrs_noResp, wkrs_rnn_len, wkrs_r0nn_len, wkrs_rnn_noLen]
    dfr = pd.DataFrame([])
    for d in ds:
        dfr = dfr.append(d)

    # Fill in status-quo leave length = 0, will need status-quo leaves for program cost benchmarking
    dfr.loc[dfr['length'].isna(), 'length'] = 0
    return dfr

# -------------
# Function to fetch the parameters
# Chris, we'll eventually change this to either a class or a command line input
# -------------

def get_params():
    
    # Number of nearest neighbors
    k = 2

    cols = ['age','male','employed',
            'wkhours','noHSdegree', 'BAplus',
            'empgov_fed', 'empgov_st', 'empgov_loc',
            'lnfaminc','black', 'asian', 'hisp', 'other',
            'ndep_kid', 'ndep_old',
            'nevermarried', 'partner', 'widowed', 'divorced', 'separated']

    ixlabel = 'empid'
    v =  'length'
    w =  'freq_weight'
    vout = 'resp_length'
    params = (k, cols, ixlabel, v, w, vout)
    return(params)

# -------------
# Get response
# -------------

def get_response(outname='./data/fmla_2012/fmla_clean_2012_resp_length.csv',fname = './data/fmla_2012/fmla_clean_2012.csv'):
    """
    the main function
    """
    # Timer
    t0 = time()

    # Read FMLA data
    df = pd.read_csv(fname)

    # Get data with added 'resp_len' variable
    df = set_data(df)
    
    # Fetch parameters
    params = get_params()

    # resulting df for all FMLA workers with response length generated
    dfr = get_resp_length(df, params)
    
    dfr.to_csv(outname, index=False)
    t1 = time()
    print('Leave response columns generated and resulting FMLA dataset saved. Time elapsed = %s seconds' % (t1-t0))

    return None

# get_response()
