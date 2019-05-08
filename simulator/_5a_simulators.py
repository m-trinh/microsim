'''
All simulators, kNN, Logistic Regression, ...

Chris Zhang 9/24/2018
'''

# -------------
# Housekeeping
# -------------

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from _1a_get_response import get_params, get_wm_col

# -------------
# K-nearest neighbors simulation function
# -------------

def simulate_knn(k, fmla, acs, Xs, vars):
    '''
    This function produces columns of length equal to the acs
    that corresponds to the weighted mean across the k nearest neighbors
    of the variables of interest (vars)
    
    inputs: k: k in knn
            fmla: fmla df with response var column
            acs: acs df
            Xs: cols used for knn
            vars: list of vars of interest to be sourced from fmla
    output: response var column for all acs ppl
    '''
    
    # Fetch columns that will be used to compute distance function in knn
    locs_fmla, locs_acs = fmla[Xs], acs[Xs]
    
    # Replace NA values with the mean from that column
    locs_fmla = locs_fmla.fillna(locs_fmla.mean())
    locs_acs = locs_acs.fillna(locs_acs.mean())
    
    # -------------
    # run kNN
    # -------------
    
    # Create a Nearest Neighbor Class
    nbrs = NearestNeighbors(n_neighbors=k).fit(locs_fmla)

    # distances is an n x k array showing the distances between acs(i) and its closest k fmla neighbors
    # indices is the index number of the closest k fmla neighbors
    distances, indices = nbrs.kneighbors(locs_acs)

    # Number of unique individuals in the fmla that are nearest neighbors
    ns_nn = [len(pd.DataFrame(indices)[x].value_counts()) for x in range(k)]
    print('Number of FMLA workers found as 1st,..., %s-th, nearest neighbors = %s' % (k, ns_nn))
    
    # Create variables of indices in both acs and fmla to use for merging
    for kk in range(k):

        # Create column in acs with index of nearest neighbor from fmla, and associated inverse distances
        acs['idx_nn%s' % kk] = pd.DataFrame(indices)[kk]
        acs['invd%s' % kk] = pd.DataFrame(1/(distances + 10**(-20)))[kk] # avoid 1/0 = inf weight
        # Create column in fmla that is equal to the index (for easy merging)
        fmla['idx_nn%s' % kk] = fmla.index

        ## Use kNN indices to merge with FMLA data
        # Add the frequency weight and relevant columns of the k nearest-neighbors onto the acs
        acs = pd.merge(acs, fmla[['idx_nn%s' % kk, 'freq_weight'] + vars], how='left', on='idx_nn%s' % kk)
        
        # Rename the merged columns
        dcol = {}
        for var in vars:
            dcol[var] = var + '%s' % kk
        dcol['freq_weight'] = 'wp%s' % kk
        acs.rename(columns=dcol, inplace=True)

        # Generate distance-pop weight
        acs['w%s' % kk] = acs['invd%s' % kk] * acs['wp%s' % kk]

        # Replace NA's with zeros
        for var in vars:
            acs['%s%s' % (var, kk)] = acs['%s%s' % (var, kk)].fillna(0)
    
    # -------------
    # Compute outvars to be simulated
    # -------------
        # initiate a dataframe to store columns of newly generated outvars
    vars_for_acs = pd.DataFrame([])
    for var in vars:
        vws = []
        for kk in range(k):
            vws += [('%s%s' % (var, kk), 'w%s' % kk)]

        # Get weighted mean of variable we're trying to impute across the k nearest neighbors
        # and add as new column to dataframe vars_for_acs, name it as var
        # This assumed a fully responsive length - if program is generous enough so that needers will have interior solutions

        vars_for_acs[var] = get_wm_col(acs, vws)


    return(vars_for_acs)
