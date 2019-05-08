'''
Auxiliary functions called in class SimulationEngine()

Chris Zhang 10/30/2018
'''

import numpy as np


# a function to simulate from wheel of fortune (e.g. simulate leave type from discrete distribution of 6 types)
def simulate_wof(ps):
    '''

    :param ps: a list of discrete probabilities, must sum up to 1
    :return: the index of group where the draw indicates
    '''
    ym = np.random.multinomial(1, ps, 1)
    ym = list(ym[0])
    ix = ym.index(1)

    return ix

# a function to get marginal probability vector, i.e. change values of probability list to 0 with given indices, and normalize
def get_marginal_probs(ps, ixs):
    for ix in ixs:
        ps[ix] = 0
    return [p/sum(ps) for p in ps]

# a function to re-calculate probability vector by applying strict lower bound
# (for computing ps for simulating counterfactual length)
def get_dps_lowerBounded(dps, b):
    '''

    :param dps: dict of possible values to associated prob
    :param b: strict lower bound = any real value
    :return: dpsb: dict of values/probs after applying lower bound b to values
    '''
    ksb, vsb = [], []
    for k, v in dps.items():
        if k > b:
            ksb.append(k)
            vsb.append(v)
    if len(ksb)>0:
        vsb = np.array(vsb)
        vsb = vsb/vsb.sum()
        dpsb = dict(zip(ksb, vsb))
    else:
        kmax = max(dps.keys())
        dpsb = {kmax*1.25 : 1}
    return dpsb