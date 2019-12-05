'''
make some plots using post-sim ACS data

chris zhang 12/5/2019
'''

import pandas as pd
import matplotlib

## Read in post-sim ACS (MD using RI para, Random Forest, seed=123)
tag = '20191205_141556'
acs = pd.read_csv('./output/output_%s_Main/acs_sim_%s.csv' % (tag, tag))

