'''
Calibrate microsimulation model parameters

1. Calibrate a parameter that captures link between replacement rate and average leave length, use CA data

Chris Zhang 10/3/2018
'''

from _5_simulation_engine import SimulationEngine
import numpy as np

# Set up
types = ['own', 'matdis', 'bond', 'illchild', 'illspouse','illparent']
st = 'ca'
fp_acs = 'C:/workfiles/Microsimulation/git/large_data_files/ACS_cleaned_forsimulation_%s.csv' % st
fp_fmla_cf = './data/fmla_2012/fmla_clean_2012_resp_length.csv'
fp_out = './output'
clf_name = 'Support Vector Machine'
rr1 = 0.25 # set at any value in calibration, only relevant if counterfactual=True in se.get_cost()
hrs = 1250
d_max_wk = dict(zip(types, 6*np.ones(6))) # set as max week in CA
max_wk_replace = 1216 # set as CA weekly benefit cap
se = SimulationEngine(st, fp_acs, fp_fmla_cf, fp_out, clf_name, rr1, hrs, d_max_wk, max_wk_replace)
acs = se.get_acs_simulated()

takeups = dict(zip(types, 0.4*np.ones(6))) # set as 40% for now per matdis takeup in CA per
                                         # http://poverty.ucdavis.edu/sites/main/files/file-attachments/cpr-pihl_basso_pfl_brief.pdf
                                         # set it low to be conservative, so weaken the role of takeup on cost
                                         # strengthen the role of other factors (leave taking, wage, etc.)
# Get calibration factor
if st=='ca':
    # Status-quo estimated cost
    cost0 = se.get_cost(acs,takeups,1,counterfactual=False)['Total'] # to get non-calibrated cost0, set alpha = 1
    # Status-quo empirical cost, CA
    cost0_emp = 5513 + 864  # CA FY17-18 total DI + PFL,
                      # https://www.edd.ca.gov/about_edd/pdf/qspfl_PFL_Program_Statistics.pdf
                      # https: // www.edd.ca.gov / about_edd / pdf / qsdi_DI_Program_Statistics.pdf
    # calibration factor
    alpha = round(cost0_emp / cost0, 3) # calibration factor to account for adjustment in eligibility, employer payment, take-up, etc.
    print('Calibration factor alpha = %s needs to be multiplied to program cost estimate based on CA PFL' % alpha)


##### compute example counterfactual program costs
# TCs = se.get_cost(acs, takeups)
# print(TCs)

'''
Simulation method: Logistic Regression
State = ca, Replacement rate = 0.55, Status = Status-quo, Total Program Cost = $735.4 million
Simulation output saved to ./output
Calibration factor alpha = 8.671 needs to be multiplied to program cost estimate based on CA PFL

Simulation method: Ridge Classifier
State = ca, Replacement rate = 0.55, Status = Status-quo, Total Program Cost = $4170.3 million
Simulation output saved to ./output
Time elapsed for computing program costs = 311 seconds
Calibration factor alpha = 1.529 needs to be multiplied to program cost estimate based on CA PFL

Simulation method: Naive Bayes
State = ca, Replacement rate = 0.55, Status = Status-quo, Total Program Cost = $1092.6 million
Simulation output saved to ./output
Time elapsed for computing program costs = 234 seconds
Calibration factor alpha = 5.837 needs to be multiplied to program cost estimate based on CA PFL


Simulation method: Random Forest
State = ca, Replacement rate = 0.55, Status = Status-quo, Total Program Cost = $1730.2 million
Simulation output saved to ./output
Time elapsed for computing program costs = 295 seconds
Calibration factor alpha = 3.686 needs to be multiplied to program cost estimate based on CA PFL

K Nearest Neighbor
State = ca, Replacement rate = 0.55, Status = Status-quo, Total Program Cost = $1936.4 million
Simulation output saved to ./output
Time elapsed for computing program costs = 292 seconds
Calibration factor alpha = 3.293 needs to be multiplied to program cost estimate based on CA PFL

Simulation method: Support Vector Machine
State = ca, Replacement rate = 0.55, Status = Status-quo, Total Program Cost = $2566.5 million
Simulation output saved to ./output
Time elapsed for computing program costs = 305 seconds
Calibration factor alpha = 2.485 needs to be multiplied to program cost estimate based on CA PFL
'''