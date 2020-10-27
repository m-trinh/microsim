'''
Code to reconcile results in Issue Brief 6:
A Guide to Perform Policy Simulation of Parental Leave for Federal Civilian Employees

chris zhang 10/21/2020
'''
import pandas as pd

## Read in post-sim ACS
fp_acs = './output/output_20201020_173557_main simulation/acs_sim_all_20201020_173557.csv'
acs = pd.read_csv(fp_acs)

## check total outlay to program participants, validate against outlay estimate in program_cost CSV
outlay = 0
outlay += (acs['annual_benefit_matdis']*acs['PWGTP']).sum()
outlay += (acs['annual_benefit_bond']*acs['PWGTP']).sum()
print('Outlay estimate before adjustment = %s million' % round(outlay/10**6, 1))
## set cap for total CPL
cap = 12*5
## get share of program participants with total CPL > 12 weeks
# fill in na for CPLs if any
acs['cpl_matdis'] = acs['cpl_matdis'].fillna(0)
acs['cpl_bond'] = acs['cpl_bond'].fillna(0)
# get num and denom for the share
num = acs[(acs['takeup_any']==1) & (acs['cpl_matdis'] + acs['cpl_bond'] > cap)]['PWGTP'].sum()
denom = acs[acs['takeup_any']==1]['PWGTP'].sum()
print('Share of program participants with total CPL > 12 weeks = %s' % "{:.2%}".format(num/denom))

## cap total CPL at 12 weeks
acs['cpl_all'] = acs['cpl_matdis'] + acs['cpl_bond']
acs['cpl_all_adj'] = [min(x, cap) for x in acs['cpl_all']]

## allocate capped CPL to leave reasons
acs['cpl_matdis_adj'] = acs['cpl_all_adj'] * acs['cpl_matdis'] / acs['cpl_all']
acs['cpl_matdis_adj'].fillna(0)
acs['cpl_matdis_adj'] = [round(x, 0) for x in acs['cpl_matdis_adj']]
acs['cpl_bond_adj'] = acs['cpl_all_adj'] - acs['cpl_matdis_adj']

## get adjusted outlay
outlay_adj = 0
outlay_adj_matdis = (acs['annual_benefit_matdis']*acs['cpl_matdis_adj']/acs['cpl_matdis']*acs['PWGTP']).sum()
outlay_adj_bond = (acs['annual_benefit_bond']*acs['cpl_bond_adj']/acs['cpl_bond']*acs['PWGTP']).sum()
outlay_adj += outlay_adj_matdis
outlay_adj += outlay_adj_bond
print('Outlay estimate after adjustment for joint cap [matdis]= %s million' % round(outlay_adj_matdis/10**6, 1))
print('Outlay estimate after adjustment for joint cap[bond]= %s million' % round(outlay_adj_bond/10**6, 1))
print('Outlay estimate after adjustment for joint cap[total]= %s million' % round(outlay_adj/10**6, 1))

## Adjust for gov worker population estimate
# CBO estimate at Congressional Research Service (2019). Federal Workforce Statistics Sources: OPM and OMB.
# CBO estimate = 2.1 million
# model estimate = 3.36 million
pop_adj_factor = 2.1/3.36
outlay_adj *= pop_adj_factor
outlay_adj_matdis *= pop_adj_factor
outlay_adj_bond *= pop_adj_factor
print('Outlay estimate after adjustment for joint cap and pop est [matdis]= %s million' % round(outlay_adj_matdis/10**6, 1))
print('Outlay estimate after adjustment for joint cap and pop est [bond]= %s million' % round(outlay_adj_bond/10**6, 1))
print('Outlay estimate after adjustment for joint cap and pop est [total]= %s million' % round(outlay_adj/10**6, 1))
