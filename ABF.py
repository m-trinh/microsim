import pandas as pd
import numpy as np


class ABF:
    def __init__(self, acs, settings, benefits):
        self.df = acs
        self.update_settings(settings)

        self.benefits = benefits

    def update_settings(self, settings):
        self._update_settings(settings)

    def _update_settings(self, settings):
        # self.state_of_work = settings.state_of_work
        # self.state = settings.state
        self.self_employed = settings.self_employed
        self.eligible_size = settings.eligible_size
        self.fed_employees = settings.fed_employees
        self.state_employees = settings.state_employees
        self.local_employees = settings.local_employees
        self.max_taxable_earnings_per_person = settings.max_taxable_earnings_per_person
        self.benefits_tax = settings.benefits_tax
        self.average_state_tax = settings.average_state_tax / 100
        self.payroll_tax = settings.payroll_tax / 100

    # FUNCTION #1: Drop unneeded observations per the Paid Family Leave Policy Parameters
    def abf_data(self):
        # Restriction #1: Exclude those who are not in the formal economy.
        # Drop COW=8 (Working without pay in family business or farm) & COW=9 (Unemployed and last worked 5 years ago
        # or earlier or never)
        # if ((self.df['COW'] == 8) | (self.df['COW'] == 9)).any():
        #
        #     index_names = self.df[(self.df['COW'] == 8) | (self.df['COW'] == 9)].index
        #     dropped_unemp = (len(index_names))
        #     message_uemp = "We dropped %s unemployed workers from the dataset" % dropped_unemp
        #     print(message_uemp)
        #     self.df.drop(index_names, inplace=True)
        #
        # else:
        #     print('Unemp workers are not in the dataset')

        # Restriction #2: Keep only selected State from either the "State or Work" or "State of Residence" ACS file
        # if self.state_of_work:  # Use "ACS State of Work" File
        #
        #     if self.state is not None:  # Need to test if we get a Nationwide estimate if geography=True or None
        #
        #         index_names = self.df[self.df['POWSP'] != self.state].index
        #         dropped_geog = (len(index_names))
        #         message_geog = "We dropped %s workers from the dataset, who live outside the geographical bounds" % \
        #                        dropped_geog
        #         print(message_geog)
        #         self.df.drop(index_names, inplace=True)
        #     else:
        #         print('State is null')
        #
        # else:  # Use "ACS State of Residence" File
        #
        #     if self.state is not None:  # Need to test if we get a nationwide estimate if geography=True or None
        #         old_length = self.df.shape[0]
        #         self.df = self.df[self.df['ST'] == self.state]
        #         dropped_geog = old_length - self.df.shape[0]
        #         # index_names = self.df[self.df['ST'] != self.state].index
        #         # dropped_geog = (len(index_names))
        #         # self.df.drop(index_names, inplace=True)
        #         message_geog = "We dropped %s workers from the dataset, who live outside the geographical bounds" % \
        #                        dropped_geog
        #         print(message_geog)
        #     else:
        #         print('State is null')

        # Restriction #3: Exclude employers smaller than the minimum employer size parameter
        # if self.eligible_size is not None:  # None = Null Value
        #
        #     self.df['emp_size'] = np.random.randint(1, 2000, self.df.shape[0])
        #     emp_size_max_id = self.df['emp_size'].idxmax()
        #     emp_size_max_val = (self.df.loc[[emp_size_max_id], ['emp_size']]).values[0]
        #
        #     if emp_size_max_val > self.eligible_size:
        #         # Need to confirm that the emp_size variable constructed on the microsim side wouldn't have any blanks
        #         index_names = self.df[self.df['emp_size'] < self.eligible_size].index
        #         type(index_names)
        #         dropped_empsize = (len(index_names))
        #         message_empsize = "We dropped %s workers from the dataset, at firms smaller than minsize" % \
        #                           dropped_empsize
        #         print(message_empsize)
        #
        #         self.df.drop(index_names, inplace=True)
        #
        #     else:
        #         print('All employers are larger than the min size')
        # else:
        #     print('There is no Minimum Emp Size restriction')

        # Restriction #4: Exclude those who are Self-Employed (drop self-employed: COW=6 & COW=7)
        # if self.self_employed is not None:
        #
        #     if not self.self_employed:
        #
        #         if ((self.df['COW'] == 6) | (self.df['COW'] == 7)).any():
        #             index_names = self.df[(self.df['COW'] == 6) | (self.df['COW'] == 7)].index
        #             dropped_selfemp = (len(index_names))
        #             message_selfemp = "We dropped %s self-employed workers from the dataset" % dropped_selfemp
        #             print(message_selfemp)
        #             self.df.drop(index_names, inplace=True)
        #
        #         else:
        #             print('Self Emp workers are not in the dataset')
        #     else:
        #         print('Self-employed workers recieve PFML')

        # Restriction #5: Exclude federal gov workers, if policy applies (drop COW=5)
        # if self.fed_employees is not None:  # TRUE (Constraint is applied)
        #
        #     if not self.fed_employees:
        #
        #         if (self.df['COW'] == 5).any():
        #             index_names = self.df[(self.df['COW'] == 5)].index
        #             dropped_fed = (len(index_names))
        #             message_fed = "We dropped %s federal employees from the dataset" % dropped_fed
        #             print(message_fed)
        #             self.df.drop(index_names, inplace=True)
        #         else:
        #             print('Fed workers are not in the dataset')
        #     else:
        #         print('Fed employees recieve PFML')
        #
        # # Restriction #6: Exclude state govs workers, if policy applies (drop COW=4)
        # if self.state_employees is not None:  # TRUE (Constraint is applied)
        #
        #     if not self.state_employees:
        #
        #         if (self.df['COW'] == 4).any():
        #             index_names = self.df[(self.df['COW'] == 4)].index
        #             dropped_state = (len(index_names))
        #             message_state = "We dropped %s state employees from the dataset" % dropped_state
        #             print(message_state)
        #
        #             self.df.drop(index_names, inplace=True)
        #         else:
        #             print('State workers are not in the dataset')
        #     else:
        #         print('State employees recieve PFML')
        #
        # # Restriction #7: Drop local gov workers, if policy applies (drop COW=3)
        # if self.local_employees is not None:  # TRUE (Constraint is applied)
        #
        #     if not self.local_employees:
        #
        #         if (self.df['COW'] == 3).any():
        #             index_names = self.df[(self.df['COW'] == 3)].index
        #             dropped_local = (len(index_names))
        #             message_local = "We dropped %s local employees from the dataset" % dropped_local
        #             print(message_local)
        #             self.df.drop(index_names, inplace=True)
        #         else:
        #             print('Local workers are not in the dataset')
        #     else:
        #         print('Local employees recieve PFML')

        # Apply Taxable Wage Max
        if self.max_taxable_earnings_per_person is not None:  # TRUE (Constraint is applied)
            self.df['taxable_income_capped'] = np.where((self.df['wage12'] > self.max_taxable_earnings_per_person),
                                          self.max_taxable_earnings_per_person, self.df['wage12'])
            index_names = self.df[self.df['wage12'] > self.max_taxable_earnings_per_person].index
            censor = len(index_names)
            message_censor = "We censored %s observations to the wage max" % censor
            print(message_censor)
        else:
            self.df['taxable_income_capped'] = self.df['income']

    # FUNCTION #2: Conduct Final Calculations on the slimmer ABF Output dataset
    def abf_calcs(self):
        # Step 1 - Calculate Point Estimates
        # Income
        # Intermediate output: unweighted income base (full geographic area)
        total_income = self.df['taxable_income_capped'].sum()

        # Total Weighted Income Base (full geographic area)
        self.df['income_w'] = self.df['taxable_income_capped'] * self.df['PWGTP']
        total_income_w = self.df['income_w'].sum()
        print('Output: Weighted Income Base for Full Geographic Area:')
        print(total_income_w)

        # Tax Revenue
        # Unweighted tax revenue collected (full geographic area)
        self.df['ptax_rev_final'] = self.df['taxable_income_capped'] * self.payroll_tax

        # Total Weighted Tax Revenue (full geographic area)
        self.df['ptax_rev_w'] = self.df['income_w'] * self.payroll_tax
        total_ptax_rev_w = self.payroll_tax * total_income_w
        print('Output: Weighted Tax Revenue for Full Geographic Area:')
        print(total_ptax_rev_w)

        message = "The weighted estimated tax revenue is %s based on a payroll tax rate of %s and a income " \
                  "base of %s " % (total_ptax_rev_w, self.payroll_tax, total_income_w)
        print(message)

        # State Tax Revenue Recouped from Taxed Benefits
        if self.benefits_tax:
            recoup_tax_rev = self.average_state_tax * self.benefits
            print('Output: "State Tax Revenue Recouped from Taxed Benefits:')
            print(recoup_tax_rev)
            message = "With a state tax rate of %s and a benefit outlay of %s, the estimated state tax revenue is %s" \
                      % (self.average_state_tax, self.benefits, recoup_tax_rev)
            print(message)
        else:
            recoup_tax_rev = 0

        # Step 2 - Calculate Standard Errors with 80 Replicate Weights
        # replication weight cols
        reps = ['PWGTP' + str(i) for i in range(1, 81)]

        # Income
        income_r = []
        for wt in reps:
            income_r.append(((self.df['taxable_income_capped'] * self.df[wt]).sum()))

        # print('80 Replicate Income:')
        # print(income_r)

        income_delta = []
        for i in income_r:
            income_delta.append((i - total_income_w) ** 2)

        income_se = ((sum(income_delta)) * (4 / 80)) ** .5

        # Tax Revenue
        tax_r = []
        for wt in reps:
            tax_r.append(((self.df['ptax_rev_final'] * self.df[wt]).sum()))

        # print('80 Replicate Tax Revenue:')
        # print(tax_r)

        tax_delta = []
        for i in tax_r:
            tax_delta.append((i - total_ptax_rev_w) ** 2)

        tax_se = ((sum(tax_delta)) * (4 / 80)) ** .5

        ###########Step 3: Calculate Confidence Intervals

        # Income
        total_income_w_uci = total_income_w + 1.96 * income_se
        print('Total Income Upper CI:')
        print(total_income_w_uci)
        total_income_w_lci = total_income_w - 1.96 * income_se
        print('Total Income Low CI:')
        print(total_income_w_lci)

        #Tax Revenue
        total_ptax_w_uci = total_ptax_rev_w + 1.96 * tax_se
        print('Total Income Upper CI:')
        print(total_ptax_w_uci)
        total_ptax_w_lci = total_ptax_rev_w - 1.96 * tax_se
        print('Total Income Low CI:')
        print(total_ptax_w_lci)

        # Return Dictionary with Final Output Values
        abf_output = {'Total Income (Weighted)': total_income_w, 'Total Income': total_income,
                      'Income Standard Error': income_se, 'Total Income Upper Confidence Interval': total_income_w_uci,
                      'Total Income Lower Confidence Interval': total_income_w_lci,
                      'Total Tax Revenue (Weighted)': total_ptax_rev_w,
                      'Tax Revenue Standard Error': tax_se,
                      'Total Tax Revenue Upper Confidence Interval': total_ptax_w_uci,
                      'Total Tax Revenue Lower Confidence Interval': total_ptax_w_lci,
                      'Tax Revenue Recouped from Benefits': recoup_tax_rev}
        print(abf_output)

        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        revenue_by_class = pd.pivot_table(self.df, index=["class"], values=["income_w", "ptax_rev_w"], aggfunc=[np.sum])
        revenue_by_age = pd.pivot_table(self.df, index=["age_cat"], values=["income_w", "ptax_rev_w"], aggfunc=[np.sum])
        revenue_by_gender = pd.pivot_table(self.df, index=["GENDER_CAT"], values=["income_w", "ptax_rev_w"],
                                           aggfunc=[np.sum])

        pivot_tables = {'Class of Worker': revenue_by_class, 'Age': revenue_by_age,
                        'Gender': revenue_by_gender}

        return abf_output, pivot_tables

    def run(self):
        # Create Class variable to aggregate the COW variable for display purposes
        self.df['class'] = ''
        cleanup = {1: "Private", 2: "Private", 3: "Local Govt.", 4: "State Govt.", 5: "Federal Govt.",
                   6: "Self-Employed", 7: "Self-Employed", 8: "Other", 9: "Other"}
        for i in self.df.index.values:
            if not pd.isnull(self.df.at[i, 'COW']):
                self.df.at[i, 'class'] = cleanup[int(float(self.df.at[i, 'COW']))]

        # Create Age Categories for display purposes (need to find out variable specification on the microsim side)
        age_ranges = ["[{0} - {1})".format(AGEP, AGEP + 10) for AGEP in range(0, 100, 10)]
        self.df['age_cat'] = pd.cut(x=self.df['age'], bins=list(range(0, 110, 10)), labels=age_ranges, right=False)

        # Create Gender Categories for display pruposes
        self.df['GENDER_CAT'] = np.where(self.df['male'] == 1, 'male', 'female')
        self.df['GENDER_CAT'] = np.where(np.isnan(self.df['male']), np.nan, self.df['GENDER_CAT'])  # code missing responses as missing

        self.abf_data()

        # for chunk in pd.read_csv(self.acs_file, chunksize=100000, low_memory=False):
        #     df = pd.concat([df2, chunk])
        #
        #     # Select variables for analysis
        #     d = df.loc[:, acs]
        #     d['income'] = d['PERNP']
        #
        #     # Create Class variable to aggregate the COW variable for display purposes
        #     d['class'] = d['COW']
        #     cleanup = {"class": {1: "Private", 2: "Private", 3: "Local", 4: "State", 5: "Federal", 6: "Self-Employed",
        #                          7: "Self-Employed", 8: "Other", 9: "Other"}}
        #     d.replace(cleanup, inplace=True)
        #
        #     # Create Age Categories for display purposes (need to find out variable specification on the microsim side)
        #     age_ranges = ["[{0} - {1})".format(AGEP, AGEP + 10) for AGEP in range(0, 100, 10)]
        #     count_unique_age_ranges = len(age_ranges)
        #     d['age_cat'] = pd.cut(x=d['AGEP'], bins=count_unique_age_ranges, labels=age_ranges)
        #
        #     # Create Gender Categories for display pruposes
        #     d['GENDER_CAT'] = np.where(d['SEX'] == 1, 'male', 'female')
        #     d['GENDER_CAT'] = np.where(np.isnan(d['SEX']), np.nan, d['GENDER_CAT'])  # code missing responses as missing
        #
        #     ######################
        #     # FUNCTION #1= abf_data: Drop unneeded observations per the Paid Family Leave Policy Parameters
        #
        #     # CA Policy Parmeters
        #     # abf_data(stateofwork=True, geography=6, selfemp=False, minsize=None, fed_gov=False, state_gov=False, local_gov=False, wagemax=106742, benefits_taxed=True, statetax_amt=.15, benefits=5000000, paytax_rate=0.009)
        #
        #     # New Jersey Policy Parmeters
        #     # abf_data(stateofwork=True, geography=34, selfemp=False, minsize=None, fed_gov=False, state_gov=True, local_gov=True, wagemax=32600, benefits_taxed=True, benefits_taxed=.15, benefits=5000000, paytax_rate=0.0008)
        #     # abf_data(stateofwork=True, geography=34, selfemp=False, minsize=None, fed_gov=False, state_gov=False, local_gov=False, wagemax=32600, benefits_taxed=True, statetax_amt=.15, benefits=5000000, paytax_rate=0.007)
        #
        #     # Rhode Island Policy Parmeters
        #     # self.abf_data(stateofwork=True, geography=44, selfemp=False, minsize=None, fed_gov=False, state_gov=False,
        #     #          local_gov=False, wagemax=66300, benefits_taxed=True, statetax_amt=.15, benefits=175000000,
        #     #          paytax_rate=0.012)
        #     self.abf_data(d, reps)

        # FUNCTION #2=abf_output: Conduct Final Calculations on the slimmer ABF Output dataset
        # CA Policy Parmeters
        # abf_output=abf_calcs(stateofwork=True, geography=6, selfemp=False, minsize=None, fed_gov=False, state_gov=False, local_gov=False, wagemax=106742, benefits_taxed=True, statetax_amt=.15, benefits=5000000, paytax_rate=0.009)

        # NJ Policy Parmeters
        # abf_output=abf_calcs(stateofwork=True, geography=34, selfemp=False, minsize=None, fed_gov=False, state_gov=True, local_gov=True, wagemax=32600, benefits_taxed=True, benefits_taxed=.15, benefits=5000000, paytax_rate=0.0008)
        # abf_output=abf_calcs(stateofwork=True, geography=34, selfemp=False, minsize=None, fed_gov=False, state_gov=False, local_gov=False, wagemax=32600, benefits_taxed=True, statetax_amt=.15, benefits=5000000, paytax_rate=0.007)

        # Rhode Island Policy Parmeters
        # abf_output = abf_calcs(stateofwork=True, geography=44, selfemp=False, minsize=None, fed_gov=False,
        #                        state_gov=False, local_gov=False, wagemax=66300, benefits_taxed=True, statetax_amt=.15,
        #                        benefits=175000000, paytax_rate=0.012)
        return self.abf_calcs()

    def rerun(self, settings):
        self.update_settings(settings)
        self.abf_data()
        return self.abf_calcs()
