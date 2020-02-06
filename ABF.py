import pandas as pd
import numpy as np


class ABF:
    def __init__(self, acs_file, benefits, eligible_size, max_taxable_earnings_per_person, benefits_tax,
                 average_state_tax, payroll_tax):
        self.reps = ['PWGTP' + str(i) for i in range(1, 81)]
        self.df = pd.read_csv(acs_file, usecols=['COW', 'POWSP', 'ST', 'wage12', 'PWGTP', 'age', 'male'] + self.reps)
        self.eligible_size = eligible_size
        self.max_taxable_earnings_per_person = max_taxable_earnings_per_person
        self.benefits_tax = benefits_tax
        self.average_state_tax = average_state_tax / 100
        self.payroll_tax = payroll_tax / 100
        self.benefits = benefits
        self.clean_data()

    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.average_state_tax = self.average_state_tax / 100
        self.payroll_tax = self.payroll_tax / 100

    # FUNCTION #1: Drop unneeded observations per the Paid Family Leave Policy Parameters
    def abf_data(self):
        # Apply Taxable Wage Max
        if self.max_taxable_earnings_per_person is not None:  # TRUE (Constraint is applied)
            self.df['taxable_income_capped'] = np.where((self.df['wage12'] > self.max_taxable_earnings_per_person),
                                                         self.max_taxable_earnings_per_person, self.df['wage12'])
            index_names = self.df[self.df['wage12'] > self.max_taxable_earnings_per_person].index
            censor = len(index_names)
            message_censor = "We censored %s observations to the wage max" % censor
            print(message_censor)
        else:
            self.df['taxable_income_capped'] = self.df['wage12']

    # FUNCTION #2: Conduct Final Calculations on the slimmer ABF Output dataset
    def abf_calcs(self):
        # Step 1 - Calculate Point Estimates
        # Income
        # Intermediate output: unweighted income base (full geographic area)
        total_income = self.df['taxable_income_capped'].sum()

        # Total Weighted Income Base (full geographic area)
        self.df['income_w'] = self.df['taxable_income_capped'] * self.df['PWGTP']
        wage_bins = list(range(0, 210000, 25000))
        wage_ranges = ['[{}k - {}k)'.format(wage_bins[i] // 1000, wage_bins[i] // 1000 + 25)
                       for i in range(len(wage_bins) - 1)]
        self.df['wage_cat'] = pd.cut(x=self.df['income_w'], bins=wage_bins, labels=wage_ranges,
                                     right=False)
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
        # Income
        income_r = []
        for wt in self.reps:
            income_r.append(((self.df['taxable_income_capped'] * self.df[wt]).sum()))

        # print('80 Replicate Income:')
        # print(income_r)

        income_delta = []
        for i in income_r:
            income_delta.append((i - total_income_w) ** 2)

        income_se = ((sum(income_delta)) * (4 / 80)) ** .5

        # Tax Revenue
        tax_r = []
        for wt in self.reps:
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
        revenue_by_wage = pd.pivot_table(self.df, index=["wage_cat"], values=["income_w", "ptax_rev_w"],
                                         aggfunc=[np.sum])

        pivot_tables = {'Class of Worker': revenue_by_class, 'Age': revenue_by_age,
                        'Gender': revenue_by_gender, 'Wage': revenue_by_wage}

        return abf_output, pivot_tables

    def clean_data(self):
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
        # Code missing responses as missing
        self.df['GENDER_CAT'] = np.where(np.isnan(self.df['male']), np.nan, self.df['GENDER_CAT'])

    def run(self):
        # TODO: Chunk this to prevent memory overflow
        # Create Class variable to aggregate the COW variable for display purposes
        self.abf_data()
        return self.abf_calcs()

    def rerun(self, variables):
        self.update_parameters(**variables)
        self.abf_data()
        return self.abf_calcs()
