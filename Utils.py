class Settings:
    def __init__(self, fmla_file=None, acs_file=None, output_directory=None, detail=None, state=None,
                 simulation_method=None, benefit_effect=False, calibrate=True, clone_factor=1, se_analysis=False,
                 extend=False, fmla_protection_constraint=False, replacement_ratio=0.5, government_employees=True,
                 needers_fully_participate=False, random_seed=False, self_employed=False, state_of_work=False,
                 top_off_rate=0, top_off_min_length=0, weekly_ben_cap=1250, weight_factor=1, eligible_earnings=3000,
                 eligible_weeks=52, eligible_hours=1250, eligible_size=50, max_weeks=None, take_up_rates=None,
                 leave_probability_factors=None, payroll_tax=1, benefits_tax=False, average_state_tax=5,
                 max_taxable_earnings_per_person=100000, total_taxable_earnings=10000000000, fed_employees=True,
                 state_employees=True, local_employees=True):
        self.fmla_file = fmla_file
        self.acs_file = acs_file
        self.output_directory = output_directory
        self.detail = detail
        self.state = state
        self.simulation_method = simulation_method
        self.benefit_effect = benefit_effect
        self.calibrate = calibrate
        self.clone_factor = clone_factor
        self.se_analysis = se_analysis
        self.extend = extend
        self.fmla_protection_constraint = fmla_protection_constraint
        self.replacement_ratio = replacement_ratio
        self.government_employees = government_employees
        self.fed_employees = fed_employees
        self.state_employees = state_employees
        self.local_employees = local_employees
        self.needers_fully_participate = needers_fully_participate
        self.random_seed = random_seed
        self.self_employed = self_employed
        self.state_of_work = state_of_work
        self.top_off_rate = top_off_rate
        self.top_off_min_length = top_off_min_length
        self.weekly_ben_cap = weekly_ben_cap
        self.weight_factor = weight_factor
        self.eligible_earnings = eligible_earnings
        self.eligible_weeks = eligible_weeks
        self.eligible_hours = eligible_hours
        self.eligible_size = eligible_size
        self.payroll_tax = payroll_tax
        self.benefits_tax = benefits_tax
        self.average_state_tax = average_state_tax
        self.max_taxable_earnings_per_person = max_taxable_earnings_per_person
        self.total_taxable_earnings = total_taxable_earnings
        if max_weeks is None:
            self.max_weeks = {'Own Health': 12, 'Maternity': 12, 'New Child': 12, 'Ill Child': 12, 'Ill Spouse': 12,
                              'Ill Parent': 12}
        else:
            self.max_weeks = max_weeks
        if take_up_rates is None:
            self.take_up_rates = {'Own Health': 0.5, 'Maternity': 0.5, 'New Child': 0.5, 'Ill Child': 0.5,
                                  'Ill Spouse': 0.5, 'Ill Parent': 0.5}
        else:
            self.take_up_rates = take_up_rates
        if leave_probability_factors is None:
            self.leave_probability_factors = {'Own Health': 0.667, 'Maternity': 0.667, 'New Child': 0.667,
                                              'Ill Child': 0.667, 'Ill Spouse': 0.667, 'Ill Parent': 0.667}
        else:
            self.leave_probability_factors = leave_probability_factors


# From https://stackoverflow.com/questions/21208376/converting-float-to-dollars-and-cents
def as_currency(amount):
    if amount >= 0:
        return '${:,.2f}'.format(amount)
    else:
        return '-${:,.2f}'.format(-amount)
