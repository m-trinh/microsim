import copy
from tkinter import E, W


DEFAULT_STATE_PARAMS = {
    '': {},
    'CA': {
        'replacement_ratio': 0.55,
        'benefit_effect': True,
        'top_off_rate': 0.01,
        'top_off_min_length': 10,
        'max_weeks': {'Own Health': 52,
                      'Maternity': 52,
                      'New Child': 6,
                      'Ill Child': 6,
                      'Ill Spouse': 6,
                      'Ill Parent': 6},
        'weekly_ben_cap': 1216,
        'fmla_protection_constraint': True,
        'eligible_earnings': 300,
        'government_employees': False,
        'fed_employees': False,
        'state_employees': False,
        'local_employees': False,
        'self_employed': False,
        # dependent allowance not implemented
        # take_up_rates = {'Own Health': 0.25, 'Maternity': 0.25, 'New Child': 0.25, 'Ill Child': 0.25,
        #                           'Ill Spouse': 0.25, 'Ill Parent': 0.25}
        # waiting period not implemented
        # extend = True
        # extend days and proportion not implemented
    },
    'NJ': {
        'replacement_ratio': 0.66,
        'benefit_effect': True,
        'top_off_rate': 0.01,
        'top_off_min_length': 10,
        'max_weeks': {'Own Health': 26,
                      'Maternity': 26,
                      'New Child': 6,
                      'Ill Child': 6,
                      'Ill Spouse': 6,
                      'Ill Parent': 6},
        'weekly_ben_cap': 594,
        'fmla_protection_constraint': True,
        'eligible_earnings': 8400,
        'government_employees': False,
        'fed_employees': False,
        'state_employees': False,
        'local_employees': False,
        'self_employed': False,
    },
    'RI': {
        'replacement_ratio': 0.6,
        'benefit_effect': True,
        'top_off_rate': 0.01,
        'top_off_min_length': 10,
        'max_weeks': {'Own Health': 30,
                      'Maternity': 30,
                      'New Child': 4,
                      'Ill Child': 4,
                      'Ill Spouse': 4,
                      'Ill Parent': 4},
        # weekly benefit cap proportion not implemented
        'fmla_protection_constraint': True,
        'eligible_earnings': 11520,
        'government_employees': False,
        'fed_employees': False,
        'state_employees': False,
        'local_employees': False,
        'self_employed': False,
    }
}


LEAVE_TYPES = ['Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent']


class Settings:
    def __init__(self, fmla_file=None, acs_directory=None, output_directory=None, detail=None, state=None,
                 simulation_method=None, benefit_effect=False, calibrate=True, clone_factor=1, se_analysis=False,
                 extend=False, fmla_protection_constraint=False, replacement_ratio=0.5, government_employees=True,
                 needers_fully_participate=False, random_seed=None, self_employed=False, state_of_work=True,
                 top_off_rate=0, top_off_min_length=0, weekly_ben_cap=99999999, weight_factor=1,
                 eligible_earnings=11520, eligible_weeks=1, eligible_hours=1, eligible_size=1, max_weeks=None,
                 take_up_rates=None, leave_probability_factors=None, payroll_tax=1, benefits_tax=False,
                 average_state_tax=5, max_taxable_earnings_per_person=100000, total_taxable_earnings=10000000000,
                 fed_employees=True, state_employees=True, local_employees=True, counterfactual='', policy_sim=False,
                 existing_program='', dual_receivers_share=0.6):
        self.fmla_file = fmla_file
        self.acs_directory = acs_directory
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
        self.counterfactual = counterfactual
        self.policy_sim = policy_sim
        self.existing_program = existing_program
        self.dual_receivers_share = dual_receivers_share
        if max_weeks is None:
            self.max_weeks = {'Own Health': 30, 'Maternity': 30, 'New Child': 4, 'Ill Child': 4, 'Ill Spouse': 4,
                              'Ill Parent': 4}
        else:
            self.max_weeks = max_weeks
        if take_up_rates is None:
            self.take_up_rates = {'Own Health': 0.25, 'Maternity': 0.25, 'New Child': 0.25, 'Ill Child': 0.25,
                                  'Ill Spouse': 0.25, 'Ill Parent': 0.25}
        else:
            self.take_up_rates = take_up_rates
        if leave_probability_factors is None:
            self.leave_probability_factors = {'Own Health': 0.667, 'Maternity': 0.667, 'New Child': 0.667,
                                              'Ill Child': 0.667, 'Ill Spouse': 0.667, 'Ill Parent': 0.667}
        else:
            self.leave_probability_factors = leave_probability_factors

    def copy(self):
        # return Settings(fmla_file=self.fmla_file, acs_directory=self.acs_directory, output_directory=self.output_directory, detail=self.detail, state=self.state,
        #          simulation_method=self.simulation_method, benefit_effect=self.benefit_effect, calibrate=self.calibrate, clone_factor=self.clone_factor, se_analysis=self.se_analysis,
        #          extend=self.extend, fmla_protection_constraint=self.fmla_protection_constraint, replacement_ratio=self.replacement_ratio, government_employees=self.government_employees,
        #          needers_fully_participate=self.needers_fully_participate, random_seed=self.random_seed, self_employed=self.self_employed, state_of_work=self.state_of_work,
        #          top_off_rate=0, top_off_min_length=0, weekly_ben_cap=99999999, weight_factor=1,
        #          eligible_earnings=11520, eligible_weeks=1, eligible_hours=1, eligible_size=1, max_weeks=None,
        #          take_up_rates=None, leave_probability_factors=None, payroll_tax=1, benefits_tax=False,
        #          average_state_tax=5, max_taxable_earnings_per_person=100000, total_taxable_earnings=10000000000,
        #          fed_employees=True, state_employees=True, local_employees=True)
        return copy.deepcopy(self)

    def update_variables(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# From https://stackoverflow.com/questions/21208376/converting-float-to-dollars-and-cents
def as_currency(amount):
    if amount >= 0:
        return '${:,.2f}'.format(amount)
    else:
        return '-${:,.2f}'.format(-amount)


def generate_default_state_params(settings, state='CA'):
    state_params = settings.copy()
    state_params.update_variables(**DEFAULT_STATE_PARAMS[state.upper()])
    return state_params


def generate_generous_params(settings):
    generous_params = settings.copy()
    generous_params.eligible_earnings = 0
    generous_params.eligible_size = 0
    generous_params.eligible_hours = 0
    generous_params.eligible_weeks = 0
    generous_params.replacement_ratio = 1
    generous_params.fed_employees = True
    generous_params.state_employees = True
    generous_params.local_employees = True
    generous_params.self_employed = True

    return generous_params


def format_chart(fig, ax, title, bg_color='#333333', fg_color='white'):
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_title(title, fontsize=10, color=fg_color)
    ax.tick_params(axis='x', labelsize=8, colors=fg_color)
    ax.tick_params(axis='y', labelsize=8, colors=fg_color)
    ax.spines['bottom'].set_color(fg_color)
    ax.spines['top'].set_color(fg_color)
    ax.spines['right'].set_color(fg_color)
    ax.spines['left'].set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)
    ax.xaxis.label.set_color(fg_color)
    fig.tight_layout()


def display_leave_objects(labels, inputs):
    for idx in range(len(labels)):
        labels[idx].grid(column=idx, row=0, sticky=(E, W))
        inputs[idx].grid(column=idx, row=1, sticky=(E, W), padx=1, pady=(0, 2))


def run_r_engine(settings):
    params = [
        settings.replacement_ratio,  # base_bene_level
        settings.simulation_method,  # impute_method
        True,  # makelog
        1,  # sample_prop?
        settings.state,  # state
        settings.fed_employees,  # FEDGOV
        settings.state_employees,  # STATEGOV
        settings.local_employees,  # LOCALGOV
        settings.self_employed,  # SELFEMP
        settings.state_of_work,  # place_of_work
        True,  # exclusive_particip
        False,  # SMOTE
        False,  # ext_resp_len
        'mean',  # len_method
        'unaffordable',  # sens_var?
        'post',  # progalt_post_or_pre
        True,  # intra_impute
        True,  # exclusive_particip
        True,  # ext_base_effect
        0.01,  # extend_prob
        1,  # extend_days
        1.01,  # extend_prop
        settings.top_off_rate,  # topoff_rate
        settings.top_off_min_length,  # topoff_minlength
        settings.benefit_effect,  # bene_effect
        0,  # dependent_allow
        settings.needers_fully_participate,  # fill_particip_needer
        5,  # wait_period
        settings.clone_factor,  # clone_factor
        settings.weekly_ben_cap,  # week_bene_cap
        0,  # weekly_ben_min
        settings.take_up_rates['Own Health'],  # own_uptake
        settings.take_up_rates['Maternity'],  # matdis_uptake
        settings.take_up_rates['New Child'],  # bond_uptake
        settings.take_up_rates['Ill Parent'],  # illparent_uptake
        settings.take_up_rates['Ill Spouse'],  # illspouse_uptake
        settings.take_up_rates['Ill Child'],  # illchild_uptake
        settings.max_weeks['Own Health'] * 5,  # maxlen_own
        settings.max_weeks['Maternity'] * 5,  # maxlen_matdis
        settings.max_weeks['New Child'] * 5,  # maxlen_bond
        settings.max_weeks['Ill Parent'] * 5,  # maxlen_illparent
        settings.max_weeks['Ill Spouse'] * 5,  # maxlen_illspouse
        settings.max_weeks['Ill Child'] * 5,  # maxlen_illchild
        260,  # maxlen_total
        30,  # maxlen_PFL
        260,  # maxlen_DI
        settings.eligible_earnings,  # earnings
        settings.eligible_weeks,  # weeks
        settings.eligible_hours,  # ann_hours
        settings.eligible_size,  # minsize
        settings.leave_probability_factors['Own Health'],  # own_elig_adj
        settings.leave_probability_factors['Maternity'],
        settings.leave_probability_factors['New Child'],
        settings.leave_probability_factors['Ill Parent'],
        settings.leave_probability_factors['Ill Spouse'],
        settings.leave_probability_factors['Ill Child'],
        'output',  # output
        123,  # random_seed
    ]

    command = '{} run_engine.R {}'.format(settings.r_path, ' '.join(params))
