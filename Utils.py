import copy
from tkinter import E, W
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Program parameters for California, Rhode Island, and New Jersey
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
        'weekly_ben_cap': 795,
        'fmla_protection_constraint': True,
        'eligible_earnings': 3840,
        'government_employees': False,
        'fed_employees': False,
        'state_employees': False,
        'local_employees': False,
        'self_employed': False,
    }
}


# The possible leave type options
LEAVE_TYPES = ['Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent']


class Parameters:
    def copy(self):
        """Create deep copy of self"""
        return copy.deepcopy(self)

    def update_variables(self, **kwargs):
        """Update object attributes

        :param kwargs: dict, optional
            Dictionary keys should be the attribute names and the values should be the new attribute values
        :return: None
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class GeneralParameters(Parameters):
    def __init__(self, fmla_file='', acs_directory='', output_directory='', state='',
                 simulation_method='Logistic Regression', engine_type='Python', r_path='', random_seed=None,
                 state_of_work=True):
        """General program parameters. A distinction needs to be made for program comparison, as each program will share
        these parameters but not others."""
        self.fmla_file = fmla_file
        self.acs_directory = acs_directory
        self.output_directory = output_directory
        self.state = state
        self.simulation_method = simulation_method
        self.engine_type = engine_type
        self.r_path = r_path
        self.random_seed = random_seed
        self.state_of_work = state_of_work


class OtherParameters(Parameters):
    def __init__(self, se_analysis=False, benefit_effect=False, calibrate=True, clone_factor=1,
                 extend=False, fmla_protection_constraint=False, replacement_ratio=0.6, government_employees=True,
                 needers_fully_participate=False, self_employed=False,  top_off_rate=0, top_off_min_length=0,
                 weekly_ben_cap=795, weight_factor=1, eligible_earnings=3840, eligible_weeks=1, eligible_hours=1,
                 eligible_size=1, max_weeks=None, take_up_rates=None, leave_probability_factors=None, payroll_tax=1.0,
                 benefits_tax=False, average_state_tax=5.0, max_taxable_earnings_per_person=100000,
                 total_taxable_earnings=10000000000, fed_employees=True, state_employees=True, local_employees=True,
                 counterfactual='', policy_sim=False, existing_program='', dual_receivers_share=0.6):
        """Other program parameters. A distinction needs to be made for program comparison, as each program will not
        share these parameters."""
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
        self.self_employed = self_employed
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
            # for demo using RI take up data
            self.take_up_rates = {'Own Health': 0.0704, 'Maternity': 0.0235, 'New Child': 0.0092,
                                  'Ill Child': 0.005, 'Ill Spouse': 0.0014, 'Ill Parent': 0.008}
            # alternatively use average of 3 states (RI/NJ/CA)
            # self.take_up_rates = {'Own Health': 0.0402, 'Maternity': 0.0132, 'New Child': 0.0093,
            #                           'Ill Child': 0.0011, 'Ill Spouse': 0.0020, 'Ill Parent': 0.0017}
        else:
            self.take_up_rates = take_up_rates

        if leave_probability_factors is None:
            self.leave_probability_factors = {'Own Health': 0.667, 'Maternity': 0.667, 'New Child': 0.667,
                                              'Ill Child': 0.667, 'Ill Spouse': 0.667, 'Ill Parent': 0.667}
        else:
            self.leave_probability_factors = leave_probability_factors


def center_window(window):
    """Puts the window in the center of the screen"""
    window.update_idletasks()  # Update changes to root first

    # Get the width and height of both the window and the user's screen
    ww = window.winfo_width()
    wh = window.winfo_height()
    sw = window.winfo_screenwidth()
    sh = window.winfo_screenheight()

    # Formula for calculating the center
    x = (sw / 2) - (ww / 2)
    y = (sh / 2) - (wh / 2) - 50

    # Set window minimum size
    window.minsize(ww, wh)

    window.geometry('%dx%d+%d+%d' % (ww, wh, x, y))
    window.update_idletasks()


# From https://stackoverflow.com/questions/21208376/converting-float-to-dollars-and-cents
def as_currency(amount):
    """Converts float to a dollar value"""
    if amount >= 0:
        return '${:,.2f}'.format(amount)
    else:
        return '-${:,.2f}'.format(-amount)


def generate_default_state_params(parameters=None, state='CA'):
    """Creates a new OtherParameters object with values for a state

    :param parameters: OtherParameters, default None
        If None, a new OtherParameters object is created with default values
    :param state: str, default 'CA'
        The state from which to get parameter values
    :return: OtherParameters
    """

    if parameters is None:
        # If parameters is not provided, create one
        state_params = OtherParameters()
    else:
        # If parameters is provided, copy it
        state_params = parameters.copy()

    # Update object with state values
    state_params.update_parameters(**DEFAULT_STATE_PARAMS[state.upper()])
    return state_params


def get_population_analysis_results(output_fp, types=None):
    """

    :param output_fp: str, required
        Name of simulated individual results
    :param types: list of str, default None
        Each element in list is a leave type
    :return: pd.DataFrame
    """
    # Read in simulated acs, this is just df returned from get_acs_simulated()
    if types is None:
        types = ['own', 'matdis', 'bond', 'illchild', 'illspouse', 'illparent']

    usecols = ['PWGTP', 'female', 'age', 'wage12', 'nochildren', 'asian', 'black', 'white', 'native', 'other',
               'hisp'] + ['takeup_%s' % t for t in types] + ['cpl_%s' % t for t in types]

    df = pd.read_csv(output_fp, usecols=usecols)
    # Restrict to taker/needer only (workers with neither status have cpl_type = nan)
    # d = d[(d['taker']==1) | (d['needer']==1)]

    # Restrict to workers who take up the program
    df['takeup_any'] = df[['takeup_%s' % t for t in types]].sum(axis=1) > 0
    df = df[df['takeup_any']]

    # Make sure cpl_type is non-missing
    for t in types:
        df['cpl_%s' % t] = df['cpl_%s' % t].fillna(0)

    # Total covered-by-program length
    df['cpl'] = [sum(x) for x in df[['cpl_%s' % t for t in types]].values]
    # Keep needed vars for population analysis plots
    keepcols = ['PWGTP', 'cpl', 'female', 'age', 'wage12', 'nochildren', 'asian', 'black', 'white', 'native',
                'other', 'hisp']
    df = df[keepcols]
    return df


def create_cost_chart(data, state):
    """Create a matplotlib bar chart with benefits paid for each type

    :param data: pd.DataFrame, required
        Summary results from simulation
    :param state: str, reauired
        The state that was simulated
    :return: matplotlib.figure.Figure
    """

    # Get total cost of program, in millions and rounded to 1 decimal point
    total_cost = round(list(data.loc[data['type'] == 'total', 'cost'])[0] / 10 ** 6, 1)
    spread = round((list(data.loc[data['type'] == 'total', 'ci_upper'])[0] -
                    list(data.loc[data['type'] == 'total', 'ci_lower'])[0]) / 10 ** 6, 1)
    title = 'State: %s. Total Benefits Cost = $%s million (\u00B1%s).' % (state.upper(), total_cost, spread)

    # Create chart to display benefit cost for each leave type
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ind = np.arange(6)
    ys = data[:-1]['cost'] / 10 ** 6
    es = 0.5 * (data[:-1]['ci_upper'] - data[:-1]['ci_lower']) / 10 ** 6  # Used for confidence intervals
    width = 0.5
    ax.bar(ind, ys, width, yerr=es, align='center', capsize=5, color='#1aff8c', ecolor='white')
    ax.set_ylabel('$ millions')
    ax.set_xticks(ind)
    ax.set_xticklabels(LEAVE_TYPES)
    ax.yaxis.grid(False)
    format_chart(fig, ax, title)
    return fig


def format_chart(fig, ax, title, bg_color='#333333', fg_color='#ffffff'):
    """Visually format matplotlib Figure

    :param fig: matplotlib.figure.Figure, required
    :param ax: matplotlib.axes.Axes, required
    :param title: str, required
    :param bg_color: str, default '#333333'
    :param fg_color: str, default '#ffffff'
    :return: None
    """

    # Set background color
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Set foreground color
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
    """Adds labels and entries to a widget. Each label is gridded above an entry and label/entry sets are laid out
    horizontally

    :param labels: Tk Labels, required
    :param inputs: Tk Entries, required
    :return: None
    """

    for idx in range(len(labels)):
        # Labels go above entry
        labels[idx].grid(column=idx, row=0, sticky=(E, W))
        inputs[idx].grid(column=idx, row=1, sticky=(E, W), padx=1, pady=(0, 2))


def get_sim_name(sim_num):
    """Converts a sim number to a name

    :param sim_num: int, required
        If number is 0, returns 'Main Simulation'. Otherwise returns 'Comparison <sim_num>'
    :return: str
    """
    return 'Main Simulation' if sim_num == 0 else 'Comparison {}'.format(sim_num)


def create_r_command(general_params, other_params, progress_file):
    """Uses parameter objects to create a command that, when run in a new process, will execute R engine

    :param general_params: GeneralParameters, required
    :param other_params: OtherParameters, required
    :param progress_file: str, required
        Name of text file that will be checked for simulation progress
    :return: str
    """

    # Create a list of parameter values
    params = [
        other_params.replacement_ratio,  # base_bene_level
        'KNN1',  # settings.simulation_method,  # impute_method
        True,  # makelog
        1,  # sample_prop?
        general_params.state,  # state
        other_params.fed_employees,  # FEDGOV
        other_params.state_employees,  # STATEGOV
        other_params.local_employees,  # LOCALGOV
        other_params.self_employed,  # SELFEMP
        general_params.state_of_work,  # place_of_work
        True,  # exclusive_particip
        False,  # SMOTE
        False,  # ext_resp_len
        'mean',  # len_method
        'unaffordable',  # sens_var?
        'post',  # progalt_post_or_pre
        True,  # intra_impute
        True,  # ext_base_effect
        0.01,  # extend_prob
        1,  # extend_days
        1.01,  # extend_prop
        other_params.top_off_rate,  # topoff_rate
        other_params.top_off_min_length,  # topoff_minlength
        other_params.benefit_effect,  # bene_effect
        0,  # dependent_allow
        other_params.needers_fully_participate,  # fill_particip_needer
        5,  # wait_period
        other_params.clone_factor,  # clone_factor
        other_params.weekly_ben_cap,  # week_bene_cap
        0,  # weekly_ben_min
        other_params.take_up_rates['Own Health'],  # own_uptake
        other_params.take_up_rates['Maternity'],  # matdis_uptake
        other_params.take_up_rates['New Child'],  # bond_uptake
        other_params.take_up_rates['Ill Parent'],  # illparent_uptake
        other_params.take_up_rates['Ill Spouse'],  # illspouse_uptake
        other_params.take_up_rates['Ill Child'],  # illchild_uptake
        other_params.max_weeks['Own Health'] * 5,  # maxlen_own
        other_params.max_weeks['Maternity'] * 5,  # maxlen_matdis
        other_params.max_weeks['New Child'] * 5,  # maxlen_bond
        other_params.max_weeks['Ill Parent'] * 5,  # maxlen_illparent
        other_params.max_weeks['Ill Spouse'] * 5,  # maxlen_illspouse
        other_params.max_weeks['Ill Child'] * 5,  # maxlen_illchild
        260,  # maxlen_total
        30,  # maxlen_PFL
        260,  # maxlen_DI
        other_params.eligible_earnings,  # earnings
        other_params.eligible_weeks,  # weeks
        other_params.eligible_hours,  # ann_hours
        other_params.eligible_size,  # minsize
        other_params.leave_probability_factors['Own Health'],  # own_elig_adj
        other_params.leave_probability_factors['Maternity'],  # matdis_elig_adj
        other_params.leave_probability_factors['New Child'],  # bond_elig_adj
        other_params.leave_probability_factors['Ill Parent'],  # illspouse_elig_adj
        other_params.leave_probability_factors['Ill Spouse'],  # illparent_elig_adj
        other_params.leave_probability_factors['Ill Child'],  # illchild_elig_adj
        'output',  # output
        123,  # random_seed
        progress_file.replace('r_engine/', '')
    ]

    # Convert the list into a string
    params = [str(p) for p in params]
    command = '{} --vanilla run_engine_python.R {}'.format(general_params.r_path, ' '.join(params))
    return command


def check_dependency(package_name, min_version):
    """

    :param package_name: str, required
        Name of the package
    :param min_version: str, required
        Minimum recommended version of the package. This should be a string instead of a float because some versions can
        appear like '1.0.1'.
    :return: bool
    """

    package = importlib.import_module(package_name)  # Get package from name

    # Convert versions, which are in string form, to a list of ints
    package_version = list(map(int, package.__version__.split('.')))
    min_version = list(map(int, min_version.split('.')))

    # If length of the package version list is shorter than minimum version, assume that the last digit(s) would be 0
    while len(package_version) < len(min_version):
        package_version.append(0)

    # Compare package version with minimum version, starting from the first number
    for i in range(len(min_version)):
        if min_version < package_version:
            return True
        if min_version[i] > package_version[i]:
            return False

    return True
