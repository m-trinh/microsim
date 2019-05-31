from tkinter import *
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import time
import numpy as np
from _5_simulation import SimulationEngine
from Utils import Settings, as_currency, generate_default_state_params, generate_generous_params
import collections
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ABF import ABF

version = sys.version_info


class MicrosimGUI(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Some data structures and settings that will be used throughout the code
        self.spreadsheet_ftypes = [('All', '*.xlsx; *.xls; *.csv'), ('Excel', '*.xlsx'),
                                   ('Excel 97-2003', '*.xls'), ('CSV', '*.csv')]
        self.leave_types = ['Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent']
        self.states = ('All', 'AK', 'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL',
                       'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV',
                       'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN',
                       'TX', 'UT', 'VT', 'VA', 'VI', 'WA', 'WV', 'WI', 'WY')
        self.simulation_methods = ('Logistic Regression', 'Ridge Classifier', 'K Nearest Neighbor', 'Naive Bayes',
                                   'Support Vector Machine', 'Random Forest')
        self.dark_bg = '#333333'
        self.light_font = '#f2f2f2'
        self.notebook_bg = '#fcfcfc'
        self.theme_color = '#0074BF'
        self.default_settings = Settings()
        self.settings = self.default_settings
        self.error_tooltips = []
        self.cwd = os.getcwd()

        # Edit the style for ttk widgets. These new styles are given their own names, which will have to be provided
        # by the widgets in order to be used.
        style = ttk.Style()
        style.configure('MSCombobox.TCombobox', relief='flat')
        style.configure('MSCheckbutton.TCheckbutton', background=self.notebook_bg, font='-size 12')
        style.configure('MSNotebook.TNotebook', background=self.notebook_bg)
        style.configure('MSNotebook.TNotebook.Tab', font='-size 12')
        style.configure('MSLabelframe.TLabelframe', background=self.notebook_bg)
        style.configure('MSLabelframe.TLabelframe.Label', background=self.notebook_bg, foreground=self.theme_color,
                        font='-size 12')

        self.title('Paid Leave Micro-Simulator')  # Add title to window
        self.option_add('*Font', '-size 12')  # Set default font
        # self.resizable(False, False)  # Prevent window from being resized
        self.bind("<MouseWheel>", self.scroll)  # Bind mouse wheel action to scroll function
        self.icon = PhotoImage(file='impaq_logo.gif')
        self.tk.call('wm', 'iconphoto', self._w, self.icon)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # The content frame will hold all widgets
        self.content = Frame(self, padx=15, pady=15, bg=self.dark_bg)
        # This frame holds general settings
        self.main_frame = Frame(self.content, bg=self.dark_bg)
        # This notebook will have three tabs for the program, population, and simulation settings
        self.settings_frame = ttk.Notebook(self.content, style='MSNotebook.TNotebook')
        self.run_button = MSRunButton(self.content, text="Run", command=self.run_simulation)

        # In order to control scrolling in the right notebook tab, we need to keep track of the tab that
        # is currently visible. Whenever a tab is clicked, update this value.
        self.settings_frame.bind('<Button-1>', self.change_current_tab)

        # Create frames for each notebook tab. Each frame needs a canvas because scroll bars cannot be added to a frame.
        # They can only be added to canvases and listboxes. So another frame needs to be added inside the canvas. This
        # frame will contain the actual user input widgets.
        self.program_container = Frame(self.settings_frame, bg=self.notebook_bg)
        self.program_canvas = Canvas(self.program_container, bg=self.notebook_bg)
        self.program_frame = Frame(self.program_container, padx=10, pady=10, bg=self.notebook_bg, width=600)
        self.program_canvas.create_window((0, 0), window=self.program_frame, anchor='nw')  # Add frame to canvas
        self.program_scroll = ttk.Scrollbar(self.program_container, orient=VERTICAL, command=self.program_canvas.yview)
        self.program_canvas.configure(yscrollcommand=self.program_scroll.set)  # Set scroll bar for notebook tab

        self.population_container = Frame(self.settings_frame, bg=self.notebook_bg)
        self.population_canvas = Canvas(self.population_container, bg=self.notebook_bg)
        self.population_frame = Frame(self.population_container, padx=10, pady=10, bg=self.notebook_bg)
        self.population_canvas.create_window((0, 0), window=self.population_frame, anchor='nw')
        self.population_scroll = ttk.Scrollbar(self.population_container, orient=VERTICAL,
                                               command=self.population_canvas.yview)
        self.population_canvas.configure(yscrollcommand=self.population_scroll.set)

        self.simulation_container = Frame(self.settings_frame, bg=self.notebook_bg)
        self.simulation_canvas = Canvas(self.simulation_container, bg=self.notebook_bg)
        self.simulation_frame = Frame(self.simulation_container, padx=10, pady=10, bg=self.notebook_bg)
        self.simulation_canvas.create_window((0, 0), window=self.simulation_frame, anchor='nw')
        self.simulation_scroll = ttk.Scrollbar(self.simulation_container, orient=VERTICAL,
                                               command=self.simulation_canvas.yview)
        self.simulation_canvas.configure(yscrollcommand=self.simulation_scroll.set)

        # Add the frames to the notebook
        self.settings_frame.add(self.program_container, text='Program')
        self.settings_frame.add(self.population_container, text='Population')
        self.settings_frame.add(self.simulation_container, text='Simulation')

        # Set the current visible tab to 0, which is the Program tab
        self.current_tab = 0

        # These are the variables that the users will update. These will be passed to the microsim.
        self.fmla_file = StringVar()
        self.acs_directory = StringVar()
        self.output_directory = StringVar()
        self.detail = IntVar()
        self.state = StringVar()
        self.simulation_method = StringVar()
        self.benefit_effect = BooleanVar(value=self.default_settings.benefit_effect)
        self.calibrate = BooleanVar(value=self.default_settings.calibrate)
        self.clone_factor = IntVar(value=self.default_settings.clone_factor)
        self.se_analysis = BooleanVar(value=self.default_settings.se_analysis)
        self.extend = BooleanVar(value=self.default_settings.extend)
        self.fmla_protection_constraint = BooleanVar(value=self.default_settings.fmla_protection_constraint)
        self.replacement_ratio = DoubleVar(value=self.default_settings.replacement_ratio)
        self.government_employees = BooleanVar(value=self.default_settings.government_employees)
        self.fed_employees = BooleanVar(value=self.default_settings.fed_employees)
        self.state_employees = BooleanVar(value=self.default_settings.state_employees)
        self.local_employees = BooleanVar(value=self.default_settings.local_employees)
        self.needers_fully_participate = BooleanVar(value=self.default_settings.needers_fully_participate)
        self.random_seed = BooleanVar(value=self.default_settings.random_seed)
        self.self_employed = BooleanVar(value=self.default_settings.self_employed)
        self.state_of_work = BooleanVar(value=self.default_settings.state_of_work)
        self.top_off_rate = DoubleVar(value=self.default_settings.top_off_rate)
        self.top_off_min_length = IntVar(value=self.default_settings.top_off_min_length)
        self.weekly_ben_cap = IntVar(value=self.default_settings.weekly_ben_cap)
        self.weight_factor = IntVar(value=self.default_settings.weight_factor)
        self.eligible_earnings = IntVar(value=self.default_settings.eligible_earnings)
        self.eligible_weeks = IntVar(value=self.default_settings.eligible_weeks)
        self.eligible_hours = IntVar(value=self.default_settings.eligible_hours)
        self.eligible_size = IntVar(value=self.default_settings.eligible_size)
        self.payroll_tax = DoubleVar(value=self.default_settings.payroll_tax)
        self.benefits_tax = BooleanVar(value=self.default_settings.benefits_tax)
        self.average_state_tax = DoubleVar(value=self.default_settings.average_state_tax)
        self.max_taxable_earnings_per_person = IntVar(value=self.default_settings.max_taxable_earnings_per_person)
        self.total_taxable_earnings = IntVar(value=self.default_settings.total_taxable_earnings)
        self.counterfactual = StringVar()
        self.policy_sim = BooleanVar(value=self.default_settings.policy_sim)

        # When the file location entries are modified, check to see if they all have some value
        # If they do, enable the run button
        if version[1] < 6:
            self.fmla_file.trace("w", self.check_file_entries)
            self.acs_directory.trace("w", self.check_file_entries)
            self.output_directory.trace("w", self.check_file_entries)
        else:
            self.fmla_file.trace_add("write", self.check_file_entries)
            self.acs_directory.trace_add("write", self.check_file_entries)
            self.output_directory.trace_add("write", self.check_file_entries)

        # Below is the code for creating the widgets for user inputs and labels. Entries, comboboxes, and checkboxes
        # are used. We also create tooltips to show when hovering the cursor over each input

        # ------------------------------------------- General Settings ----------------------------------------------

        self.fmla_label = Label(self.main_frame, text="FMLA File:", bg=self.dark_bg, fg=self.light_font, anchor=N)
        self.fmla_input = MSEntry(self.main_frame, textvariable=self.fmla_file, width=45)
        self.fmla_button = MSButton(self.main_frame, text="Browse", command=lambda: self.browse_file(self.fmla_input))
        self.fmla_button.config(width=None)
        CreateToolTip(self.fmla_label, 'A CSV or Excel file that contains leave taking data to use to train '
                                       'model. This should be FMLA survey data.')

        self.acs_label = Label(self.main_frame, text="ACS Directory:", bg=self.dark_bg, fg=self.light_font)
        self.acs_input = MSEntry(self.main_frame, textvariable=self.acs_directory)
        self.acs_button = MSButton(self.main_frame, text="Browse",
                                   command=lambda: self.browse_directory(self.acs_input))
        CreateToolTip(self.acs_label,
                      'A directory that contains ACS files that the model will use to estimate the cost of a paid '
                      'leave program. There should be one household and one person file for the selected state.')

        self.output_directory_label = Label(self.main_frame, text="Output Directory:", bg=self.dark_bg,
                                            fg=self.light_font)
        self.output_directory_input = MSEntry(self.main_frame, textvariable=self.output_directory)
        self.output_directory_button = MSButton(self.main_frame, text="Browse",
                                                command=lambda: self.browse_directory(self.output_directory_input))
        CreateToolTip(self.output_directory_label,
                      'The directory where the spreadsheet containing simulation results will be saved.')

        self.detail_label = Label(self.main_frame, text="Output Detail Level:", bg=self.dark_bg, fg=self.light_font)
        self.detail_input = ttk.Combobox(self.main_frame, textvariable=self.detail, state="readonly", width=5,
                                         style='MSCombobox.TCombobox')
        self.detail_input['values'] = (1, 2, 3, 4, 5, 6, 7, 8)
        self.detail_input.current(0)
        CreateToolTip(self.detail_label,
                      'The level of detail of the results. \n1 = low detail \n8 = high detail')

        self.state_label = Label(self.main_frame, text='State:', bg=self.dark_bg, fg=self.light_font)
        self.state_input = ttk.Combobox(self.main_frame, textvariable=self.state, state="readonly", width=5,
                                        values=self.states)
        self.state_input.current(self.states.index('RI'))
        CreateToolTip(self.state_label, 'The state that will be used to estimate program cost. Only people '
                                        'from this state will be chosen from the input and output files.')

        self.simulation_method_label = Label(self.main_frame, text='Simulation Method:',
                                             bg=self.dark_bg, fg=self.light_font)
        self.simulation_method_input = ttk.Combobox(self.main_frame, textvariable=self.simulation_method,
                                                    state="readonly", width=21, values=self.simulation_methods)
        self.simulation_method_input.current(0)
        CreateToolTip(self.simulation_method_label, 'The method used to train model.')

        # ------------------------------------------ Program Settings ------------------------------------------------

        self.eligibility_frame_label = ttk.Label(self.program_frame, text='Eligibility Rules:',
                                                 style='MSLabelframe.TLabelframe.Label')
        self.eligibility_frame = ttk.Labelframe(self.program_frame, labelwidget=self.eligibility_frame_label,
                                                style='MSLabelframe.TLabelframe')
        self.eligible_earnings_label = Label(self.eligibility_frame, text="Earnings", bg=self.notebook_bg)
        self.eligible_earnings_input = Entry(self.eligibility_frame, textvariable=self.eligible_earnings,
                                             justify='center', width=15)
        self.eligible_weeks_label = Label(self.eligibility_frame, text="Weeks", bg=self.notebook_bg)
        self.eligible_weeks_input = Entry(self.eligibility_frame, textvariable=self.eligible_weeks,
                                          justify='center', width=15)
        self.eligible_hours_label = Label(self.eligibility_frame, text="Hours", bg=self.notebook_bg)
        self.eligible_hours_input = Entry(self.eligibility_frame, textvariable=self.eligible_hours,
                                          justify='center', width=15)
        self.eligible_size_label = Label(self.eligibility_frame, text="Employer Size", bg=self.notebook_bg)
        self.eligible_size_input = Entry(self.eligibility_frame, textvariable=self.eligible_size,
                                         justify='center', width=15)
        CreateToolTip(self.eligibility_frame_label,
                      'The requirements to be eligible for the paid leave program. This includes '
                      'the amount of money earned in the last year, the number of weeks worked '
                      'in the last year, the number of hours worked in the ast year, and the '
                      'size of the employer.')

        self.max_weeks_frame_label = ttk.Label(self.program_frame, text='Max Weeks:',
                                               style='MSLabelframe.TLabelframe.Label')
        self.max_weeks_frame = ttk.Labelframe(self.program_frame, labelwidget=self.max_weeks_frame_label,
                                              style='MSLabelframe.TLabelframe')
        self.max_weeks, self.max_weeks_labels, self.max_weeks_inputs = self.create_leave_objects(
            self.max_weeks_frame, self.default_settings.max_weeks)
        CreateToolTip(self.max_weeks_frame_label,
                      'The maximum number of weeks for each leave type that the program will pay for.')

        self.replacement_ratio_label = Label(self.program_frame, text="Replacement Ratio:", bg=self.notebook_bg)
        self.replacement_ratio_input = Entry(self.program_frame, textvariable=self.replacement_ratio)
        CreateToolTip(self.replacement_ratio_label, 'The percentage of wage that the program will pay.')

        self.weekly_ben_cap_label = Label(self.program_frame, text="Weekly Benefit Cap:", bg=self.notebook_bg)
        self.weekly_ben_cap_input = Entry(self.program_frame, textvariable=self.weekly_ben_cap)
        CreateToolTip(self.weekly_ben_cap_label, 'The maximum amount of benefits paid out per week.')

        # self.benefit_financing_frame_label = ttk.Label(self.program_frame, text='Benefit Financing:',
        #                                                style='MSLabelframe.TLabelframe.Label')
        self.benefit_financing_frame = ttk.LabelFrame(self.program_frame, text='Benefit Financing:',
                                                      style='MSLabelframe.TLabelframe')
        self.payroll_tax_label = Label(self.benefit_financing_frame, text='Payroll Tax (%):', bg=self.notebook_bg)
        self.payroll_tax_input = Entry(self.benefit_financing_frame, textvariable=self.payroll_tax)
        CreateToolTip(self.payroll_tax_label, 'The payroll tax that will be implemented to fund benefits program.')

        self.benefits_tax_input = ttk.Checkbutton(self.benefit_financing_frame, text='Benefits Tax', onvalue=True,
                                                  offvalue=False, variable=self.benefits_tax,
                                                  style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.benefits_tax_input, 'Whether or not program benefits are taxed.')

        self.average_state_tax_label = Label(self.benefit_financing_frame, text='State Average Tax Rate (%):',
                                             bg=self.notebook_bg)
        self.average_state_tax_input = Entry(self.benefit_financing_frame, textvariable=self.average_state_tax)
        CreateToolTip(self.average_state_tax_label, 'The average tax rate of a selected state.')

        self.max_taxable_earnings_per_person_label = Label(self.benefit_financing_frame,
                                                           text='Maximum Taxable Earnings Per Person ($):',
                                                           bg=self.notebook_bg)
        self.max_taxable_earnings_per_person_input = Entry(self.benefit_financing_frame,
                                                           textvariable=self.max_taxable_earnings_per_person)
        CreateToolTip(self.max_taxable_earnings_per_person_label, 'The maximum amount that a person can be taxed.')

        self.total_taxable_earnings_label = Label(self.benefit_financing_frame, text='Total Taxable Earnings ($):',
                                                  bg=self.notebook_bg)
        self.total_taxable_earnings_input = Entry(self.benefit_financing_frame,
                                                  textvariable=self.total_taxable_earnings)
        CreateToolTip(self.total_taxable_earnings_label, 'The total earnings that can be taxed.')

        self.government_employees_input = ttk.Checkbutton(self.program_frame, text="Government Employees", onvalue=True,
                                                          offvalue=False, variable=self.government_employees,
                                                          style='MSCheckbutton.TCheckbutton',
                                                          command=self.check_all_gov_employees)
        CreateToolTip(self.government_employees_input,
                      'Whether or not government employees are eligible for program.')

        self.federal_employees_input = ttk.Checkbutton(self.program_frame, text="Federal Employees", onvalue=True,
                                                       offvalue=False, variable=self.fed_employees,
                                                       style='MSCheckbutton.TCheckbutton',
                                                       command=self.check_gov_employees)
        CreateToolTip(self.federal_employees_input,
                      'Whether or not federal employees are eligible for program.')

        self.state_employees_input = ttk.Checkbutton(self.program_frame, text="State Employees", onvalue=True,
                                                     offvalue=False, variable=self.state_employees,
                                                     style='MSCheckbutton.TCheckbutton',
                                                     command=self.check_gov_employees)
        CreateToolTip(self.state_employees_input,
                      'Whether or not state employees are eligible for program.')

        self.local_employees_input = ttk.Checkbutton(self.program_frame, text="Local Employees", onvalue=True,
                                                     offvalue=False, variable=self.local_employees,
                                                     style='MSCheckbutton.TCheckbutton',
                                                     command=self.check_gov_employees)
        CreateToolTip(self.local_employees_input,
                      'Whether or not local employees are eligible for program.')

        self.self_employed_input = ttk.Checkbutton(self.program_frame, text="Self Employed", onvalue=True,
                                                   offvalue=False, variable=self.self_employed,
                                                   style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.self_employed_input, 'Whether or not self employed workers are eligible for program.')

        self.state_of_work_input = ttk.Checkbutton(self.program_frame, text="State of Work", onvalue=True,
                                                   offvalue=False, variable=self.state_of_work,
                                                   style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.state_of_work_input,
                      'Whether or not the analysis is to be done for persons who work in particular state – '
                      'rather than for residents of a particular state.')

        # ----------------------------------------- Population Settings ----------------------------------------------

        self.take_up_rates_frame_label = ttk.Label(self.population_frame, text='Take Up Rates:',
                                                   style='MSLabelframe.TLabelframe.Label')
        self.take_up_rates_frame = ttk.Labelframe(self.population_frame, labelwidget=self.take_up_rates_frame_label,
                                                  style='MSLabelframe.TLabelframe')
        self.take_up_rates, self.take_up_rates_labels, self.take_up_rates_inputs = \
            self.create_leave_objects(self.take_up_rates_frame, self.default_settings.take_up_rates, dtype='double')
        CreateToolTip(self.take_up_rates_frame_label, 'The proportion of eligible leave takers who decide to use the '
                                                      'program for each leave type.')

        self.leave_probability_factors_frame_label = ttk.Label(self.population_frame, text='Leave Probability Factors:',
                                                               style='MSLabelframe.TLabelframe.Label')
        self.leave_probability_factors_frame = ttk.Labelframe(self.population_frame,
                                                              labelwidget=self.leave_probability_factors_frame_label,
                                                              style='MSLabelframe.TLabelframe')
        self.leave_probability_factors, self.leave_probability_factors_labels, self.leave_probability_factors_inputs = \
            self.create_leave_objects(self.leave_probability_factors_frame,
                                      self.default_settings.leave_probability_factors, dtype='double')
        CreateToolTip(self.leave_probability_factors_frame_label, 'Factors the probability of needing or taking '
                                                                  'a leave for each type of leave.')

        self.benefit_effect_input = ttk.Checkbutton(self.population_frame, text="Benefit Effect", onvalue=True,
                                                    offvalue=False, variable=self.benefit_effect,
                                                    style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.benefit_effect_input,
                      'Whether or not the benefit amount affects participation in the program.')

        self.extend_input = ttk.Checkbutton(self.population_frame, text="Extend", onvalue=True, offvalue=False,
                                            variable=self.extend, style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.extend_input,
                      'Whether or not participants extend their leave in the presence of the program.')

        self.needers_fully_participate_input = ttk.Checkbutton(self.population_frame, text="Needers Fully Participate",
                                                               onvalue=True, offvalue=False,
                                                               variable=self.needers_fully_participate,
                                                               style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.needers_fully_participate_input,
                      'Whether or not all people who need leave take leave in the presnce of the program.')

        self.top_off_rate_label = Label(self.population_frame, text="Top Off Rate:", bg=self.notebook_bg)
        self.top_off_rate_input = Entry(self.population_frame, textvariable=self.top_off_rate)
        CreateToolTip(self.top_off_rate_label,
                      'The proportion of employers already paying full wages in the absence of the program '
                      'that will top off benefits in the presence of a program to reach full wages.')

        self.top_off_min_length_label = Label(self.population_frame, text="Top Off Minimum Length:",
                                              bg=self.notebook_bg)
        self.top_off_min_length_input = Entry(self.population_frame, textvariable=self.top_off_min_length)
        CreateToolTip(self.top_off_min_length_label, 'The number of days employers will top off benefits.')

        # ----------------------------------------- Simulation Settings ----------------------------------------------

        self.clone_factor_label = Label(self.simulation_frame, text="Clone Factor:", bg=self.notebook_bg)
        self.clone_factor_input = Entry(self.simulation_frame, textvariable=self.clone_factor)
        CreateToolTip(self.clone_factor_label,
                      'The number of times each sample person will be run through the simulation.')

        self.se_analysis_input = ttk.Checkbutton(self.simulation_frame, text="SE Analysis", onvalue=True,
                                                 offvalue=False, variable=self.se_analysis,
                                                 style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.se_analysis_input, 'Whether or not weight should be divided by clone factor value.')

        self.weight_factor_label = Label(self.simulation_frame, text="Weight Factor:", bg=self.notebook_bg)
        self.weight_factor_input = Entry(self.simulation_frame, textvariable=self.weight_factor)
        CreateToolTip(self.weight_factor_label, 'Multiplies the sample weights by value.')

        self.fmla_protection_constraint_input = ttk.Checkbutton(
            self.simulation_frame, text="FMLA Protection Constraint", onvalue=True, offvalue=False,
            variable=self.fmla_protection_constraint, style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.fmla_protection_constraint_input,
                      'If checked, leaves that are extended due to a paid '
                      'leave program will be capped at 12 weeks.')

        self.calibrate_input = ttk.Checkbutton(self.simulation_frame, text="Calibrate", onvalue=True, offvalue=False,
                                               variable=self.calibrate, style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.calibrate_input,
                      'Indicates whether or not the calibration add-factors are used in the equations giving '
                      'the probability of taking or needing leaves. These calibration factors adjust the '
                      'simulated probabilities of taking or needing the most recent leave to equal those in '
                      'the Family and Medical Leave in 2012: Revised Public Use File Documentation (McGarry '
                      'et al, Abt Associates, 2013).')

        self.random_seed_input = ttk.Checkbutton(self.simulation_frame, text="Random Seed", onvalue=True,
                                                 offvalue=False, variable=self.random_seed,
                                                 style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.random_seed_input,
                      'Whether or not a seed will be created using a random number generator.')

        self.counterfactual_label = Label(self.simulation_frame, text='Compare Against Existing:', bg=self.notebook_bg)
        self.counterfactual_input = ttk.Combobox(self.simulation_frame, textvariable=self.counterfactual,
                                                 state="readonly", width=21, values=['', 'CA', 'NJ', 'RI'])
        self.counterfactual_input.current(0)
        CreateToolTip(self.counterfactual_label, 'Simulate a counterfactual scenario to compare user parameters '
                                                 'against a real paid leave program.')

        self.policy_sim_input = ttk.Checkbutton(self.simulation_frame, text="Compare Against Generous", onvalue=True,
                                                offvalue=False, variable=self.policy_sim,
                                                style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.counterfactual_label, 'Simulate a policy scenario to compare user parameters against a '
                                                 'generous paid leave program.')

        # ----------------------------------------- Add Widgets to Window --------------------------------------------

        self.content.pack(expand=True, fill=BOTH)
        self.main_frame.pack(fill=X)
        self.settings_frame.pack(expand=True, fill=BOTH, pady=8)
        self.run_button.pack(anchor=E)

        self.fmla_label.grid(column=0, row=0, sticky=W)
        self.fmla_input.grid(column=1, row=0, columnspan=3, sticky=(E, W), padx=8)
        self.fmla_button.grid(column=4, row=0, sticky=W)
        self.acs_label.grid(column=0, row=1, sticky=W)
        self.acs_input.grid(column=1, row=1, columnspan=3, sticky=(E, W), padx=8)
        self.acs_button.grid(column=4, row=1)
        self.output_directory_label.grid(column=0, row=2, sticky=W)
        self.output_directory_input.grid(column=1, row=2, columnspan=3, sticky=(E, W), padx=8)
        self.output_directory_button.grid(column=4, row=2)
        self.detail_label.grid(column=0, row=3, sticky=W)
        self.detail_input.grid(column=1, row=3, sticky=W, padx=8)
        self.state_label.grid(column=0, row=4, sticky=W)
        self.state_input.grid(column=1, row=4, sticky=W, padx=8)
        self.simulation_method_label.grid(column=0, row=5, sticky=W)
        self.simulation_method_input.grid(column=1, row=5, sticky=W, padx=8)

        self.program_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.program_scroll.pack(side=RIGHT, fill=Y)
        self.population_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.population_scroll.pack(side=RIGHT, fill=Y)
        self.simulation_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.simulation_scroll.pack(side=RIGHT, fill=Y)

        self.eligibility_frame.grid(column=0, row=0, columnspan=2, sticky=(N, E, W))
        self.eligible_earnings_label.grid(column=0, row=0)
        self.eligible_weeks_label.grid(column=1, row=0)
        self.eligible_hours_label.grid(column=2, row=0)
        self.eligible_size_label.grid(column=3, row=0)
        self.eligible_earnings_input.grid(column=0, row=1, sticky=(E, W))
        self.eligible_weeks_input.grid(column=1, row=1, sticky=(E, W))
        self.eligible_hours_input.grid(column=2, row=1, sticky=(E, W))
        self.eligible_size_input.grid(column=3, row=1, sticky=(E, W))
        self.max_weeks_frame.grid(column=0, row=1, columnspan=2, sticky=(N, E, W))
        self.display_leave_objects(self.max_weeks_labels, self.max_weeks_inputs)
        self.benefit_financing_frame.grid(column=0, row=2, columnspan=2, sticky=(N, E, W))
        self.payroll_tax_label.grid(column=0, row=0, sticky=W, padx=(8, 0))
        self.payroll_tax_input.grid(column=1, row=0, sticky=W)
        self.average_state_tax_label.grid(column=0, row=1, sticky=W, padx=(8, 0))
        self.average_state_tax_input.grid(column=1, row=1, sticky=W)
        self.benefits_tax_input.grid(column=0, row=2, columnspan=2, sticky=W, padx=(16, 0))
        self.max_taxable_earnings_per_person_label.grid(column=0, row=3, sticky=W, padx=(8, 0))
        self.max_taxable_earnings_per_person_input.grid(column=1, row=3, sticky=W)
        self.total_taxable_earnings_label.grid(column=0, row=4, sticky=W, padx=(8, 0))
        self.total_taxable_earnings_input.grid(column=1, row=4, sticky=W)
        self.replacement_ratio_label.grid(column=0, row=3, sticky=W)
        self.replacement_ratio_input.grid(column=1, row=3, sticky=W)
        self.weekly_ben_cap_label.grid(column=0, row=4, sticky=W)
        self.weekly_ben_cap_input.grid(column=1, row=4, sticky=W)
        self.government_employees_input.grid(column=0, row=5, columnspan=2, sticky=W)
        self.federal_employees_input.grid(column=0, row=6, columnspan=2, sticky=W, padx=(15, 0))
        self.state_employees_input.grid(column=0, row=7, columnspan=2, sticky=W, padx=(15, 0))
        self.local_employees_input.grid(column=0, row=8, columnspan=2, sticky=W, padx=(15, 0))
        self.self_employed_input.grid(column=0, row=9, columnspan=2, sticky=W)
        self.state_of_work_input.grid(column=0, row=10, columnspan=2, sticky=W)

        self.take_up_rates_frame.grid(column=0, row=0, columnspan=2, sticky=(N, E, W))
        self.display_leave_objects(self.take_up_rates_labels, self.take_up_rates_inputs)
        self.leave_probability_factors_frame.grid(column=0, row=1, columnspan=2, sticky=(N, E, W))
        self.display_leave_objects(self.leave_probability_factors_labels, self.leave_probability_factors_inputs)
        self.benefit_effect_input.grid(column=0, row=2, columnspan=2, sticky=W)
        self.extend_input.grid(column=0, row=3, columnspan=3, sticky=W)
        self.needers_fully_participate_input.grid(column=0, row=4, columnspan=2, sticky=W)
        self.top_off_rate_label.grid(column=0, row=5, sticky=W)
        self.top_off_rate_input.grid(column=1, row=5, sticky=W)
        self.top_off_min_length_label.grid(column=0, row=6, sticky=W)
        self.top_off_min_length_input.grid(column=1, row=6, sticky=W)

        self.clone_factor_label.grid(column=0, row=0, sticky=W)
        self.clone_factor_input.grid(column=1, row=0)
        self.se_analysis_input.grid(column=0, row=1, columnspan=2, sticky=W)
        self.weight_factor_label.grid(column=0, row=2, sticky=W)
        self.weight_factor_input.grid(column=1, row=2)
        self.fmla_protection_constraint_input.grid(column=0, row=3, columnspan=2, sticky=W)
        self.calibrate_input.grid(column=0, row=4, columnspan=2, sticky=W)
        self.random_seed_input.grid(column=0, row=5, columnspan=2, sticky=W)
        self.counterfactual_label.grid(column=0, row=6, sticky=W)
        self.counterfactual_input.grid(column=1, row=6)
        self.policy_sim_input.grid(column=0, row=7, columnspan=2, sticky=W)

        # This code adds padding to each row. This is needed when using grid() to add widgets.
        self.row_padding = 8
        for i in range(6):
            self.main_frame.rowconfigure(i, pad=self.row_padding)
        for i in range(11):
            self.program_frame.rowconfigure(i, pad=self.row_padding)
        for i in range(7):
            self.population_frame.rowconfigure(i, pad=self.row_padding)
        for i in range(8):
            self.simulation_frame.rowconfigure(i, pad=self.row_padding)
        for i in range(5):
            self.benefit_financing_frame.rowconfigure(i, pad=self.row_padding)

        # Set column weights. This will cause certain columns to take up more space.
        self.main_frame.columnconfigure(1, weight=1)
        self.program_frame.columnconfigure(0, weight=0)
        self.program_frame.columnconfigure(1, weight=1)
        self.population_frame.columnconfigure(0, weight=0)
        self.population_frame.columnconfigure(1, weight=1)
        for i in range(4):
            self.eligibility_frame.columnconfigure(i, weight=1)

        self.position_window()
        self.settings_frame.bind('<Configure>', self.resize)

        self.set_notebook_width(self.settings_frame.winfo_width() - 30)
        self.set_scroll_region()

        # TODO: Remove
        # --------- TEST ONLY -------------
        self.fmla_file.set('./data/fmla_2012/fmla_2012_employee_restrict_puf.csv')
        self.acs_directory.set('./data/acs')
        self.output_directory.set('./output')
        # self.test_result_output()

    def check_all_gov_employees(self, event=None):
        checked = self.government_employees.get()
        self.fed_employees.set(checked)
        self.state_employees.set(checked)
        self.local_employees.set(checked)

    def check_gov_employees(self):
        if self.fed_employees.get() and self.state_employees.get() and self.local_employees.get():
            self.government_employees.set(1)
        else:
            self.government_employees.set(0)

    # Puts the window in the center of the screen
    def position_window(self):
        self.update()  # Update changes to root first

        # Get the width and height of both the window and the user's screen
        ww = self.winfo_width()
        wh = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        # Formula for calculating the center
        x = (sw / 2) - (ww / 2)
        y = (sh / 2) - (wh / 2) - 50

        # Set window minimum size
        self.minsize(ww, wh)

        self.geometry('%dx%d+%d+%d' % (ww, wh, x, y))

    def on_close(self):
        self.destroy()

    def run_simulation(self):
        self.clear_errors()
        errors = self.validate_settings()

        if len(errors) > 0:
            self.display_errors(errors)
            return

        settings = self.create_settings()

        # run simulation
        # initiate a SimulationEngine instance

        se = self.create_simulation_engine(settings)
        self.se = se
        counterfactual_se = None
        counterfactual = self.counterfactual.get()
        if counterfactual != '':
            counterfactual_se = self.create_simulation_engine(generate_default_state_params(settings),
                                                              engine_type='counterfactual')

        policy_se = None
        if settings.policy_sim:
            policy_se = self.create_simulation_engine(generate_generous_params(settings), engine_type='policy_sim')
        self.policy_se = policy_se

        self.counterfactual_se = counterfactual_se
        self.run_button.config(state=DISABLED, bg='#99d6ff')
        self.progress_window = ProgressWindow(self, se, counterfactual_se, policy_se)
        # Run model
        thread_se = threading.Thread(target=se.run)
        thread_se.start()

        thread_progress = threading.Thread(target=self.progress_window.update_progress)
        thread_progress.start()

    def show_results(self):
        # compute program costs
        costs = self.se.get_cost_df()
        # d_bars = {'Own Health': list(costs.loc[costs['type'] == 'own', 'cost'])[0],
        #           'Maternity': list(costs.loc[costs['type'] == 'matdis', 'cost'])[0],
        #           'New Child': list(costs.loc[costs['type'] == 'bond', 'cost'])[0],
        #           'Ill Child': list(costs.loc[costs['type'] == 'illchild', 'cost'])[0],
        #           'Ill Spouse': list(costs.loc[costs['type'] == 'illspouse', 'cost'])[0],
        #           'Ill Parent': list(costs.loc[costs['type'] == 'illparent', 'cost'])[0]}
        #
        # ks = ['Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent']
        # od_bars = collections.OrderedDict((k, d_bars[k]) for k in ks)

        total_benefits = list(costs.loc[costs['type'] == 'total', 'cost'])[0]
        self.abf_module = ABF(self.se.get_results(), self.settings, total_benefits)
        abf_output, pivot_tables = self.abf_module.run()

        self.results_window = ResultsWindow(self, self.se, abf_output, pivot_tables,
                                            counterfactual_engine=self.counterfactual_se, policy_engine=self.policy_se)
        self.run_button.config(state=NORMAL, bg=self.theme_color)

    def run_abf(self):
        abf_output, pivot_tables = self.abf_module.rerun(self.settings)
        # self.results_window.update_abf_output(abf_output)
        # self.results_window.update_pivot_tables(pivot_tables)
        self.results_window.update_abf_output(abf_output, pivot_tables)

    # def test_result_output(self):
    #     # This code currently generates a mock up of a results window
    #     simulation_results = {'Own Health': 10, 'Maternity': 200, 'New Child': 10, 'Ill Child': 8,
    #                           'Ill Spouse': 5, 'Ill Parent': 0}
    #     abf_output = {'Total Income (Weighted)': 0, 'Total Income': 0,
    #                   'Income Standard Error': 0, 'Total Income Upper Confidence Interval': 0,
    #                   'Total Income Lower Confidence Interval': 0,
    #                   'Total Tax Revenue': 0,
    #                   'Tax Standard Error': 0, 'Total Tax Revenue Upper Confidence Interval': 0,
    #                   'Total Tax Revenue Lower Confidence Interval': 0, 'Tax Revenue Recouped from Benefits': 0}
    #
    #     pivot_table = pd.DataFrame({('sum', 'income_w'): 0, ('sum', 'ptax_rev_w'): 0},
    #                                index=['Private', 'Federal', 'State', 'Local'])
    #
    #     pivot_tables = {'class': pivot_table, 'age': pivot_table, 'gender': pivot_table}
    #
    #     self.results_window = ResultsWindow(self, simulation_results, abf_output, pivot_tables)

    # Create an object with all of the setting values
    def create_settings(self):
        # The inputs are linked to a tkinter variable. Those values will have to be retrieved from each variable
        # and passed on to the settings objects
        self.settings = Settings(self.fmla_file.get(), self.acs_directory.get(), self.output_directory.get(), self.detail.get(),
                        self.state.get(), self.simulation_method.get(), self.benefit_effect.get(), self.calibrate.get(),
                        self.clone_factor.get(), self.se_analysis.get(), self.extend.get(),
                        self.fmla_protection_constraint.get(), self.replacement_ratio.get(),
                        self.government_employees.get(), self.needers_fully_participate.get(),
                        self.random_seed.get(), self.self_employed.get(), self.state_of_work.get(),
                        self.top_off_rate.get(), self.top_off_min_length.get(), self.weekly_ben_cap.get(),
                        self.weight_factor.get(), self.eligible_earnings.get(), self.eligible_weeks.get(),
                        self.eligible_hours.get(), self.eligible_size.get(),
                        {key: value.get() for key, value in self.max_weeks.items()},
                        {key: value.get() for key, value in self.take_up_rates.items()},
                        {key: value.get() for key, value in self.leave_probability_factors.items()},
                        self.payroll_tax.get(), self.benefits_tax.get(), self.average_state_tax.get(),
                        self.max_taxable_earnings_per_person.get(), self.total_taxable_earnings_input.get(),
                        self.fed_employees.get(), self.state_employees.get(), self.local_employees.get(),
                        self.counterfactual.get(), self.policy_sim.get())

        return self.settings

    def create_simulation_engine(self, settings, engine_type='main'):
        st = settings.state.lower()
        yr = 16
        fp_fmla_in = settings.fmla_file
        fp_cps_in = './data/cps/CPS2014extract.csv'
        fp_acsh_in = settings.acs_directory
        fp_acsp_in = settings.acs_directory
        fp_fmla_out = './data/fmla_2012/fmla_clean_2012.csv'
        fp_cps_out = './data/cps/cps_for_acs_sim.csv'
        fp_acs_out = './data/acs/'
        fp_length_distribution_out = './data/fmla_2012/length_distributions.json'
        fps_in = [fp_fmla_in, fp_cps_in, fp_acsh_in, fp_acsp_in]
        fps_out = [fp_fmla_out, fp_cps_out, fp_acs_out, fp_length_distribution_out]

        # fullFp_acs, fullFp_fmla, fullFp_out = settings.acs_file, settings.fmla_file, settings.output_directory
        # fp_fmla = '.'+fullFp_fmla[fullFp_fmla.find('/data/fmla_2012/'):]
        # print(fp_fmla)
        # fp_acs = '.'+fullFp_acs[fullFp_acs.find('/data/acs/'):]
        # fp_out = fullFp_out
        clf_name = settings.simulation_method

        # prog_para
        elig_wage12 = settings.eligible_earnings
        elig_wkswork = settings.eligible_weeks
        elig_yrhours = settings.eligible_hours
        elig_empsize = settings.eligible_size
        rrp = settings.replacement_ratio
        wkbene_cap = settings.weekly_ben_cap

        d_maxwk = {
            'own': settings.max_weeks['Own Health'],
            'matdis': settings.max_weeks['Maternity'],
            'bond': settings.max_weeks['New Child'],
            'illchild': settings.max_weeks['Ill Child'],
            'illspouse': settings.max_weeks['Ill Spouse'],
            'illparent': settings.max_weeks['Ill Parent']
        }

        d_takeup = {
            'own': settings.take_up_rates['Own Health'],
            'matdis': settings.take_up_rates['Maternity'],
            'bond': settings.take_up_rates['New Child'],
            'illchild': settings.take_up_rates['Ill Child'],
            'illspouse': settings.take_up_rates['Ill Spouse'],
            'illparent': settings.take_up_rates['Ill Parent']
        }

        incl_empgov_fed = settings.fed_employees
        incl_empgov_st = settings.state_employees
        incl_empgov_loc = settings.local_employees
        incl_empself = settings.self_employed
        sim_method = settings.simulation_method

        prog_para = [elig_wage12, elig_wkswork, elig_yrhours, elig_empsize, rrp, wkbene_cap, d_maxwk, d_takeup,
                     incl_empgov_fed, incl_empgov_st, incl_empgov_loc, incl_empself, sim_method]

        return SimulationEngine(st, yr, fps_in, fps_out, clf_name, prog_para, engine_type=engine_type)

    def run_engine(self, engine):
        thread_se = threading.Thread(target=engine.run)
        thread_se.start()

    def browse_file(self, file_input):
        # Open a file dialogue where user can choose a file. Possible options are limited to CSV and Excel files.
        file_name = filedialog.askopenfilename(initialdir=self.cwd, filetypes=self.spreadsheet_ftypes)
        file_input.delete(0, END)  # Clear current value in entry widget
        file_input.insert(0, file_name)  # Add user-selected value to entry widget

    def browse_directory(self, directory_input):
        # Open a file dialogue where user can choose a directory.
        directory_name = filedialog.askdirectory(initialdir=self.cwd)
        directory_input.delete(0, END)  # Clear current value in entry widget
        directory_input.insert(0, directory_name)  # Add user-selected value to entry widget

    def save_file(self, figure):
        filename = filedialog.asksaveasfilename(defaultextension='.png', initialdir=self.cwd,
                                                filetypes=[('PNG', '.png'), ('PDF', '*.pdf'), ('PGF', '*.pgf'),
                                                           ('EPS', '*.eps'), ('PS', '*.ps'), ('Raw', '*.raw'),
                                                           ('RGBA', '*.rgba'), ('SVG', '*.svg'), ('SVGZ', '*.svgz')])
        if filename is None:
            return

        figure.savefig(filename)

    def check_file_entries(self, *_):
        if self.fmla_file.get() and self.acs_directory.get() and self.output_directory.get():
            self.run_button.config(state=NORMAL, bg=self.theme_color)
        else:
            self.run_button.config(state=DISABLED, bg='#99d6ff')

    def validate_settings(self):
        errors = []

        integer_entries = [self.eligible_earnings_input, self.eligible_weeks_input, self.eligible_hours_input,
                           self.eligible_size_input, self.weekly_ben_cap_input, self.top_off_min_length_input,
                           self.clone_factor_input, self.weight_factor_input,
                           self.max_taxable_earnings_per_person_input, self.total_taxable_earnings_input]
        integer_entries += [entry for entry in self.max_weeks_inputs]

        float_entries = [self.payroll_tax_input, self.average_state_tax_input]

        rate_entries = [self.replacement_ratio_input, self.top_off_rate_input]
        rate_entries += [entry for entry in self.take_up_rates_inputs]
        rate_entries += [entry for entry in self.leave_probability_factors_inputs]

        for entry in integer_entries:
            if not self.validate_integer(entry.get()):
                errors.append((entry, 'This field should contain an integer greater than or equal to 0'))

        for entry in float_entries:
            if not self.validate_float(entry.get()):
                errors.append((entry, 'This field should contain an integer greater than or equal to 0'))

        for entry in rate_entries:
            if not self.validate_rate(entry.get()):
                errors.append((entry, 'This field should contain a number greater than or equal to '
                                      '0 and less than or equal to 1'))

        return errors

    @staticmethod
    def validate_integer(value):
        try:
            return int(value) >= 0
        except ValueError:
            return False

    @staticmethod
    def validate_float(value):
        try:
            return float(value) >= 0
        except ValueError:
            return False

    @staticmethod
    def validate_rate(value):
        try:
            return 0 <= float(value) <= 1
        except ValueError:
            return False

    def display_errors(self, errors):
        for widget, error in errors:
            widget.config(bg='red', fg='white')
            self.error_tooltips.append((widget, CreateToolTip(widget, error)))

        messagebox.showinfo('Error', message='There was an error with one or more entries.')

    def clear_errors(self):
        for widget, tooltip in self.error_tooltips:
            widget.config(bg='white', fg='black')
            tooltip.hidetip()

        self.error_tooltips = []

    # Allows the scroll wheel to move a scrollbar
    def scroll(self, event):
        # In Windows, the delta will be either 120 or -120. In Mac, it will be 1 or -1.
        # The delta value will determine whether the user is scrolling up or down.
        move_unit = 0
        if event.num == 5 or event.delta > 0:
            move_unit = -1
        elif event.num == 4 or event.delta < 0:
            move_unit = 1

        # Only scroll the tab that is currently visible.
        if self.current_tab == 0:
            self.program_canvas.yview_scroll(move_unit, 'units')
        elif self.current_tab == 1:
            self.population_canvas.yview_scroll(move_unit, 'units')
        elif self.current_tab == 2:
            self.simulation_canvas.yview_scroll(move_unit, 'units')

    # Change the currently visible tab.
    def change_current_tab(self, event):
        self.current_tab = self.settings_frame.tk.call(self.settings_frame._w, "identify", "tab", event.x, event.y)

    def set_scroll_region(self, height=-1):
        canvas_frame_list = [(self.population_canvas, self.population_frame),
                             (self.simulation_canvas, self.simulation_frame),
                             (self.program_canvas, self.program_frame)]

        canvas_height = self.program_canvas.winfo_height() if height < 0 else height

        for canvas, frame in canvas_frame_list:
            frame_height = frame.winfo_height()

            new_height = frame_height if frame_height > canvas_height else canvas_height
            canvas.configure(scrollregion=(0, 0, 0, new_height))

    def set_notebook_width(self, width):
        self.program_canvas.itemconfig(1, width=width)
        self.population_canvas.itemconfig(1, width=width)
        self.simulation_canvas.itemconfig(1, width=width)

    def resize(self, event):
        new_width = event.width - 30
        self.set_notebook_width(new_width)
        self.set_scroll_region(event.height - 30)

    # Some inputs require an entry value for each leave type. It is better to store each input in a list than
    # create separate variables for all of them.
    def create_leave_objects(self, parent, default_input, dtype='int'):
        leave_vars = {}  # A dictionary of the variables that will be updated by the user
        leave_type_labels = []  # A list of label widgets for inputs
        leave_type_inputs = []  # A list of entry inputs
        for i, leave_type in enumerate(self.leave_types):
            # The only data types right now for variables are integer and double.
            if dtype == 'double':
                leave_vars[leave_type] = DoubleVar(value=default_input[leave_type])
            else:
                leave_vars[leave_type] = IntVar(value=default_input[leave_type])

            # Create the label and entry widgets
            leave_type_labels.append(Label(parent, text=leave_type, bg=self.notebook_bg))
            leave_type_inputs.append(Entry(parent, textvariable=leave_vars[leave_type], justify='center', width=10))
            parent.columnconfigure(i, weight=1)

        return leave_vars, leave_type_labels, leave_type_inputs

    # Display label and entry widgets for inputs that exist for each leave type.
    @staticmethod
    def display_leave_objects(labels, inputs):
        for idx in range(len(labels)):
            labels[idx].grid(column=idx, row=0, sticky=(E, W))
            inputs[idx].grid(column=idx, row=1, sticky=(E, W))


class ResultsWindow(Toplevel):
    def __init__(self, parent, simulation_engine, abf_output, pivot_tables, counterfactual_engine=None,
                 policy_engine=None):
        super().__init__(parent)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.icon = PhotoImage(file='impaq_logo.gif')
        self.tk.call('wm', 'iconphoto', self._w, self.icon)

        self.parent = parent
        self.dark_bg = parent.dark_bg
        self.notebook_bg = parent.notebook_bg
        self.light_font = parent.light_font

        self.content = Frame(self, bg=self.dark_bg)
        self.notebook = ttk.Notebook(self.content, style='MSNotebook.TNotebook')
        self.notebook.bind('<Button-1>', self.change_current_tab)
        self.current_tab = 0

        self.summary_frame = ResultsSummary(self, simulation_engine)
        self.notebook.add(self.summary_frame, text='Summary')

        self.abf = Frame(self.notebook, bg=self.notebook_bg)
        # self.abf_info_container = Frame(self.abf, bg=self.dark_bg)
        self.abf_canvas = Canvas(self.abf, bg=self.dark_bg)
        self.abf_info = Frame(self.abf, bg=self.dark_bg)
        self.abf_canvas.create_window((0, 0), window=self.abf_info, anchor='nw')  # Add frame to canvas
        self.abf_info_scroll = ttk.Scrollbar(self.abf, orient=VERTICAL,
                                             command=self.abf_canvas.yview)
        self.abf_canvas.configure(yscrollcommand=self.abf_info_scroll.set)
        # self.abf_info_container.pack(side=TOP, fill=BOTH, expand=True)
        self.abf_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=0, pady=0)
        self.abf_info_scroll.pack(side=RIGHT, fill=Y)
        self.abf_summary = ABFResultsSummary(self.abf_info, abf_output, self.dark_bg, self.light_font)
        self.abf_summary.pack(padx=10, pady=10)
        self.abf_pivot_tables = Frame(self.abf_info, bg=self.dark_bg)
        self.abf_pivot_tables.pack(fill=X, expand=True)
        self.abf_params_reveal = MSButton(self.abf, text='ABF Parameters', padx=4, command=self.show_params, width=16,
                                          background='#00e600')
        self.abf_params_reveal.pack(side=BOTTOM, anchor='se', padx=3, pady=2)
        self.abf_params = Frame(self.abf, bg=self.notebook_bg, borderwidth=1, relief='solid', padx=3, pady=3)
        # self.abf_params.pack(side=BOTTOM, anchor='se', padx=1)
        self.abf_params_inputs = Frame(self.abf_params, bg=self.notebook_bg, pady=4)
        self.abf_params_inputs.pack(fill=X, side=TOP)
        self.abf_params_buttons = Frame(self.abf_params, bg=self.notebook_bg, pady=4)
        self.abf_params_buttons.pack(side=BOTTOM, fill=X, expand=True)
        self.abf_params_hide = MSButton(self.abf_params_buttons, text='Hide', padx=4, command=self.hide_params,
                                        background='#00e600')
        self.abf_params_hide.pack(side=LEFT, pady=3, padx=5)
        self.run_button = MSButton(self.abf_params_buttons, font='-size 12 -weight bold', text="Run ABF",
                                   command=parent.run_abf, padx=4)
        self.run_button.pack(side=RIGHT, pady=3, padx=5)
        self.notebook.add(self.abf, text="Benefit Financing")

        simulation_data = simulation_engine.get_population_analysis_results()
        self.population_analysis_container = Frame(self.notebook, bg=self.notebook_bg)
        self.population_analysis_canvas = Canvas(self.population_analysis_container, bg=self.dark_bg)
        self.population_analysis = Frame(self.population_analysis_container, bg=self.dark_bg)
        self.population_analysis_canvas.create_window((0, 0), window=self.population_analysis, anchor='nw')
        self.population_analysis_scroll = ttk.Scrollbar(self.population_analysis_container, orient=VERTICAL,
                                                        command=self.population_analysis_canvas.yview)
        self.population_analysis_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=0, pady=0)
        self.population_analysis_scroll.pack(side=RIGHT, fill=Y)
        self.notebook.add(self.population_analysis_container, text='Population Analysis')
        self.generate_population_analysis_histograms(simulation_data)

        self.canvases = [(self.abf_canvas, self.abf_info), (self.population_analysis_canvas, self.population_analysis)]

        if policy_engine is not None:
            self.policy_sim_container = Frame(self.notebook, bg=self.dark_bg)
            self.policy_sim_canvas = Canvas(self.policy_sim_container, bg=self.dark_bg)
            self.policy_sim = Frame(self.policy_sim_container, bg=self.dark_bg)
            self.policy_sim_canvas.create_window((0, 0), window=self.policy_sim, anchor='nw')
            self.policy_sim_scroll = ttk.Scrollbar(self.policy_sim_container, orient=VERTICAL,
                                                   command=self.policy_sim_canvas.yview)
            self.policy_sim_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=0, pady=0)
            self.policy_sim_scroll.pack(side=RIGHT, fill=Y)
            self.generate_policy_histograms(simulation_data, policy_engine.get_population_analysis_results())
            self.notebook.add(self.policy_sim_container, text='Policy Simulation')
            self.canvases.append((self.policy_sim_canvas, self.policy_sim))

        if counterfactual_engine is not None:
            # self.counterfactual_frame = ResultsSummary(self, counterfactual_engine)
            self.counterfactual_container = Frame(self.notebook, bg=self.dark_bg)
            self.counterfactual_canvas = Canvas(self.counterfactual_container, bg=self.dark_bg)
            self.counterfactual = Frame(self.counterfactual_container, bg=self.dark_bg)
            self.counterfactual_canvas.create_window((0, 0), window=self.counterfactual, anchor='nw')
            self.counterfactual_scroll = ttk.Scrollbar(self.counterfactual_container, orient=VERTICAL,
                                                       command=self.counterfactual_canvas.yview)
            self.counterfactual_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=0, pady=0)
            self.counterfactual_scroll.pack(side=RIGHT, fill=Y)
            self.generate_counterfactual_histograms(simulation_data,
                                                    counterfactual_engine.get_population_analysis_results())
            self.notebook.add(self.counterfactual_container, text='Counterfactual Simulation')
            self.canvases.append((self.counterfactual_canvas, self.counterfactual))

        self.content.pack(expand=True, fill=BOTH)
        self.notebook.pack(expand=True, fill=BOTH)
        self.notebook.select(self.summary_frame)
        self.notebook.enable_traversal()
        self.display_abf_bar_graphs(pivot_tables)

        self.bind("<MouseWheel>", self.scroll_abf)

        self.payroll_tax_label = Label(self.abf_params_inputs, text='Payroll Tax (%):', bg=self.notebook_bg)
        self.payroll_tax_input = Entry(self.abf_params_inputs, textvariable=parent.payroll_tax)
        CreateToolTip(self.payroll_tax_label, 'The payroll tax that will be implemented to fund benefits program.')

        self.benefits_tax_input = ttk.Checkbutton(self.abf_params_inputs, text='Benefits Tax', onvalue=True,
                                                  offvalue=False, variable=parent.benefits_tax,
                                                  style='MSCheckbutton.TCheckbutton')
        CreateToolTip(self.benefits_tax_input, 'Whether or not program benefits are taxed.')

        self.average_state_tax_label = Label(self.abf_params_inputs, text='State Average Tax Rate (%):',
                                             bg=self.notebook_bg)
        self.average_state_tax_input = Entry(self.abf_params_inputs, textvariable=parent.average_state_tax)
        CreateToolTip(self.average_state_tax_label, 'The average tax rate of a selected state.')

        self.max_taxable_earnings_per_person_label = Label(self.abf_params_inputs,
                                                           text='Maximum Taxable Earnings\nPer Person ($):',
                                                           bg=self.notebook_bg, justify=LEFT)
        self.max_taxable_earnings_per_person_input = Entry(self.abf_params_inputs,
                                                           textvariable=parent.max_taxable_earnings_per_person)
        CreateToolTip(self.max_taxable_earnings_per_person_label, 'The maximum amount that a person can be taxed.')

        self.total_taxable_earnings_label = Label(self.abf_params_inputs, text='Total Taxable Earnings ($):',
                                                  bg=self.notebook_bg)
        self.total_taxable_earnings_input = Entry(self.abf_params_inputs,
                                                  textvariable=parent.total_taxable_earnings)
        CreateToolTip(self.total_taxable_earnings_label, 'The total earnings that can be taxed.')

        self.payroll_tax_label.grid(column=0, row=0, sticky=W, padx=(8, 0))
        self.payroll_tax_input.grid(column=1, row=0, sticky=W)
        self.average_state_tax_label.grid(column=0, row=1, sticky=W, padx=(8, 0))
        self.average_state_tax_input.grid(column=1, row=1, sticky=W)
        self.benefits_tax_input.grid(column=0, row=2, columnspan=2, sticky=W, padx=(16, 0))
        self.max_taxable_earnings_per_person_label.grid(column=0, row=3, sticky=W, padx=(8, 0))
        self.max_taxable_earnings_per_person_input.grid(column=1, row=3, sticky=W)
        self.total_taxable_earnings_label.grid(column=0, row=4, sticky=W, padx=(8, 0))
        self.total_taxable_earnings_input.grid(column=1, row=4, sticky=W)

        self.update()
        for canvas, frame in self.canvases:
            canvas.configure(scrollregion=(0, 0, 0, frame.winfo_height()))
        # self.abf_canvas.configure(scrollregion=(0, 0, 0, self.abf_info.winfo_height()))

    def show_params(self):
        self.abf_params_reveal.pack_forget()
        self.abf_params.pack(side=BOTTOM, anchor='se', padx=1)

    def hide_params(self):
        self.abf_params.pack_forget()
        self.abf_params_reveal.pack(side=BOTTOM, anchor='se', padx=3, pady=2)

    def display_sim_bar_graph(self, simulation_chart, frame):
        canvas = FigureCanvasTkAgg(simulation_chart, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        save_button = MSButton(frame, text='Save Figure', command=lambda: self.save_file(simulation_chart))
        save_button.config(width=0)
        save_button.pack(side=RIGHT, padx=10, pady=10)

    def display_abf_bar_graphs(self, pivot_tables):
        graphs = self.create_abf_bar_graphs(pivot_tables)
        for graph in graphs:
            chart_container = ChartContainer(self.abf_pivot_tables, graph, self.dark_bg)
            chart_container.pack()
            # canvas = FigureCanvasTkAgg(graph, self.abf_pivot_tables)
            # canvas.draw()
            # canvas.get_tk_widget().config(height=300)
            # canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=10)

    def create_abf_bar_graphs(self, pivot_tables):
        graphs = []
        fg_color = 'white'
        bg_color = self.dark_bg
        # bg_color = '#1a1a1a'

        for pivot_table_category, pivot_table in pivot_tables.items():
            fig_pivot = Figure(figsize=(8, 4))
            ax_pivot = fig_pivot.add_subplot(111)

            categories = pivot_table.index.tolist()
            ind_pivot = np.arange(len(categories))
            width_pivot = 0.5
            ys_pivot = pivot_table[('sum', 'ptax_rev_w')].values / 10 ** 6
            title_pivot = 'State: {}. {} by {}'.format(self.parent.state.get(), 'Total Tax Revenue',
                                                       pivot_table_category)
            if len(categories) > 3:
                ax_pivot.bar(ind_pivot, ys_pivot, width_pivot, align='center', color='#1aff8c')
                ax_pivot.set_ylabel('$ millions', fontsize=9)
                ax_pivot.set_xticks(ind_pivot)
                ax_pivot.set_xticklabels(categories)
                ax_pivot.yaxis.grid(False)
            else:
                ax_pivot.barh(ind_pivot, ys_pivot, width_pivot, align='center', color='#1aff8c')
                ax_pivot.set_xlabel('$ millions', fontsize=9)
                ax_pivot.set_yticks(ind_pivot)
                ax_pivot.set_yticklabels(categories)
                ax_pivot.xaxis.grid(False)

            self.format_chart(fig_pivot, ax_pivot, title_pivot, bg_color, fg_color)

            graphs.append(fig_pivot)

        return graphs

    def format_chart(self, fig, ax, title, bg_color, fg_color):
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

    def save_file(self, figure):
        filename = filedialog.asksaveasfilename(defaultextension='.png', initialdir=os.getcwd(),
                                                filetypes=[('PNG', '.png'), ('PDF', '*.pdf'), ('PGF', '*.pgf'),
                                                           ('EPS', '*.eps'), ('PS', '*.ps'), ('Raw', '*.raw'),
                                                           ('RGBA', '*.rgba'), ('SVG', '*.svg'), ('SVGZ', '*.svgz')])
        if filename is None:
            return

        figure.savefig(filename)

    def generate_population_analysis_histograms(self, population_analysis_data):
        fg_color = 'white'
        bg_color = self.dark_bg

        self.create_histogram(self.population_analysis, population_analysis_data['cpl'], 20,
                              population_analysis_data['PWGTP'], bg_color, fg_color, 'All Workers')

        female_data = population_analysis_data[population_analysis_data['female'] == 1]
        self.create_histogram(self.population_analysis, female_data['cpl'], 20,
                              female_data['PWGTP'], bg_color, fg_color, 'Female Workers')

        child_bearing_data = female_data[(female_data['age'] >= 25) & (female_data['age'] <= 40)]
        self.create_histogram(self.population_analysis, child_bearing_data['cpl'], 20,
                              child_bearing_data['PWGTP'], bg_color, fg_color, 'Female Child Bearing Age')

    def generate_counterfactual_histograms(self, simulation_data, counterfactual_data):
        fg_color = 'white'
        bg_color = self.dark_bg
        self.create_histogram(self.counterfactual, simulation_data['cpl'], 20,
                              simulation_data['PWGTP'], bg_color, fg_color, 'Current Program')
        self.create_histogram(self.counterfactual, counterfactual_data['cpl'], 20,
                              counterfactual_data['PWGTP'], bg_color, fg_color,
                              'Counterfactual Program ({})'.format(self.parent.settings.counterfactual))

    def generate_policy_histograms(self, simulation_data, policy_sim_data):
        fg_color = 'white'
        bg_color = self.dark_bg
        self.create_histogram(self.policy_sim, simulation_data['cpl'], 20,
                              simulation_data['PWGTP'], bg_color, fg_color, 'Current Program')
        self.create_histogram(self.policy_sim, policy_sim_data['cpl'], 20,
                              policy_sim_data['PWGTP'], bg_color, fg_color,
                              'Most Generous Program')

    def create_histogram(self, parent, data, bins, weights, bg_color, fg_color, title_str):
        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.hist(data, bins, weights=weights, color='#1aff8c')
        title = 'State: {}. {}'.format(self.parent.state.get(), title_str)
        self.format_chart(fig, ax, title, bg_color, fg_color)
        chart_container = ChartContainer(parent, fig, self.dark_bg)
        chart_container.pack()

    def update_abf_output(self, abf_output, pivot_tables):
        for graph in self.abf_pivot_tables.winfo_children():
            graph.destroy()

        self.display_abf_bar_graphs(pivot_tables)
        self.abf_summary.update_results(abf_output)

    def scroll_abf(self, event):
        move_unit = 0
        if event.num == 5 or event.delta > 0:
            move_unit = -1
        elif event.num == 4 or event.delta < 0:
            move_unit = 1

        if self.current_tab > 0:
            self.canvases[self.current_tab - 1][0].yview_scroll(move_unit, 'units')
        # self.abf_canvas.yview_scroll(move_unit, 'units')

    def change_current_tab(self, event):
        self.current_tab = self.notebook.tk.call(self.notebook._w, "identify", "tab", event.x, event.y)

    def on_close(self):
        self.destroy()


class ResultsSummary(Frame):
    def __init__(self, parent, engine):
        super().__init__(parent)
        self.chart = engine.create_chart(engine.get_cost_df())
        self.chart_container = Frame(self)

        self.chart_container.pack(fill=X, padx=15, pady=15)
        canvas = FigureCanvasTkAgg(self.chart, self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        save_button = MSButton(self.chart_container, text='Save Figure', command=lambda: self.save_file())
        save_button.config(width=0)
        save_button.pack(side=RIGHT, padx=10, pady=10)

    def save_file(self):
        filename = filedialog.asksaveasfilename(defaultextension='.png', initialdir=os.getcwd(),
                                                filetypes=[('PNG', '.png'), ('PDF', '*.pdf'), ('PGF', '*.pgf'),
                                                           ('EPS', '*.eps'), ('PS', '*.ps'), ('Raw', '*.raw'),
                                                           ('RGBA', '*.rgba'), ('SVG', '*.svg'), ('SVGZ', '*.svgz')])
        if filename is None:
            return
        self.chart.savefig(filename)


class ChartContainer(Frame):
    def __init__(self, parent, chart, bg_color):
        super().__init__(parent, bg=bg_color)
        self.chart = chart
        canvas = FigureCanvasTkAgg(chart, self)
        canvas.draw()
        canvas.get_tk_widget().config(height=300)
        canvas.get_tk_widget().pack(side=TOP, fill=X)

        save_button = MSButton(self, text='Save Figure', command=lambda: self.save_file())
        save_button.config(width=0)
        save_button.pack(side=LEFT, padx=10, pady=4)

    def save_file(self):
        filename = filedialog.asksaveasfilename(defaultextension='.png', initialdir=os.getcwd(),
                                                filetypes=[('PNG', '.png'), ('PDF', '*.pdf'), ('PGF', '*.pgf'),
                                                           ('EPS', '*.eps'), ('PS', '*.ps'), ('Raw', '*.raw'),
                                                           ('RGBA', '*.rgba'), ('SVG', '*.svg'), ('SVGZ', '*.svgz')])
        if filename is None:
            return
        self.chart.savefig(filename)

class ABFResultsSummary(Frame):
    def __init__(self, parent, output, dark_bg, light_font):
        super().__init__(parent, bg=dark_bg, highlightcolor='white', highlightthickness=1, pady=8, padx=10)

        self.income_label = Label(self, text='Total Income:', bg=dark_bg, fg=light_font, anchor='e',
                                  font='-size 12 -weight bold')
        self.tax_revenue_label = Label(self, text='Total Tax Revenue:', bg=dark_bg, fg=light_font, anchor='e',
                                       font='-size 12 -weight bold')
        self.benefits_recouped_label = Label(self, text='Tax Revenue Recouped from Benefits:', bg=dark_bg,
                                             fg=light_font, anchor='e', font='-size 12 -weight bold')

        self.income_value = Label(self, bg=light_font, fg=dark_bg, anchor='e', padx=5, font='-size 12')
        self.tax_revenue_value = Label(self, bg=light_font, fg=dark_bg, anchor='e', padx=5, font='-size 12')
        self.benefits_recouped_value = Label(self, bg=light_font, fg=dark_bg, anchor='e', padx=5, font='-size 12')

        self.update_results(output)

        self.income_label.grid(row=0, column=0, sticky='we', padx=3, pady=2)
        self.tax_revenue_label.grid(row=1, column=0, sticky='we', padx=3, pady=2)
        self.benefits_recouped_label.grid(row=2, column=0, sticky='we', padx=3, pady=2)
        self.income_value.grid(row=0, column=1, sticky='we', padx=3, pady=2)
        self.tax_revenue_value.grid(row=1, column=1, sticky='we', padx=3, pady=2)
        self.benefits_recouped_value.grid(row=2, column=1, sticky='we', padx=3, pady=2)

    def update_results(self, output):
        income = '{} (\u00B1{:,.1f}) million'.format(as_currency(output['Total Income (Weighted)'] / 10 ** 6),
                                                     0.5 * (output['Total Income Upper Confidence Interval'] -
                                                            output['Total Income Lower Confidence Interval']) / 10 ** 6)

        tax = '{} (\u00B1{:,.1f}) million'.format(as_currency(output['Total Tax Revenue (Weighted)'] / 10 ** 6),
                                                  0.5 * (output['Total Tax Revenue Upper Confidence Interval'] -
                                                         output[
                                                             'Total Tax Revenue Lower Confidence Interval']) / 10 ** 6)

        benefits_recouped = '{} million'.format(as_currency(output['Tax Revenue Recouped from Benefits'] / 10 ** 6))

        self.income_value.config(text=income)
        self.tax_revenue_value.config(text=tax)
        self.benefits_recouped_value.config(text=benefits_recouped)


class ProgressWindow(Toplevel):
    def __init__(self, parent, se, counterfactual_se=None, policy_sim_se=None):
        super().__init__(parent)
        self.icon = PhotoImage(file='impaq_logo.gif')
        self.tk.call('wm', 'iconphoto', self._w, self.icon)

        self.parent = parent
        self.se = se
        self.counterfactual_se = counterfactual_se
        self.policy_sim_se = policy_sim_se
        self.content = Frame(self, width=100)
        self.content.pack(fill=BOTH, expand=True)
        self.progress = DoubleVar(0)
        self.progress_bar = ttk.Progressbar(self.content, orient=HORIZONTAL, length=100, variable=self.progress,
                                            max=100)
        self.progress_bar.pack(fill=X, padx=10, pady=5)
        self.updates_container = Frame(self.content, height=30, bg=parent.notebook_bg)
        self.updates_canvas = Canvas(self.updates_container, bg=parent.notebook_bg)
        self.updates = Frame(self.updates_container, bg=parent.notebook_bg)
        self.updates_canvas.create_window((0, 0), window=self.updates, anchor='nw')  # Add frame to canvas
        self.updates_info_scroll = ttk.Scrollbar(self.updates_container, orient=VERTICAL,
                                                 command=self.updates_canvas.yview)
        self.updates_canvas.configure(yscrollcommand=self.updates_info_scroll.set)
        # self.abf_info_container.pack(side=TOP, fill=BOTH, expand=True)
        self.updates_container.pack(fill=BOTH, expand=True)
        self.updates_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.updates_info_scroll.pack(side=RIGHT, fill=Y)

        self.bind("<MouseWheel>", self.scroll)
        self.position_window()

    def update_progress(self):
        engines = 1
        if self.counterfactual_se is not None:
            engines += 1

        if self.policy_sim_se is not None:
            engines += 1

        while self.se.progress < 101:
            progress, updates = self.se.get_progress()
            time.sleep(0.5)
            self.update_idletasks()
            if len(updates) > 0:
                self.progress.set(progress / engines)
                self.add_updates(updates)

            if progress == 100:
                break

        if self.policy_sim_se is not None:
            last_progress = self.progress.get()
            self.parent.run_engine(self.policy_sim_se)

            while self.policy_sim_se.progress < 101:
                progress, updates = self.policy_sim_se.get_progress()
                time.sleep(0.5)
                self.update_idletasks()
                if len(updates) > 0:
                    self.progress.set(last_progress + progress / engines)
                    self.add_updates(updates, 'Policy Simulation')

                if progress == 100:
                    break

        if self.counterfactual_se is not None:
            last_progress = self.progress.get()
            self.parent.run_engine(self.counterfactual_se)

            while self.counterfactual_se.progress < 101:
                progress, updates = self.counterfactual_se.get_progress()
                time.sleep(0.5)
                self.update_idletasks()
                if len(updates) > 0:
                    self.progress.set(last_progress + progress / engines)
                    self.add_updates(updates, 'Counterfactual')

                if progress == 100:
                    break

        self.parent.show_results()

    def add_updates(self, updates, engine='Main'):
        for update in updates:
            label = Message(self.updates, text=engine + ': ' + update, bg=self.parent.notebook_bg, fg='#006600',
                            anchor='w', width=350)
            label.pack(padx=3, fill=X)
            self.update()
            self.updates_canvas.configure(scrollregion=(0, 0, 0, self.updates.winfo_height()))

    def scroll(self, event):
        move_unit = 0
        if event.num == 5 or event.delta > 0:
            move_unit = -1
        elif event.num == 4 or event.delta < 0:
            move_unit = 1

        self.updates_canvas.yview_scroll(move_unit, 'units')

    def position_window(self):
        self.update()  # Update changes to root first

        # Get the width and height of both the window and the user's screen
        ww = self.winfo_width()
        wh = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        # Formula for calculating the center
        x = (sw / 2) - (ww / 2)
        y = (sh / 2) - (wh / 2) - 50

        # Set window minimum size
        self.minsize(ww, wh)

        self.geometry('%dx%d+%d+%d' % (ww, wh, x, y))


# From StackOverflow: https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # milliseconds
        self.wraplength = 180  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left', background="#ffffff", relief='solid',
                      borderwidth=1, wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


# The following classes are used so that style options don't have to be reentered for each widget that should be styled
# a certain way.
class MSButton(Button):
    def __init__(self, parent=None, background='#0074BF', font='-size 12', width=8, **kwargs):
        super().__init__(parent, foreground='#FFFFFF', background=background, font=font, width=width,
                         relief='flat', activebackground='#FFFFFF', pady=0, bd=0, **kwargs)


class MSRunButton(Button):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, foreground='#FFFFFF', background='#ccebff', font='-size 11 -weight bold', width=8,
                         relief='flat', activebackground='#FFFFFF', disabledforeground='#FFFFFF', state=DISABLED,
                         **kwargs)


class MSEntry(Entry):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, borderwidth=2, highlightbackground='#FFFFFF', relief='flat', highlightthickness=1,
                         font='-size 11', **kwargs)

# style.configure('Button.border', relief='flat')
# style.configure('Button.label', foreground='#FFFFFF', background='#0074BF', font='-size 11 -weight bold', width=8)

# print(style.layout('TLabelframe'))
# print(style.element_options('TCombobox.border'))
# print(style.element_options('TCombobox.padding'))
# print(style.element_options('TCombobox.textarea'))
# print(style.element_options('TNotebook.client'))
# print(style.element_options('Button.spacing'))
# print(style.element_options('Button.label'))
# print(style.element_options('Button.focus'))

# root.overrideredirect(True)
# top_bar = Frame(root, bg='#FFFFFF')

# close_button = Label(top_bar, text='\u00D7', font='-size 16', bg='#FFFFFF')
# minimize_button = Label(top_bar, text='\u2013', font='-size 14', bg='#FFFFFF')

# top_bar.pack(fill=X)

# close_button.pack(side=RIGHT)
# minimize_button.pack(side=RIGHT, anchor='ne')
