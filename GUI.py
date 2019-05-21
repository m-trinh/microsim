from tkinter import *
from tkinter import ttk, filedialog, messagebox
import os
import sys
from simulator._5_simulation_engine import SimulationEngine
from Utils import Settings, as_currency
import collections
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ABF import ABF
import pandas as pd


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
                                   'Support Vector Machine', 'Random Forest', 'Stochastic Gradient Descent')
        self.dark_bg = '#333333'
        self.light_font = '#f2f2f2'
        self.notebook_bg = '#fcfcfc'
        self.theme_color = '#0074BF'
        self.default_settings = Settings()
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
        self.acs_file = StringVar()
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

        # When the file location entries are modified, check to see if they all have some value
        # If they do, enable the run button
        if version[1] < 6:
            self.fmla_file.trace("w", self.check_file_entries)
            self.acs_file.trace("w", self.check_file_entries)
            self.output_directory.trace("w", self.check_file_entries)
        else:
            self.fmla_file.trace_add("write", self.check_file_entries)
            self.acs_file.trace_add("write", self.check_file_entries)
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

        self.acs_label = Label(self.main_frame, text="ACS File:", bg=self.dark_bg, fg=self.light_font)
        self.acs_input = MSEntry(self.main_frame, textvariable=self.acs_file)
        self.acs_button = MSButton(self.main_frame, text="Browse", command=lambda: self.browse_file(self.acs_input))
        CreateToolTip(self.acs_label,
                      'A CSV or Excel file that contains population data that the model will use to estimate '
                      'the cost of a paid leave program. This should be ACS data.')

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
        self.state_input.current(self.states.index('MA'))
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

        # This code adds padding to each row. This is needed when using grid() to add widgets.
        self.row_padding = 8
        for i in range(6):
            self.main_frame.rowconfigure(i, pad=self.row_padding)
        for i in range(11):
            self.program_frame.rowconfigure(i, pad=self.row_padding)
        for i in range(7):
            self.population_frame.rowconfigure(i, pad=self.row_padding)
        for i in range(6):
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
        self.fmla_file.set('C:/Users/mtrinh/PycharmProjects/MicrosimGUI/data/fmla_2012/fmla_clean_2012_resp_length.csv')
        self.acs_file.set('C:/Users/mtrinh/PycharmProjects/MicrosimGUI/data/acs/ACS_cleaned_forsimulation_ma_small.csv')
        self.output_directory.set('C:/Users/mtrinh/PycharmProjects/MicrosimGUI/output')
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
        st = settings.state.lower()
        fullFp_acs, fullFp_fmla, fullFp_out = settings.acs_file, settings.fmla_file, settings.output_directory
        fp_fmla = '.'+fullFp_fmla[fullFp_fmla.find('/data/fmla_2012/'):]
        print(fp_fmla)
        fp_acs = '.'+fullFp_acs[fullFp_acs.find('/data/acs/'):]
        fp_out = fullFp_out
        clf_name = settings.simulation_method
        rr1 = settings.replacement_ratio
        hrs = settings.eligible_hours
        max_wk = settings.max_weeks  # this returns a dict

        max_wk['own'] = max_wk.pop('Own Health')
        max_wk['matdis'] = max_wk.pop('Maternity')
        max_wk['bond'] = max_wk.pop('New Child')
        max_wk['illchild'] = max_wk.pop('Ill Child')
        max_wk['illspouse'] = max_wk.pop('Ill Spouse')
        max_wk['illparent'] = max_wk.pop('Ill Parent')
        max_wk_replace = settings.weekly_ben_cap

        empgov = settings.government_employees
        empself = settings.self_employed

        se = SimulationEngine(st, fp_acs, fp_fmla, fp_out, clf_name, rr1, hrs, max_wk, max_wk_replace, empgov,
                              empself)
        # compute program costs
        acs = se.get_acs_simulated()
        d_type = se.d_type  # get a dict from varname style leave types to user-readable leave types
        takeups = settings.take_up_rates  # this returns a dict of user input of takeup in GUI
        for k, v in d_type.items():
            takeups[k] = takeups.pop(v)  # convert readable types to varname types
        costs = se.get_cost(acs, takeups)

        d_bars = {'Own Health': costs['Own Health'],
                  'Maternity': costs['Maternity'],
                  'New Child': costs['New Child'],
                  'Ill Child': costs['Ill Child'],
                  'Ill Spouse': costs['Ill Spouse'],
                  'Ill Parent': costs['Ill Parent']}

        ks = ['Own Health', 'Maternity', 'New Child', 'Ill Child', 'Ill Spouse', 'Ill Parent']
        od_bars = collections.OrderedDict((k, d_bars[k]) for k in ks)

        total_benefits = sum([costs[leave_type] for leave_type in ks])
        self.abf_module = ABF(settings, total_benefits * 10**6)
        abf_output, pivot_tables = self.abf_module.run()

        self.results_window = ResultsWindow(self, od_bars, abf_output, pivot_tables)

    def run_abf(self):
        settings = self.create_settings()
        print(vars(settings))

        abf_output, pivot_tables = self.abf_module.rerun(settings)
        self.results_window.update_abf_output(abf_output)
        self.results_window.update_pivot_tables(pivot_tables)

    def test_result_output(self):
        # This code currently generates a mock up of a results window
        simulation_results = {'Own Health': 10, 'Maternity': 200, 'New Child': 10, 'Ill Child': 8,
                                'Ill Spouse': 5, 'Ill Parent': 0}
        abf_output = {'Total Income (Weighted)': 0, 'Total Income': 0,
                      'Income Standard Error': 0, 'Total Income Upper Confidence Interval': 0,
                      'Total Income Lower Confidence Interval': 0,
                      'Total Tax Revenue': 0,
                      'Tax Standard Error': 0, 'Total Tax Revenue Upper Confidence Interval': 0,
                      'Total Tax Revenue Lower Confidence Interval': 0, 'Tax Revenue Recouped from Benefits': 0}

        pivot_table = pd.DataFrame({('sum', 'income_w'): 0, ('sum', 'ptax_rev_w'): 0},
                                   index=['Private', 'Federal', 'State', 'Local'])

        pivot_tables = {'class': pivot_table, 'age': pivot_table, 'gender': pivot_table}

        self.results_window = ResultsWindow(self, simulation_results, abf_output, pivot_tables)

    # Create an object with all of the setting values
    def create_settings(self):
        # The inputs are linked to a tkinter variable. Those values will have to be retrieved from each variable
        # and passed on to the settings objects
        return Settings(self.fmla_file.get(), self.acs_file.get(), self.output_directory.get(), self.detail.get(),
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
                        self.max_taxable_earnings_per_person.get(), self.total_taxable_earnings_input.get())

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
        if self.fmla_file.get() and self.acs_file.get() and self.output_directory.get():
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
    def __init__(self, parent, simulation_output, abf_output, pivot_tables):
        super().__init__(parent)
        self.dark_bg = parent.dark_bg
        self.notebook_bg = parent.notebook_bg
        self.light_font = parent.light_font

        self.content = Frame(self, bg=self.dark_bg)
        self.notebook = ttk.Notebook(self.content, style='MSNotebook.TNotebook')

        self.summary_frame = Frame(self.notebook, bg=self.dark_bg)
        self.notebook.add(self.summary_frame, text='Summary')

        self.abf = Frame(self.notebook, bg=self.notebook_bg)
        self.abf_info_container = Frame(self.abf, bg=self.dark_bg)
        self.abf_canvas = Canvas(self.abf_info_container, bg=self.dark_bg)
        self.abf_info = Frame(self.abf_info_container, bg=self.dark_bg)
        self.abf_canvas.create_window((2, 2), window=self.abf_info, anchor='nw')  # Add frame to canvas
        self.abf_info_scroll = ttk.Scrollbar(self.abf_info_container, orient=VERTICAL,
                                             command=self.abf_canvas.yview)
        self.abf_canvas.configure(yscrollcommand=self.abf_info_scroll.set)
        self.abf_info_container.pack(side=TOP, fill=BOTH, expand=True)
        self.abf_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=0, pady=0)
        self.abf_info_scroll.pack(side=RIGHT, fill=Y)
        self.abf_summary = ABFResultsSummary(self.abf_info, abf_output, self.dark_bg, self.light_font)
        self.abf_summary.pack(fill=X)
        self.abf_pivot_tables = Frame(self.abf_info, bg=self.dark_bg)
        self.abf_pivot_tables.pack(fill=X)
        self.abf_params = Frame(self.abf, bg=self.notebook_bg, borderwidth=1, relief='solid')
        self.abf_params.pack(fill=X, side=BOTTOM, padx=2, pady=2)
        self.abf_params_inputs = Frame(self.abf_params, bg=self.notebook_bg, pady=4)
        self.abf_params_inputs.pack(fill=X, side=TOP)
        self.run_button = MSButton(self.abf_params, text="Run ABF", command=parent.run_abf, padx=4)
        self.run_button.pack(side=BOTTOM, anchor='center', pady=(5, 2))
        self.notebook.add(self.abf, text="Benefits Financing")

        self.cost_result_label = Label(self.summary_frame,
                                       text="Cost: ${} Million".format(round(sum(simulation_output.values()))),
                                       font='-size 14', bg=self.dark_bg, fg=self.light_font, justify=LEFT)
        self.state_result_label = Label(self.summary_frame, text="State: {}".format(parent.state.get()),
                                        font='-size 12', bg=self.dark_bg, fg=self.light_font)
        self.result_graph = Frame(self.summary_frame)

        self.content.pack(fill=BOTH)
        self.notebook.pack(expand=True, fill=BOTH)
        self.notebook.select(self.summary_frame)
        self.notebook.enable_traversal()

        self.cost_result_label.pack(fill=X, pady=(10, 5))
        self.state_result_label.pack(fill=X)
        self.result_graph.pack(fill=X, padx=15, pady=15)

        self.display_bar_graph(simulation_output, self.result_graph)

        self.display_abf_pivot_table(pivot_tables)

        self.abf_canvas.configure(scrollregion=(0, 0, 0, 940))
        self.abf_canvas.bind("<MouseWheel>", self.scroll_abf)

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

    def display_bar_graph(self, data, frame):
        figure = Figure(figsize=(8, 6), dpi=100)
        index = range(len(data.keys()))
        graph = figure.add_subplot(111, ylabel='Cost')
        graph.bar(index, data.values(), tick_label=list(data.keys()))
        graph.set_xlabel('Leave Type', fontsize=14, labelpad=10)
        graph.set_ylabel('Cost', fontsize=14, labelpad=10)
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        save_button = MSButton(frame, text='Save Figure', command=lambda: self.save_file(figure))
        save_button.config(width=0)
        save_button.pack(side=RIGHT, padx=10, pady=10)

    def save_file(self, figure):
        filename = filedialog.asksaveasfilename(defaultextension='.png', initialdir=os.getcwd(),
                                                filetypes=[('PNG', '.png'), ('PDF', '*.pdf'), ('PGF', '*.pgf'),
                                                           ('EPS', '*.eps'), ('PS', '*.ps'), ('Raw', '*.raw'),
                                                           ('RGBA', '*.rgba'), ('SVG', '*.svg'), ('SVGZ', '*.svgz')])
        if filename is None:
            return

        figure.savefig(filename)

    def display_abf_pivot_table(self, pivot_tables):
        for pivot_table in pivot_tables.values():
            pt_frame = Frame(self.abf_pivot_tables, bg=self.light_font, padx=5, pady=2)
            income_w_label = Label(pt_frame, text="Income (Weighted)", bg=self.light_font, fg=self.dark_bg,
                                   font='-weight bold')
            tax_rev_w_label = Label(pt_frame, text="Tax Revenue (Weighted)", bg=self.light_font, fg=self.dark_bg,
                                    font='-weight bold')
            income_w_label.grid(column=1, row=0, sticky='we', padx=4, pady=2)
            tax_rev_w_label.grid(column=2, row=0, sticky='we', padx=4, pady=2)
            index_values = list(pivot_table.index.values)
            for i, index in enumerate(index_values):
                index_label = Label(pt_frame, text=index, bg=self.light_font, fg=self.dark_bg)
                cat_income_w_label = Label(pt_frame, text=as_currency(pivot_table.loc[index, ('sum', 'income_w')]),
                                           bg=self.light_font, fg=self.dark_bg, borderwidth=1, relief='solid',
                                           justify=RIGHT, padx=4, pady=2, width=20, anchor=E)
                cat_tax_rev_w_label = Label(pt_frame, text=as_currency(pivot_table.loc[index, ('sum', 'ptax_rev_w')]),
                                            bg=self.light_font, fg=self.dark_bg, borderwidth=1, relief='solid',
                                            justify=RIGHT, padx=4, pady=2, width=20, anchor=E)
                index_label.grid(column=0, row=i + 1, sticky='w', padx=4, pady=2)
                cat_income_w_label.grid(column=1, row=i + 1)
                cat_tax_rev_w_label.grid(column=2, row=i + 1)
            pt_frame.pack(side=BOTTOM, anchor='w', padx=8, pady=5)

    def update_abf_output(self, abf_output):
        self.abf_summary.update_results(abf_output)

    def update_pivot_tables(self, pivot_tables):
        for pivot_table in self.abf_pivot_tables.winfo_children():
            pivot_table.destroy()

        self.display_abf_pivot_table(pivot_tables)

    def scroll_abf(self, event):
        move_unit = 0
        if event.num == 5 or event.delta > 0:
            move_unit = -1
        elif event.num == 4 or event.delta < 0:
            move_unit = 1

        self.abf_canvas.yview_scroll(move_unit, 'units')


class ABFResultsSummary(Frame):
    def __init__(self, parent, output, dark_bg, light_font):
        super().__init__(parent, bg=dark_bg)

        self.components = {}
        idx = 0
        for key, value in output.items():
            component_label = Label(self, text=key + ':', bg=dark_bg, fg=light_font, anchor='e')
            component_value = Label(self, text=as_currency(value), bg=light_font, fg=dark_bg, anchor='e', padx=5)

            self.components[key] = {'label': component_label, 'value': component_value}

            component_label.grid(column=0, row=idx, sticky='we', padx=3, pady=2)
            component_value.grid(column=1, row=idx, sticky='we', padx=3, pady=2)
            idx += 1

    def update_results(self, output):
        for key, value in output.items():
            self.components[key]['value'].config(text=as_currency(value))


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
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, foreground='#FFFFFF', background='#0074BF', font='-size 12', width=8,
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
