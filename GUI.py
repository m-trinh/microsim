from tkinter import *
from tkinter import ttk, filedialog, messagebox
import os
import sys
import multiprocessing
import ast
import queue
import numpy as np
from abc import ABCMeta, abstractmethod
from _5_simulation import SimulationEngine
from Utils import *
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ABF import ABF


DARK_COLOR = '#333333'
LIGHT_COLOR = '#f2f2f2'
VERY_LIGHT_COLOR = '#fcfcfc'
THEME_COLOR = '#0074BF'


class MicrosimGUI(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Remove file locations
        self.general_settings = GeneralSettings(fmla_file='./data/fmla_2012/fmla_2012_employee_revised_puf.csv',
                                                acs_directory='./data/acs', output_directory='./output',
                                                r_path='/Users/mtrinh/R-3.6.1/bin/Rscript.exe', state='All')
        self.default_settings = OtherSettings()
        self.all_settings = [self.default_settings]
        self.comparing = False
        self.current_sim_num = 0
        self.error_tooltips = []
        # Set the current visible tab to 0, which is the Program tab
        self.current_tab = 0
        self.existing_programs = ['', 'CA', 'NJ', 'RI']
        self.variables = self.__create_variables()
        self.__set_up_style()

        self.title('Paid Leave Micro-Simulator')  # Add title to window
        self.option_add('*Font', '-size 12')  # Set default font
        # self.resizable(False, False)  # Prevent window from being resized
        self.bind("<MouseWheel>", self.scroll)  # Bind mouse wheel action to scroll function
        self.icon = PhotoImage(file='impaq_logo.gif')
        self.tk.call('wm', 'iconphoto', self._w, self.icon)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # The content frame will hold all widgets
        self.content = Frame(self, padx=15, pady=15, bg=DARK_COLOR)
        # This frame holds general settings
        self.general_settings_frame = GeneralSettingsFrame(self.content, bg=DARK_COLOR)
        # This frame holds buttons for comparing program parameters
        self.simulation_comparison = ComparisonFrame(self.content, bg=DARK_COLOR)
        # This notebook will have three tabs for the program, population, and simulation settings
        self.settings_notebook = SettingsNotebook(self.content)
        self.showing_advanced = False
        self.advanced_frame = AdvancedFrame(self.content, self.toggle_advanced_parameters)
        self.run_button = RunButton(self.content, text="Run", height=1, command=self.__run_simulation)

        # Add callbacks that will run when certain variables are changed
        self.__add_variable_callbacks()

        # ----------------------------------------- Add Widgets to Window --------------------------------------------

        self.content.pack(expand=True, fill=BOTH)
        self.general_settings_frame.pack(fill=X)
        self.simulation_comparison.pack(fill=X, pady=(4, 0))
        self.settings_notebook.pack(expand=True, fill=BOTH, pady=(4, 8))
        self.advanced_frame.pack(anchor=E, pady=(0, 6))
        self.run_button.pack(anchor=E, fill=Y)

        self.position_window()
        self.original_height = self.winfo_height()

        self.abf_module = None
        self.results_windows = []
        self.progress_windows = []
        self.check_file_entries()

    def __set_up_style(self):
        # Edit the style for ttk widgets. These new styles are given their own names, which will have to be provided
        # by the widgets in order to be used.
        style = ttk.Style()
        style.configure('MSCombobox.TCombobox', relief='flat')
        style.configure('MSCheckbutton.TCheckbutton', background=VERY_LIGHT_COLOR, font='-size 12')
        style.configure('DarkCheckbutton.TCheckbutton', background=DARK_COLOR, foreground=LIGHT_COLOR, font='-size 12')
        style.configure('MSNotebook.TNotebook', background=VERY_LIGHT_COLOR)
        style.configure('MSNotebook.TNotebook.Tab', font='-size 12', padding=(4, 0))
        style.configure('MSLabelframe.TLabelframe', background=VERY_LIGHT_COLOR)
        style.configure('MSLabelframe.TLabelframe.Label', background=VERY_LIGHT_COLOR, foreground=THEME_COLOR,
                        font='-size 12')

    def __create_variables(self):
        # These are the variables that the users will update. These will be passed to the engine.
        g = self.general_settings
        d = self.default_settings
        variables = {
            'fmla_file': StringVar(value=g.fmla_file),
            'acs_directory': StringVar(value=g.acs_directory),
            'output_directory': StringVar(value=g.output_directory),
            'state': StringVar(value=g.state),
            'simulation_method': StringVar(value=g.simulation_method),
            'existing_program': StringVar(),
            'engine_type': StringVar(value=g.engine_type),
            'r_path': StringVar(),
            'benefit_effect': BooleanVar(value=d.benefit_effect),
            'calibrate': BooleanVar(value=d.calibrate),
            'clone_factor': IntVar(value=d.clone_factor),
            'se_analysis': BooleanVar(value=d.se_analysis),
            'extend': BooleanVar(value=d.extend),
            'fmla_protection_constraint': BooleanVar(value=d.fmla_protection_constraint),
            'replacement_ratio': DoubleVar(value=d.replacement_ratio),
            'government_employees': BooleanVar(value=d.government_employees),
            'fed_employees': BooleanVar(value=d.fed_employees),
            'state_employees': BooleanVar(value=d.state_employees),
            'local_employees': BooleanVar(value=d.local_employees),
            'needers_fully_participate': BooleanVar(value=d.needers_fully_participate),
            'random_seed': StringVar(value=g.random_seed),
            'self_employed': BooleanVar(value=d.self_employed),
            'state_of_work': BooleanVar(value=g.state_of_work),
            'top_off_rate': DoubleVar(value=d.top_off_rate),
            'top_off_min_length': IntVar(value=d.top_off_min_length),
            'weekly_ben_cap': IntVar(value=d.weekly_ben_cap),
            'weight_factor': IntVar(value=d.weight_factor),
            'eligible_earnings': IntVar(value=d.eligible_earnings),
            'eligible_weeks': IntVar(value=d.eligible_weeks),
            'eligible_hours': IntVar(value=d.eligible_hours),
            'eligible_size': IntVar(value=d.eligible_size),
            'payroll_tax': DoubleVar(value=d.payroll_tax),
            'benefits_tax': BooleanVar(value=d.benefits_tax),
            'average_state_tax': DoubleVar(value=d.average_state_tax),
            'max_taxable_earnings_per_person': IntVar(value=d.max_taxable_earnings_per_person),
            'total_taxable_earnings': IntVar(value=d.total_taxable_earnings),
            'counterfactual': StringVar(),
            'policy_sim': BooleanVar(value=d.policy_sim),
            'dual_receivers_share': DoubleVar(value=d.dual_receivers_share),
            'max_weeks': {leave_type: IntVar(value=d.max_weeks[leave_type]) for leave_type in LEAVE_TYPES},
            'take_up_rates': {leave_type: DoubleVar(value=d.take_up_rates[leave_type]) for leave_type in LEAVE_TYPES},
            'leave_probability_factors': {leave_type: DoubleVar(value=d.leave_probability_factors[leave_type])
                                          for leave_type in LEAVE_TYPES},
        }

        return variables

    def __add_variable_callbacks(self):
        # When the file location entries are modified, check to see if they all have some value
        # If they do, enable the run button
        if sys.version_info[1] < 6:
            self.variables['fmla_file'].trace("w", self.check_file_entries)
            self.variables['acs_directory'].trace("w", self.check_file_entries)
            self.variables['output_directory'].trace("w", self.check_file_entries)
        else:
            self.variables['fmla_file'].trace_add("write", self.check_file_entries)
            self.variables['acs_directory'].trace_add("write", self.check_file_entries)
            self.variables['output_directory'].trace_add("write", self.check_file_entries)

        # When users change the existing_program variable, change all parameters to match an existing state program
        self.variables['existing_program'].trace('w', self.set_existing_parameters)

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
        for w in self.progress_windows:
            w.quit()
            w.destroy()
        for w in self.results_windows:
            w.quit()
            w.destroy()
        self.quit()
        self.destroy()

    def set_existing_parameters(self, *_):
        # Change all relevant parameters to match an existing state program
        state = self.variables['existing_program'].get().upper()
        if state not in DEFAULT_STATE_PARAMS or state == '':
            return
        state_params = DEFAULT_STATE_PARAMS[state]

        for param_key, param_val in state_params.items():
            # If value for the parameter is a dictionary, then traverse that dictionary
            if type(param_val) == dict:
                for k, v in param_val.items():
                    self.variables[param_key][k].set(v)
            else:
                self.variables[param_key].set(param_val)

    def __run_simulation(self):
        self.__clear_errors()
        errors = self.validate_settings()

        if len(errors) > 0:
            self.display_errors(errors)
            return

        self.save_general_settings()
        self.save_settings()
        if self.general_settings.engine_type == 'Python':
            self.__run_simulation_python()
        elif self.general_settings.engine_type == 'R':
            self.__run_simulation_r(self.general_settings)

    def __run_simulation_python(self):
        # initiate a SimulationEngine instance
        q = multiprocessing.Queue()
        self.se = self.create_simulation_engine(q)
        for settings in self.all_settings:
            self.add_engine_params(settings)
            if not self.comparing:
                break

        self.run_button.disable()
        progress_window = ProgressWindow(self, engine_type='Python', se=self.se)
        self.progress_windows.append(progress_window)
        # Run model
        self.engine_process = multiprocessing.Process(None, target=run_engines, args=(self.se, q))
        self.engine_process.start()

        progress_window.update_progress(q)

    def __run_simulation_r(self, settings):
        progress_file = './log/progress_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
        command = create_r_command(settings, progress_file)
        counterfactual_command = create_r_command(generate_default_state_params(settings), progress_file)
        policy_command = create_r_command(generate_generous_params(settings), progress_file)
        open(progress_file, 'w+').close()
        self.engine_process = multiprocessing.Process(None, target=run_engines_r, args=(command, counterfactual_command,
                                                                                        policy_command))
        self.engine_process.start()

        progress_window = ProgressWindow(self, engine_type='R')
        with open(progress_file, 'r') as f:
            progress_window.update_progress_r(f)

    def show_results(self):
        self.engine_process.terminate()
        # compute program costs
        print('Showing results')
        costs = self.se.get_cost_df(0)

        total_benefits = list(costs.loc[costs['type'] == 'total', 'cost'])[0]
        abf_module = ABF(self.se.get_results(0), self.all_settings[0], total_benefits)

        self.results_windows.append(ResultsWindow(self, self.se, abf_module))
        self.run_button.enable()

    def create_settings(self):
        return self.__create_settings()

    # Create an object with all of the setting values
    def __create_settings(self):
        # The inputs are linked to a tkinter variable. Those values will have to be retrieved from each variable
        # and passed on to the settings objects
        variable_values = {}
        valid_var_names = vars(self.default_settings).keys()
        for var_name, var_obj in self.variables.items():
            if var_name not in valid_var_names:
                continue

            if type(var_obj) == dict:
                variable_values[var_name] = {k: v.get() for k, v in var_obj.items()}
            else:
                variable_values[var_name] = var_obj.get()

        return OtherSettings(**variable_values)

    def start_comparing(self):
        self.comparing = True

    def stop_comparing(self):
        self.comparing = False
        self.switch_comparison(0)

    def __set_current_sim_num(self, sim_num):
        self.current_sim_num = sim_num

    def add_comparison(self):
        self.all_settings.append(self.default_settings)

    def remove_comparison(self, sim_num):
        del self.all_settings[sim_num]
        self.switch_comparison(sim_num - 1, save=False)

    def switch_comparison(self, sim_num, save=True):
        if save:
            self.save_settings()
        self.__set_current_sim_num(sim_num)
        self.change_comparison_parameters(sim_num)

    def change_comparison_parameters(self, sim_num):
        settings = self.all_settings[sim_num]
        for param_key, param_val in vars(settings).items():
            # If value for the parameter is a dictionary, then traverse that dictionary
            if type(param_val) == dict:
                for k, v in param_val.items():
                    self.variables[param_key][k].set(v)
            else:
                self.variables[param_key].set(param_val)

    def save_general_settings(self):
        variable_values = {}
        for var_name in vars(self.general_settings).keys():
            variable_values[var_name] = self.variables[var_name].get()
        variable_values['random_seed'] = self.check_random_seed(variable_values['random_seed'])

        self.general_settings.update_variables(**variable_values)

    def save_settings(self):
        settings = self.__create_settings()
        self.all_settings[self.current_sim_num] = settings
        return settings

    @staticmethod
    def check_random_seed(random_seed):
        if random_seed is None or random_seed == '':
            return None

        try:
            return int(random_seed)
        except ValueError:
            return int.from_bytes(random_seed.encode(), 'big')

    def create_simulation_engine(self, q):
        st = self.general_settings.state.lower()
        yr = 16
        fp_fmla_in = self.general_settings.fmla_file
        fp_cps_in = './data/cps/CPS2014extract.csv'
        fp_acsh_in = self.general_settings.acs_directory + '/household_files'
        fp_acsp_in = self.general_settings.acs_directory + '/person_files'
        state_of_work = self.general_settings.state_of_work
        if state_of_work:
            fp_acsh_in = self.general_settings.acs_directory + '/pow_household_files'
            fp_acsp_in = self.general_settings.acs_directory + '/pow_person_files'
        fp_fmla_out = './data/fmla_2012/fmla_clean_2012.csv'
        fp_cps_out = './data/cps/cps_for_acs_sim.csv'
        fp_acs_out = './data/acs/'
        fp_length_distribution_out = './data/fmla_2012/length_distributions_exact_days.json'
        fps_in = [fp_fmla_in, fp_cps_in, fp_acsh_in, fp_acsp_in]
        fps_out = [fp_fmla_out, fp_cps_out, fp_acs_out, fp_length_distribution_out]

        clf_name = self.general_settings.simulation_method
        random_seed = self.general_settings.random_seed
        sim_method = self.general_settings.simulation_method
        return SimulationEngine(st, yr, fps_in, fps_out, clf_name=clf_name, random_state=random_seed,
                                state_of_work=state_of_work, sim_method=sim_method, q=q)

    def add_engine_params(self, settings):
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
        needers_fully_participate = settings.needers_fully_participate
        # state_of_work value see above next to fp_acsh_in/fp_acsp_in
        # weight_factor = settings.weight_factor
        clone_factor = settings.clone_factor
        dual_receivers_share = settings.dual_receivers_share

        self.se.set_simulation_params(elig_wage12, elig_wkswork, elig_yrhours, elig_empsize, rrp, wkbene_cap, d_maxwk,
                                      d_takeup, incl_empgov_fed, incl_empgov_st, incl_empgov_loc, incl_empself,
                                      needers_fully_participate, clone_factor, dual_receivers_share, sim_num=None)

    def check_file_entries(self, *_):
        if self.variables['fmla_file'].get() and self.variables['acs_directory'].get() and \
                self.variables['output_directory'].get():
            self.run_button.enable()
        else:
            self.run_button.disable()

    # TODO: Fix parameter validation
    def validate_settings(self):
        errors = []

        integer_entries = [self.settings_notebook.program_frame.eligible_earnings_input,
                           self.settings_notebook.program_frame.eligible_weeks_input,
                           self.settings_notebook.program_frame.eligible_hours_input,
                           self.settings_notebook.program_frame.eligible_size_input,
                           self.settings_notebook.program_frame.weekly_ben_cap_input,
                           self.settings_notebook.population_frame.top_off_min_length_input,
                           self.settings_notebook.simulation_frame.clone_factor_input,
                           self.settings_notebook.program_frame.max_taxable_earnings_per_person_input,
                           ]  # self.settings_notebook.program_frame.total_taxable_earnings_input
        # self.settings_notebook.simulation_frame.weight_factor_input
        integer_entries += [entry for entry in self.settings_notebook.program_frame.max_weeks_inputs]

        float_entries = [self.settings_notebook.program_frame.payroll_tax_input,
                         self.settings_notebook.program_frame.average_state_tax_input,
                         ]

        rate_entries = [self.settings_notebook.program_frame.replacement_ratio_input,
                        self.settings_notebook.population_frame.top_off_rate_input]
        rate_entries += [entry for entry in self.settings_notebook.population_frame.take_up_rates_inputs]
        rate_entries += [entry for entry in self.settings_notebook.population_frame.leave_probability_factors_inputs]

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

    def __clear_errors(self):
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
            move_unit = -2
        elif event.num == 4 or event.delta < 0:
            move_unit = 2

        # Only scroll the tab that is currently visible.
        if self.current_tab == 0:
            self.settings_notebook.program_frame.canvas.yview_scroll(move_unit, 'units')
        elif self.current_tab == 1:
            self.settings_notebook.population_frame.canvas.yview_scroll(move_unit, 'units')
        elif self.current_tab == 2:
            self.settings_notebook.simulation_frame.canvas.yview_scroll(move_unit, 'units')

    # Change the currently visible tab.
    def change_current_tab(self, event):
        self.current_tab = self.settings_notebook.tk.call(self.settings_notebook._w, "identify", "tab", event.x,
                                                          event.y)

    def hide_advanced_parameters(self):
        self.general_settings_frame.hide_advanced_parameters()
        self.settings_notebook.hide_advanced_parameters()

    def show_advanced_parameters(self):
        self.general_settings_frame.show_advanced_parameters()
        self.settings_notebook.show_advanced_parameters()

    def toggle_advanced_parameters(self):
        height_change = 125
        if self.showing_advanced:
            self.showing_advanced = False
            self.hide_advanced_parameters()
            self.update()
            self.minsize(self.winfo_width(), self.original_height)
        else:
            self.showing_advanced = True
            self.show_advanced_parameters()
            self.update()
            self.minsize(self.winfo_width(), self.original_height + height_change)

        self.advanced_frame.on_button.toggle()
        self.advanced_frame.off_button.toggle()


class GeneralSettingsFrame(Frame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__create_attributes()

        # Create the input widgets for general parameters
        # ------------------------------------------------ FMLA File ------------------------------------------------
        tip = 'A CSV or Excel file that contains leave taking data to use to train model. ' \
              'This should be FMLA survey data.'
        self.fmla_label = TipLabel(self, tip, text="FMLA File:", bg=DARK_COLOR, fg=LIGHT_COLOR, anchor=N)
        self.fmla_input = GeneralEntry(self, textvariable=self.variables['fmla_file'])
        self.fmla_button = BorderButton(self, text="Browse",
                                        command=lambda: self.browse_file(self.fmla_input, self.spreadsheet_ftypes))
        self.fmla_button.config(width=None)

        # ------------------------------------------------ ACS File -------------------------------------------------
        tip = 'A directory that contains ACS files that the model will use to estimate the cost of a paid ' \
              'leave program. There should be one household and one person file for the selected state.'
        self.acs_label = TipLabel(self, tip, text="ACS Directory:", bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.acs_input = GeneralEntry(self, textvariable=self.variables['acs_directory'])
        self.acs_button = BorderButton(self, text="Browse",
                                       command=lambda: self.browse_directory(self.acs_input))

        # -------------------------------------------- Output Directory ---------------------------------------------
        tip = 'The directory where the spreadsheet containing simulation results will be saved.'
        self.output_directory_label = TipLabel(self, tip, text="Output Directory:", bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.output_directory_input = GeneralEntry(self, textvariable=self.variables['output_directory'])
        self.output_directory_button = BorderButton(self, text="Browse",
                                                    command=lambda: self.browse_directory(self.output_directory_input))

        # ---------------------------------------------- Output Detail ----------------------------------------------
        # tip = 'The level of detail of the results. \n1 = low detail \n8 = high detail'
        # self.detail_label = TipLabel(self, tip, text="Output Detail Level:", bg=DARK_COLOR, fg=LIGHT_COLOR)
        # self.detail_input = ttk.Combobox(self, textvariable=self.variables['detail'], state="readonly", width=5,
        #                                  style='MSCombobox.TCombobox')
        # self.detail_input['values'] = (1, 2, 3, 4, 5, 6, 7, 8)
        # self.detail_input.current(0)

        # -------------------------------------------- State to Simulate --------------------------------------------
        tip = 'The state that will be used to estimate program cost. Only people  living or working in ' \
              'this state will be chosen from the input and  output files.'
        self.state_label = TipLabel(self, tip, text='State to Simulate:', bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.state_input = ttk.Combobox(self, textvariable=self.variables['state'], state="readonly", width=5,
                                        values=self.states)
        self.state_input.current(self.states.index('RI'))

        # ---------------------------------------------- State of Work ----------------------------------------------
        tip = 'Whether or not the analysis is to be done for persons who work in particular state – ' \
              'rather than for residents of the state.'
        self.state_of_work_input = TipCheckButton(self, tip, text="State of Work",
                                                  style='DarkCheckbutton.TCheckbutton',
                                                  variable=self.variables['state_of_work'])

        # -------------------------------------------- Simulation Method --------------------------------------------
        tip = 'The method used to train model.'
        self.simulation_method_label = TipLabel(self, tip, text='Simulation Method:', bg=DARK_COLOR,
                                                fg=LIGHT_COLOR)
        self.simulation_method_input = ttk.Combobox(self, textvariable=self.variables['simulation_method'],
                                                    state="readonly", width=21, values=self.simulation_methods)
        self.simulation_method_input.current(0)

        # ----------------------------------------------- Random Seed -----------------------------------------------
        tip = 'The value that will be used in random number generation. Can be used to recreate results as long ' \
              'as all other parameters are unchanged.'
        self.random_seed_label = TipLabel(self, tip, text="Random Seed:", bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.random_seed_input = GeneralEntry(self, textvariable=self.variables['random_seed'])

        # ----------------------------------------------- Engine Type -----------------------------------------------
        tip = 'Choose between the Python and R model.'
        self.engine_type_label = TipLabel(self, tip, text='Engine Type:', bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.engine_type_input = ttk.Combobox(self, textvariable=self.variables['engine_type'], state="readonly",
                                              width=7, values=['Python', 'R'])
        self.engine_type_input.current(0)

        tip = 'The Rscript path on your system.'
        self.r_path_label = TipLabel(self, tip, text="Rscript Path:", bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.r_path_input = GeneralEntry(self, textvariable=self.variables['r_path'])
        self.r_path_button = BorderButton(self, text="Browse",
                                          command=lambda: self.browse_file(self.r_path_input, [('Rscript', 'Rscript.exe')]))
        self.variables['engine_type'].trace('w', self.toggle_r_path)

        # Add the input widgets to the parent widget
        self.row_padding = 4
        self.fmla_label.grid(column=0, row=0, sticky=W, pady=self.row_padding)
        self.fmla_input.grid(column=1, row=0, columnspan=3, sticky=(E, W), padx=8, pady=self.row_padding)
        self.fmla_button.grid(column=4, row=0, sticky=W, pady=self.row_padding)
        self.acs_label.grid(column=0, row=1, sticky=W, pady=self.row_padding)
        self.acs_input.grid(column=1, row=1, columnspan=3, sticky=(E, W), padx=8, pady=self.row_padding)
        self.acs_button.grid(column=4, row=1, pady=self.row_padding)
        self.output_directory_label.grid(column=0, row=2, sticky=W, pady=self.row_padding)
        self.output_directory_input.grid(column=1, row=2, columnspan=3, sticky=(E, W), padx=8, pady=self.row_padding)
        self.output_directory_button.grid(column=4, row=2, pady=self.row_padding)
        self.state_label.grid(column=0, row=4, sticky=W, pady=(self.row_padding, 2))
        self.state_input.grid(column=1, row=4, sticky=W, padx=8, pady=(self.row_padding, 2))

        # Give the second column more space than the others
        self.columnconfigure(1, weight=1)

    def __create_attributes(self):
        self.spreadsheet_ftypes = [('All', '*.xlsx; *.xls; *.csv'), ('Excel', '*.xlsx'),
                                   ('Excel 97-2003', '*.xls'), ('CSV', '*.csv')]
        self.states = ('All', 'AK', 'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL',
                       'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV',
                       'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN',
                       'TX', 'UT', 'VT', 'VA', 'VI', 'WA', 'WV', 'WI', 'WY')
        self.simulation_methods = ('Logistic Regression', 'Ridge Classifier', 'K Nearest Neighbor', 'Naive Bayes',
                                   'Support Vector Machine', 'Random Forest')
        self.cwd = os.getcwd()
        self.variables = self.winfo_toplevel().variables

    def hide_advanced_parameters(self):
        # self.detail_label.grid_forget()
        # self.detail_input.grid_forget()
        self.state_of_work_input.grid_forget()
        self.simulation_method_label.grid_forget()
        self.simulation_method_input.grid_forget()
        self.random_seed_label.grid_forget()
        self.random_seed_input.grid_forget()
        self.engine_type_label.grid_forget()
        self.engine_type_input.grid_forget()
        self.r_path_label.grid_forget()
        self.r_path_input.grid_forget()
        self.r_path_button.grid_forget()

    def show_advanced_parameters(self):
        # self.detail_label.grid(column=0, row=3, sticky=W, pady=self.row_padding)
        # self.detail_input.grid(column=1, row=3, sticky=W, padx=8, pady=self.row_padding)
        self.state_of_work_input.grid(column=1, row=6, columnspan=2, sticky=W, padx=8, pady=(0, self.row_padding))
        self.simulation_method_label.grid(column=0, row=7, sticky=W, pady=self.row_padding)
        self.simulation_method_input.grid(column=1, row=7, sticky=W, padx=8, pady=self.row_padding)
        self.random_seed_label.grid(column=0, row=8, sticky=W, pady=self.row_padding)
        self.random_seed_input.grid(column=1, row=8, sticky=W, padx=8, pady=self.row_padding)
        self.engine_type_label.grid(column=0, row=9, sticky=W, pady=self.row_padding)
        self.engine_type_input.grid(column=1, row=9, sticky=W, padx=8, pady=self.row_padding)
        self.toggle_r_path()

    def browse_file(self, file_input, filetypes):
        # Open a file dialogue where user can choose a file. Possible options are limited to CSV and Excel files.
        file_name = filedialog.askopenfilename(initialdir=self.cwd, filetypes=filetypes)
        file_input.delete(0, END)  # Clear current value in entry widget
        file_input.insert(0, file_name)  # Add user-selected value to entry widget

    def browse_directory(self, directory_input):
        # Open a file dialogue where user can choose a directory.
        directory_name = filedialog.askdirectory(initialdir=self.cwd)
        directory_input.delete(0, END)  # Clear current value in entry widget
        directory_input.insert(0, directory_name)  # Add user-selected value to entry widget

    def toggle_r_path(self, *_):
        if self.variables['engine_type'].get() == 'R':
            self.r_path_label.grid(column=0, row=10, sticky=W, pady=self.row_padding)
            self.r_path_input.grid(column=1, row=10, padx=8, sticky=(E, W), pady=self.row_padding)
            self.r_path_button.grid(column=4, row=10, pady=self.row_padding)
        else:
            self.r_path_label.grid_forget()
            self.r_path_input.grid_forget()
            self.r_path_button.grid_forget()


class ComparisonFrame(Frame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.showing_comparisons = False
        self.comparison_count = 0
        self.comparison_max = 4

        self.buttons = Frame(self, bg=DARK_COLOR)
        self.buttons.pack(side=RIGHT, anchor=E, fill=Y)

        self.start_button = BorderButton(self.buttons, custom=True, borderwidth=1, relief='flat',
                                         highlightbackground='#FFFFFF')
        button_content = SubtleToggle(self.start_button, text='Compare', font='-size 9 -weight bold',
                                      command=self.toggle_show_comparison)
        self.start_button.add_content(button_content)
        self.start_button.pack(side=RIGHT)

        self.add_simulation_button = SubtleButton(self.buttons, text='+', font='-size 11 -weight bold',
                                                  background=THEME_COLOR, width=3, padx=0, pady=0,
                                                  command=self.add_simulation)

        self.simulations = []
        self.simulations_frame = Frame(self, bg=DARK_COLOR)
        self.main_sim = SimulationSelectFrame(self, parent=self.simulations_frame)
        self.main_sim.toggle_on()
        self.main_sim.pack(side=LEFT, anchor=E, padx=(0, 3))

        self.simulations.append(self.main_sim)

    def toggle_show_comparison(self):
        if not self.showing_comparisons:
            self.simulations_frame.pack(side=RIGHT, anchor=E, fill=BOTH)
            self.add_simulation_button.pack(side=LEFT, padx=(0, 6))
            self.start_button.button.toggle()
            self.showing_comparisons = True
            self.winfo_toplevel().start_comparing()
        else:
            self.simulations_frame.pack_forget()
            self.add_simulation_button.pack_forget()
            self.start_button.button.toggle()
            self.showing_comparisons = False
            self.winfo_toplevel().stop_comparing()

    def add_simulation(self):
        if self.comparison_count >= self.comparison_max:
            return

        self.comparison_count += 1
        new_sim = SimulationSelectFrame(self, parent=self.simulations_frame, sim_num=self.comparison_count)
        new_sim.pack(side=LEFT, anchor=E, padx=(0, 3))
        self.simulations.append(new_sim)
        self.winfo_toplevel().add_comparison()

    def select_simulation(self, sim_num):
        for i in range(len(self.simulations)):
            if i != sim_num:
                self.simulations[i].toggle_off()
            else:
                self.simulations[i].toggle_on()

    def remove_simulation(self, sim_num):
        self.select_simulation(sim_num - 1)
        for i in range(sim_num + 1, self.comparison_count + 1):
            self.simulations[i].update_sim_num(i - 1)
        self.simulations[sim_num].pack_forget()
        del self.simulations[sim_num]
        self.comparison_count -= 1


class SimulationSelectFrame(Frame):
    def __init__(self, comparison_frame, parent=None, sim_num=0, width=None, **kwargs):
        self.comparison_frame = comparison_frame
        self.non_selected_color = '#808080'
        self.selected_color = '#FFFFFF'
        super().__init__(parent, relief='flat', highlightbackground=self.non_selected_color, highlightthickness=1,
                         **kwargs)
        self.sim_num = sim_num

        self.name = get_sim_name(sim_num)
        self.select_button = SubtleButton(self, text=self.name, fg=self.non_selected_color, width=width,
                                          command=self.select)
        self.select_button.pack(side=LEFT)
        self.remove_button = SubtleButton(self, text=u'\u2A09', bg='#d9d9d9', fg='#666666', width=2,
                                          command=self.remove)

    def toggle_off(self):
        self.config(highlightbackground=self.non_selected_color)
        self.select_button.config(fg=self.non_selected_color)
        self.remove_button.pack_forget()

    def toggle_on(self):
        self.config(highlightbackground=self.selected_color)
        self.select_button.config(fg=self.selected_color)
        if self.sim_num > 0:
            self.remove_button.pack(side=LEFT)

    def select(self):
        self.comparison_frame.select_simulation(self.sim_num)
        self.winfo_toplevel().switch_comparison(self.sim_num)

    def remove(self):
        self.comparison_frame.remove_simulation(self.sim_num)
        self.winfo_toplevel().remove_comparison(self.sim_num)

    def update_sim_num(self, sim_num):
        self.sim_num = sim_num
        self.name = 'Comparison {}'.format(self.sim_num)
        self.select_button.config(text=self.name)


class SettingsNotebook(ttk.Notebook):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, style='MSNotebook.TNotebook', **kwargs)

        # Create frames for each notebook tab. Each frame needs a canvas because scroll bars cannot be added to a frame.
        # They can only be added to canvases and listboxes. So another frame needs to be added inside the canvas. This
        # frame will contain the actual user input widgets.
        self.program_frame = ProgramFrame(self)
        self.population_frame = PopulationFrame(self)
        self.simulation_frame = SimulationFrame(self)

        # Add the frames to the notebook
        self.add(self.program_frame, text='Program')
        self.add(self.population_frame, text='Population')
        self.add(self.simulation_frame, text='Simulation')

        # In order to control scrolling in the right notebook tab, we need to keep track of the tab that
        # is currently visible. Whenever a tab is clicked, update this value.
        self.bind('<Button-1>', self.change_current_tab)
        # When the top window gets resized, the scroll regions in the notebook should be resized too
        self.bind('<Configure>', self.resize)

        self.update()
        self.set_notebook_width(self.program_frame.content.winfo_width())
        self.config(width=self.program_frame.content.winfo_width() + 18)

    # Change the currently visible tab.
    def change_current_tab(self, event):
        self.winfo_toplevel().current_tab = self.tk.call(self._w, "identify", "tab", event.x, event.y)

    def set_scroll_region(self, height=-1):
        scroll_frames = [self.program_frame, self.population_frame, self.simulation_frame]
        canvas_height = self.program_frame.canvas.winfo_height() if height < 0 else height

        for frame in scroll_frames:
            frame_height = frame.content.winfo_height()

            new_height = frame_height if frame_height > canvas_height else canvas_height
            frame.canvas.configure(scrollregion=(0, 0, 0, new_height))

    def set_notebook_width(self, width):
        self.program_frame.canvas.itemconfig(1, width=width)
        self.population_frame.canvas.itemconfig(1, width=width)
        self.simulation_frame.canvas.itemconfig(1, width=width)

    def resize(self, event):
        new_width = event.width - 30
        self.set_notebook_width(new_width)
        self.set_scroll_region(event.height - 30)

    def hide_advanced_parameters(self):
        self.program_frame.hide_advanced_parameters()
        self.population_frame.hide_advanced_parameters()
        self.simulation_frame.hide_advanced_parameters()

    def show_advanced_parameters(self):
        self.program_frame.show_advanced_parameters()
        self.population_frame.show_advanced_parameters()
        self.simulation_frame.show_advanced_parameters()


class ScrollFrame(Frame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        # Create scroll bar
        self.scroll_bar = ttk.Scrollbar(self, orient=VERTICAL)
        self.scroll_bar.pack(side=RIGHT, fill=Y)

        # Canvas needed to hold scroll wheel and content frame
        self.canvas = Canvas(self, bg=VERY_LIGHT_COLOR, borderwidth=0, highlightthickness=0,
                             yscrollcommand=self.scroll_bar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=0, pady=0)

        self.content = Frame(self, padx=10, pady=10, bg=VERY_LIGHT_COLOR, width=600)  # Frame holds the actual content
        self.canvas.create_window((0, 0), window=self.content, anchor='nw')  # Add frame to canvas
        self.scroll_bar.config(command=self.canvas.yview)

    def update_scroll_region(self):
        self.update()
        self.canvas.configure(scrollregion=(0, 0, 0, self.content.winfo_height()))


class NotebookFrame(ScrollFrame):
    __metaclass__ = ABCMeta

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.default_settings = self.winfo_toplevel().default_settings
        self.row_padding = 4

    @abstractmethod
    def __create_attributes(self):
        raise NotImplementedError

    @abstractmethod
    def hide_advanced_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def show_advanced_parameters(self):
        raise NotImplementedError

    # Some inputs require an entry value for each leave type. It is better to store each input in a list than
    # create separate variables for all of them.
    def create_leave_objects(self, parent, leave_vars):
        leave_type_labels = []  # A list of label widgets for inputs
        leave_type_inputs = []  # A list of entry inputs
        for i, leave_type in enumerate(LEAVE_TYPES):
            # Create the label and entry widgets
            leave_type_labels.append(Label(parent, text=leave_type, bg=VERY_LIGHT_COLOR, font='-size 10'))
            leave_type_inputs.append(NotebookEntry(parent, textvariable=leave_vars[leave_type], justify='center',
                                                   width=10))
            parent.columnconfigure(i, weight=1)

        return leave_type_labels, leave_type_inputs


class ProgramFrame(NotebookFrame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__create_attributes()
        v = self.variables

        # Create the input widgets for program parameters
        # ------------------------------------------- Program Eligibility -------------------------------------------
        # Inputs related to eligibility will be grouped in a label frame
        self.eligibility_frame_label = ttk.Label(self.content, text='Eligibility Rules:', cursor='question_arrow',
                                                 style='MSLabelframe.TLabelframe.Label', font='-size 10')
        self.eligibility_frame = ttk.Labelframe(self.content, labelwidget=self.eligibility_frame_label,
                                                style='MSLabelframe.TLabelframe')
        CreateToolTip(self.eligibility_frame_label, 'The requirements to be eligible for the paid leave program.')

        # Earnings
        tip = 'The amount of money earned in the last year.'
        self.eligible_earnings_label = TipLabel(self.eligibility_frame, tip, text="Earnings", bg=VERY_LIGHT_COLOR,
                                                font='-size 10')
        self.eligible_earnings_input = NotebookEntry(self.eligibility_frame, textvariable=v['eligible_earnings'],
                                                     justify='center', width=15)

        # Weeks worked
        tip = 'The number of weeks worked in the last year.'
        self.eligible_weeks_label = TipLabel(self.eligibility_frame, tip, text="Weeks", bg=VERY_LIGHT_COLOR,
                                             font='-size 10')
        self.eligible_weeks_input = NotebookEntry(self.eligibility_frame, textvariable=v['eligible_weeks'],
                                                  justify='center', width=15)

        # Hours worked
        tip = 'The number of hours worked in the last year.'
        self.eligible_hours_label = TipLabel(self.eligibility_frame, tip, text="Hours", bg=VERY_LIGHT_COLOR,
                                             font='-size 10')
        self.eligible_hours_input = NotebookEntry(self.eligibility_frame, textvariable=v['eligible_hours'],
                                                  justify='center', width=15)

        # Employer size
        tip = 'Size of the employer.'
        self.eligible_size_label = TipLabel(self.eligibility_frame, tip, text="Employer Size", bg=VERY_LIGHT_COLOR,
                                            font='-size 10')
        self.eligible_size_input = NotebookEntry(self.eligibility_frame, textvariable=v['eligible_size'],
                                                 justify='center', width=15)

        # ----------------------------------------- Max Weeks with Benefits -----------------------------------------
        self.max_weeks_frame_label = ttk.Label(self.content, text='Max Weeks:', style='MSLabelframe.TLabelframe.Label',
                                               cursor='question_arrow')
        self.max_weeks_frame = ttk.Labelframe(self.content, labelwidget=self.max_weeks_frame_label,
                                              style='MSLabelframe.TLabelframe')
        self.max_weeks_labels, self.max_weeks_inputs = self.create_leave_objects(self.max_weeks_frame, v['max_weeks'])
        CreateToolTip(self.max_weeks_frame_label,
                      'The maximum number of weeks for each leave type that the program will pay for.')

        # ----------------------------------------- Wage Replacement Ratio ------------------------------------------
        tip = 'The percentage of wage that the program will pay.'
        self.replacement_ratio_label = TipLabel(self.content, tip, text="Replacement Ratio:", bg=VERY_LIGHT_COLOR)
        self.replacement_ratio_input = NotebookEntry(self.content, textvariable=v['replacement_ratio'])

        # ------------------------------------------- Weekly Benefit Cap --------------------------------------------
        tip = 'The maximum amount of benefits paid out per week.'
        self.weekly_ben_cap_label = TipLabel(self.content, tip, text="Weekly Benefit Cap:", bg=VERY_LIGHT_COLOR)
        self.weekly_ben_cap_input = NotebookEntry(self.content, textvariable=v['weekly_ben_cap'])

        # -------------------------------------------- Benefit Financing --------------------------------------------
        # self.benefit_financing_frame_label = ttk.Label(self.content, text='Benefit Financing:',
        #                                                style='MSLabelframe.TLabelframe.Label')
        self.benefit_financing_frame = ttk.LabelFrame(self.content, text='Benefit Financing:',
                                                      style='MSLabelframe.TLabelframe')

        # Tax on Payroll
        tip = 'The payroll tax that will be implemented to fund benefits program.'
        self.payroll_tax_label = TipLabel(self.benefit_financing_frame, tip, text='Payroll Tax (%):',
                                          bg=VERY_LIGHT_COLOR)
        self.payroll_tax_input = NotebookEntry(self.benefit_financing_frame, textvariable=v['payroll_tax'])

        # Tax on Benefits
        tip = 'Whether or not program benefits are taxed.'
        self.benefits_tax_input = TipCheckButton(self.benefit_financing_frame, tip, text='Benefits Tax',
                                                 variable=v['benefits_tax'])

        # Average State Tax
        tip = 'The average tax rate of a selected state.'
        self.average_state_tax_label = TipLabel(self.benefit_financing_frame, tip, text='State Average Tax Rate (%):',
                                                bg=VERY_LIGHT_COLOR)
        self.average_state_tax_input = NotebookEntry(self.benefit_financing_frame,
                                                     textvariable=v['average_state_tax'])

        # Maximum Taxable Earnings per Person
        tip = 'The maximum amount that a person can be taxed.'
        self.max_taxable_earnings_per_person_label = TipLabel(self.benefit_financing_frame, tip,
                                                              text='Maximum Taxable Earnings Per Person ($):',
                                                              bg=VERY_LIGHT_COLOR)
        self.max_taxable_earnings_per_person_input = NotebookEntry(self.benefit_financing_frame,
                                                                   textvariable=v['max_taxable_earnings_per_person'])

        # Maximum Taxable Earnings Total
        tip = 'The total earnings that can be taxed.'
        self.total_taxable_earnings_label = TipLabel(self.benefit_financing_frame, tip,
                                                     text='Total Taxable Earnings ($):', bg=VERY_LIGHT_COLOR)
        self.total_taxable_earnings_input = NotebookEntry(self.benefit_financing_frame,
                                                          textvariable=v['total_taxable_earnings'])

        # ------------------------------------ Government Employees Eligibility -------------------------------------
        # All Government Employees
        tip = 'Whether or not government employees are eligible for program.'
        self.government_employees_input = TipCheckButton(self.content, tip, text="Government Employees",
                                                         variable=v['government_employees'],
                                                         command=self.check_all_gov_employees)

        # Federal Employees
        tip = 'Whether or not federal employees are eligible for program.'
        self.federal_employees_input = TipCheckButton(self.content, tip, text="Federal Employees",
                                                      variable=v['fed_employees'], command=self.check_gov_employees)

        # State Employees
        tip = 'Whether or not state employees are eligible for program.'
        self.state_employees_input = TipCheckButton(self.content, tip, text="State Employees",
                                                    variable=v['state_employees'], command=self.check_gov_employees)

        # Local Government Employees
        tip = 'Whether or not local employees are eligible for program.'
        self.local_employees_input = TipCheckButton(self.content, tip, text="Local Employees",
                                                    variable=v['local_employees'], command=self.check_gov_employees)

        # ------------------------------------ Self-Employed Worker Eligibility -------------------------------------
        tip = 'Whether or not self employed workers are eligible for program.'
        self.self_employed_input = TipCheckButton(self.content, tip, text="Self Employed", variable=v['self_employed'])

        # Add input widgets to the parent widget
        self.eligibility_frame.grid(column=0, row=0, columnspan=2, sticky=(N, E, W), pady=self.row_padding)
        self.eligible_earnings_label.grid(column=0, row=0)
        self.eligible_weeks_label.grid(column=1, row=0)
        self.eligible_hours_label.grid(column=2, row=0)
        self.eligible_size_label.grid(column=3, row=0)
        self.eligible_earnings_input.grid(column=0, row=1, sticky=(E, W), padx=1, pady=(0, 2))
        self.eligible_weeks_input.grid(column=1, row=1, sticky=(E, W), padx=1, pady=(0, 2))
        self.eligible_hours_input.grid(column=2, row=1, sticky=(E, W), padx=1, pady=(0, 2))
        self.eligible_size_input.grid(column=3, row=1, sticky=(E, W), padx=1, pady=(0, 2))
        self.max_weeks_frame.grid(column=0, row=1, columnspan=2, sticky=(N, E, W), pady=self.row_padding)
        display_leave_objects(self.max_weeks_labels, self.max_weeks_inputs)
        self.benefit_financing_frame.grid(column=0, row=2, columnspan=2, sticky=(N, E, W), pady=self.row_padding)
        self.payroll_tax_label.grid(column=0, row=0, sticky=W, padx=(8, 0), pady=self.row_padding)
        self.payroll_tax_input.grid(column=1, row=0, sticky=W, pady=self.row_padding)
        self.average_state_tax_label.grid(column=0, row=1, sticky=W, padx=(8, 0), pady=self.row_padding)
        self.average_state_tax_input.grid(column=1, row=1, sticky=W, pady=self.row_padding)
        self.benefits_tax_input.grid(column=0, row=2, columnspan=2, sticky=W, padx=(16, 0), pady=self.row_padding)
        self.max_taxable_earnings_per_person_label.grid(column=0, row=3, sticky=W, padx=(8, 0), pady=self.row_padding)
        self.max_taxable_earnings_per_person_input.grid(column=1, row=3, sticky=W, pady=self.row_padding)
        # self.total_taxable_earnings_label.grid(column=0, row=4, sticky=W, padx=(8, 0), pady=self.row_padding)
        # self.total_taxable_earnings_input.grid(column=1, row=4, sticky=W, pady=self.row_padding)
        self.replacement_ratio_label.grid(column=0, row=3, sticky=W, pady=self.row_padding)
        self.replacement_ratio_input.grid(column=1, row=3, sticky=W, pady=self.row_padding)
        self.weekly_ben_cap_label.grid(column=0, row=4, sticky=W, pady=self.row_padding)
        self.weekly_ben_cap_input.grid(column=1, row=4, sticky=W, pady=self.row_padding)
        self.government_employees_input.grid(column=0, row=5, columnspan=2, sticky=W, pady=(self.row_padding, 0))
        self.federal_employees_input.grid(column=0, row=6, columnspan=2, sticky=W, padx=(15, 0))
        self.state_employees_input.grid(column=0, row=7, columnspan=2, sticky=W, padx=(15, 0))
        self.local_employees_input.grid(column=0, row=8, columnspan=2, sticky=W, padx=(15, 0),
                                        pady=(0, self.row_padding))
        self.self_employed_input.grid(column=0, row=9, columnspan=2, sticky=W, pady=self.row_padding)

        # Give weight to columns
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        for i in range(4):
            self.eligibility_frame.columnconfigure(i, weight=1)

    def __create_attributes(self):
        self.variables = self.winfo_toplevel().variables

    def hide_advanced_parameters(self):
        pass

    def show_advanced_parameters(self):
        pass

    def check_all_gov_employees(self, _=None):
        checked = self.variables['government_employees'].get()
        self.variables['fed_employees'].set(checked)
        self.variables['state_employees'].set(checked)
        self.variables['local_employees'].set(checked)

    def check_gov_employees(self):
        if self.variables['fed_employees'].get() and self.variables['state_employees'].get() and \
                self.variables['local_employees'].get():
            self.variables['government_employees'].set(1)
        else:
            self.variables['government_employees'].set(0)


class PopulationFrame(NotebookFrame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__create_attributes()

        # Create the input widgets for population parameters
        # ---------------------------------------------- Take Up Rates ----------------------------------------------
        self.take_up_rates_frame_label = ttk.Label(self.content, text='Take Up Rates:', cursor='question_arrow',
                                                   style='MSLabelframe.TLabelframe.Label')
        self.take_up_rates_frame = ttk.Labelframe(self.content, labelwidget=self.take_up_rates_frame_label,
                                                  style='MSLabelframe.TLabelframe')
        self.take_up_rates_labels, self.take_up_rates_inputs = \
            self.create_leave_objects(self.take_up_rates_frame, self.variables['take_up_rates'])
        CreateToolTip(self.take_up_rates_frame_label, 'The proportion of eligible leave takers who decide to use the '
                                                      'program for each leave type.')

        # ---------------------------------------- Leave Probability Factors ----------------------------------------
        self.leave_probability_factors_frame_label = ttk.Label(self.content, text='Leave Probability Factors:',
                                                               style='MSLabelframe.TLabelframe.Label',
                                                               cursor='question_arrow')
        self.leave_probability_factors_frame = ttk.Labelframe(self.content,
                                                              labelwidget=self.leave_probability_factors_frame_label,
                                                              style='MSLabelframe.TLabelframe')
        self.leave_probability_factors_labels, self.leave_probability_factors_inputs = \
            self.create_leave_objects(self.leave_probability_factors_frame, self.variables['leave_probability_factors'])
        CreateToolTip(self.leave_probability_factors_frame_label, 'Factors the probability of needing or taking '
                                                                  'a leave for each type of leave.')

        # --------------------------------------------- Benefit Effect ----------------------------------------------
        tip = 'Whether or not the benefit amount affects participation in the program.'
        self.benefit_effect_input = TipCheckButton(self.content, tip, text="Benefit Effect",
                                                   variable=self.variables['benefit_effect'])

        # ---------------------------------------- Participants Extend Leave ----------------------------------------
        tip = 'Whether or not participants extend their leave in the presence of the program.'
        self.extend_input = TipCheckButton(self.content, tip, text="Extend", variable=self.variables['extend'])

        # ---------------------------------------- Needers Fully Participate ----------------------------------------
        tip = 'Whether or not all people who need leave take leave in the presnce of the program.'
        self.needers_fully_participate_input = TipCheckButton(self.content, tip, text="Needers Fully Participate",
                                                              variable=self.variables['needers_fully_participate'])

        # ---------------------------------------------- Top Off Rate -----------------------------------------------
        tip = 'The proportion of employers already paying full wages in the absence of the program ' \
              'that will top off benefits in the presence of a program to reach full wages.'
        self.top_off_rate_label = TipLabel(self.content, tip, text="Top Off Rate:", bg=VERY_LIGHT_COLOR)
        self.top_off_rate_input = NotebookEntry(self.content, textvariable=self.variables['top_off_rate'])

        # ----------------------------------------- Top Off Minimum Length ------------------------------------------
        tip = 'The number of days employers will top off benefits.'
        self.top_off_min_length_label = TipLabel(self.content, tip, text="Top Off Minimum Length:",
                                                 bg=VERY_LIGHT_COLOR)
        self.top_off_min_length_input = NotebookEntry(self.content, textvariable=self.variables['top_off_min_length'])

        # ----------------------------------------- Share of Dual Receivers -----------------------------------------
        tip = 'Dual receiver of company and state benefits'
        self.dual_receivers_share_label = TipLabel(self.content, tip, text='Share of Dual Receivers:',
                                                   bg=VERY_LIGHT_COLOR)
        self.dual_receivers_share_input = NotebookEntry(self.content,
                                                        textvariable=self.variables['dual_receivers_share'])

        # Add input widgets to the parent widget
        self.take_up_rates_frame.grid(column=0, row=0, columnspan=2, sticky=(N, E, W), pady=self.row_padding)
        display_leave_objects(self.take_up_rates_labels, self.take_up_rates_inputs)
        display_leave_objects(self.leave_probability_factors_labels, self.leave_probability_factors_inputs)
        # self.benefit_effect_input.grid(column=0, row=2, columnspan=2, sticky=W)
        # self.extend_input.grid(column=0, row=3, columnspan=3, sticky=W)
        # self.dual_receivers_share_label.grid(column=0, row=4, sticky=W)
        # self.dual_receivers_share_input.grid(column=1, row=4, sticky=W)

        # Make second column take up more space
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)

    def __create_attributes(self):
        self.variables = self.winfo_toplevel().variables

    def hide_advanced_parameters(self):
        # self.leave_probability_factors_frame.grid_forget()
        # self.needers_fully_participate_input.grid_forget()
        # self.top_off_rate_label.grid_forget()
        # self.top_off_rate_input.grid_forget()
        # self.top_off_min_length_label.grid_forget()
        # self.top_off_min_length_input.grid_forget()
        pass

    def show_advanced_parameters(self):
        # self.leave_probability_factors_frame.grid(column=0, row=1, columnspan=2, sticky=(N, E, W),
        #                                           pady=self.row_padding)
        # self.needers_fully_participate_input.grid(column=0, row=4, columnspan=2, sticky=W, pady=self.row_padding)
        # self.top_off_rate_label.grid(column=0, row=5, sticky=W, pady=self.row_padding)
        # self.top_off_rate_input.grid(column=1, row=5, sticky=W, pady=self.row_padding)
        # self.top_off_min_length_label.grid(column=0, row=6, sticky=W, pady=self.row_padding)
        # self.top_off_min_length_input.grid(column=1, row=6, sticky=W, pady=self.row_padding)
        pass


class SimulationFrame(NotebookFrame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__create_attributes()
        v = self.variables

        # ----------------------------------------- Existing State Program ------------------------------------------
        tip = 'Sets program parameters to match an existing state program.'
        self.existing_program_label = TipLabel(self.content, tip, text='Existing State Program:', bg=VERY_LIGHT_COLOR)
        self.existing_program_input = ttk.Combobox(self.content, textvariable=v['existing_program'],
                                                   state="readonly", width=5, values=list(DEFAULT_STATE_PARAMS.keys()))
        self.existing_program_input.current(0)

        # Create the input widgets for simulation parameters
        # ---------------------------------------------- Clone Factor -----------------------------------------------
        tip = 'The number of times each sample person will be run through the simulation.'
        self.clone_factor_label = TipLabel(self.content, tip, text="Clone Factor:", bg=VERY_LIGHT_COLOR)
        self.clone_factor_input = NotebookEntry(self.content, textvariable=v['clone_factor'])

        # ----------------------------------------------- SE Analysis -----------------------------------------------
        tip = 'Whether or not weight should be divided by clone factor value.'
        self.se_analysis_input = TipCheckButton(self.content, tip, text="SE Analysis", variable=v['se_analysis'])

        # ---------------------------------------------- Weight Factor ----------------------------------------------
        # tip = 'Multiplies the sample weights by value.'
        # self.weight_factor_label = TipLabel(self.content, tip, text="Weight Factor:", bg=VERY_LIGHT_COLOR)
        # self.weight_factor_input = MSNotebookEntry(self.content, textvariable=v['weight_factor'])

        # --------------------------------------- FMLA Protection Constraint ----------------------------------------
        tip = 'If checked, leaves that are extended due to a paid leave program will be capped at 12 weeks.'
        self.fmla_protection_constraint_input = TipCheckButton(self.content, tip, text="FMLA Protection Constraint",
                                                               variable=v['fmla_protection_constraint'])

        # ------------------------------------------------ Calibrate ------------------------------------------------
        tip = '''Indicates whether or not the calibration add-factors are used in the equations giving the probability 
        of taking or needing leaves. These calibration factors adjust the simulated probabilities of taking or needing 
        the most recent leave to equal those in the Family and Medical Leave in 2012: Revised Public Use File 
        Documentation (McGarry et al, Abt Associates, 2013).'''
        self.calibrate_input = TipCheckButton(self.content, tip, text="Calibrate", variable=v['calibrate'])

        # # ---------------------------------------- Compare Against Existing -----------------------------------------
        # tip = 'Simulate a counterfactual scenario to compare user parameters  against a real paid leave program.'
        # self.counterfactual_label = TipLabel(self.content, tip, text='Compare Against Existing:', bg=VERY_LIGHT_COLOR)
        # self.counterfactual_input = ttk.Combobox(self.content, textvariable=v['counterfactual'],
        #                                          state="readonly", width=5, values=list(DEFAULT_STATE_PARAMS.keys()))
        # self.counterfactual_input.current(0)
        #
        # # ---------------------------------------- Compare Against Generous -----------------------------------------
        # tip = 'Simulate a policy scenario to compare user parameters against a  generous paid leave program ' \
        #       'in which everyone is eligible and the wage replacement is 1.'
        # self.policy_sim_input = TipCheckButton(self.content, tip, text="Compare Against Generous",
        #                                        variable=v['policy_sim'])

        # Add input widgets to the parent widget
        self.existing_program_label.grid(column=0, row=0, sticky=W, pady=self.row_padding)
        self.existing_program_input.grid(column=1, row=0, sticky=W, pady=self.row_padding)
        # self.counterfactual_label.grid(column=0, row=0, sticky=W, pady=self.row_padding)
        # self.counterfactual_input.grid(column=1, row=0, sticky=W, pady=self.row_padding)
        # self.policy_sim_input.grid(column=0, row=1, columnspan=2, sticky=W, pady=self.row_padding)

        # self.se_analysis_input.grid(column=0, row=1, columnspan=2, sticky=W)
        # self.calibrate_input.grid(column=0, row=4, columnspan=2, sticky=W)

    def __create_attributes(self):
        self.variables = self.winfo_toplevel().variables

    def hide_advanced_parameters(self):
        # self.weight_factor_label.grid_forget()
        # self.weight_factor_input.grid_forget()
        self.clone_factor_label.grid_forget()
        self.clone_factor_input.grid_forget()
        # self.fmla_protection_constraint_input.grid_forget()

    def show_advanced_parameters(self):
        # self.weight_factor_label.grid(column=0, row=2, sticky=W, pady=self.row_padding)
        # self.weight_factor_input.grid(column=1, row=2, sticky=W, pady=self.row_padding)
        self.clone_factor_label.grid(column=0, row=2, sticky=W, pady=self.row_padding)
        self.clone_factor_input.grid(column=1, row=2, sticky=W, pady=self.row_padding)
        # self.fmla_protection_constraint_input.grid(column=0, row=3, columnspan=2, sticky=W, pady=self.row_padding)


class ResultsWindow(Toplevel):
    def __init__(self, parent, simulation_engine, abf_module):
        super().__init__(parent)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.icon = PhotoImage(file='impaq_logo.gif')
        self.tk.call('wm', 'iconphoto', self._w, self.icon)

        self.parent = parent

        self.content = Frame(self, bg=DARK_COLOR)
        self.notebook = ttk.Notebook(self.content, style='MSNotebook.TNotebook')
        self.notebook.bind('<Button-1>', self.change_current_tab)
        self.current_tab = 0

        print('Creating summary frame')
        self.summary_frame = ResultsSummary(self, simulation_engine)
        print('Finished summary frame, adding to notebook')
        self.notebook.add(self.summary_frame, text='Summary')
        print('Finished adding summary frame to notebook')

        self.abf = ABFResults(self.notebook, abf_module, bg=VERY_LIGHT_COLOR)
        print('Creating ABF results summary frame')
        self.notebook.add(self.abf, text="Benefit Financing")

        self.population_analysis = PopulationAnalysis(self.notebook, simulation_engine)
        self.notebook.add(self.population_analysis, text='Population Analysis')

        self.content.pack(expand=True, fill=BOTH)
        self.notebook.pack(expand=True, fill=BOTH)
        self.notebook.select(self.summary_frame)
        self.notebook.enable_traversal()

        self.bind("<MouseWheel>", self.scroll)
        # self.bind('<Configure>', self.resize)

        print('Updating widgets')
        self.update()
        self.notebook.config(width=self.abf.content.winfo_width() + 25)
        self.abf.update_scroll_region()
        self.population_analysis.update_scroll_region()
        self.resizable(False, False)

    def scroll(self, event):
        # In Windows, the delta will be either 120 or -120. In Mac, it will be 1 or -1.
        # The delta value will determine whether the user is scrolling up or down.
        move_unit = 0
        if event.num == 5 or event.delta > 0:
            move_unit = -2
        elif event.num == 4 or event.delta < 0:
            move_unit = 2

        # Only scroll the tab that is currently visible.
        if self.current_tab == 1:
            self.abf.canvas.yview_scroll(move_unit, 'units')
        elif self.current_tab == 2:
            self.population_analysis.canvas.yview_scroll(move_unit, 'units')

    def change_current_tab(self, event):
        self.current_tab = self.notebook.tk.call(self.notebook._w, "identify", "tab", event.x, event.y)

    def set_notebook_width(self, width):
        self.abf.canvas.itemconfig(1, width=width)
        self.population_analysis.canvas.itemconfig(1, width=width)

    def on_close(self):
        self.destroy()


class PopulationAnalysis(ScrollFrame):
    def __init__(self, parent, simulation_engine):
        super().__init__(parent)
        self.simulation_engine = simulation_engine

        self.parameters_frame = Frame(self.content, padx=4, pady=4, bg=DARK_COLOR)
        self.parameters_frame.pack(fill=X, pady=(0, 4))

        self.gender = StringVar()
        self.gender_label = Label(self.parameters_frame, text='Gender:', font='Helvetica 12 bold', bg=DARK_COLOR,
                                  fg=LIGHT_COLOR)
        self.gender_input = ttk.Combobox(self.parameters_frame, textvariable=self.gender, state="readonly", width=10,
                                         values=['Both', 'Male', 'Female'])
        self.gender_input.current(0)

        self.age_min = IntVar()
        self.age_max = IntVar()
        self.age_label = Label(self.parameters_frame, text='Age:', font='Helvetica 12 bold', bg=DARK_COLOR,
                               fg=LIGHT_COLOR)
        self.age_min_label = Label(self.parameters_frame, text='Min', bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.age_max_label = Label(self.parameters_frame, text='Max', bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.age_min_input = GeneralEntry(self.parameters_frame, textvariable=self.age_min)
        self.age_max_input = GeneralEntry(self.parameters_frame, textvariable=self.age_max)

        self.wage_min = DoubleVar()
        self.wage_max = DoubleVar()
        self.wage_label = Label(self.parameters_frame, text='Wage:', font='Helvetica 12 bold', bg=DARK_COLOR,
                                fg=LIGHT_COLOR)
        self.wage_min_label = Label(self.parameters_frame, text='Min', bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.wage_max_label = Label(self.parameters_frame, text='Max', bg=DARK_COLOR, fg=LIGHT_COLOR)
        self.wage_min_input = GeneralEntry(self.parameters_frame, textvariable=self.wage_min)
        self.wage_max_input = GeneralEntry(self.parameters_frame, textvariable=self.wage_max)

        self.submit_button = BorderButton(self.parameters_frame, text='Submit', command=lambda: self.__update_histograms())

        self.gender_label.grid(column=0, row=1, sticky=W, pady=2)
        self.gender_input.grid(column=0, row=2, sticky=W, pady=2)
        self.age_label.grid(column=1, row=0, sticky=W, pady=2)
        self.age_min_label.grid(column=1, row=1, sticky=W, pady=2)
        self.age_max_label.grid(column=2, row=1, sticky=W, pady=2)
        self.age_min_input.grid(column=1, row=2, sticky=W, padx=2)
        self.age_max_input.grid(column=2, row=2, sticky=W, padx=2)
        self.wage_label.grid(column=3, row=0, sticky=W, pady=2)
        self.wage_min_label.grid(column=3, row=1, sticky=W, pady=2)
        self.wage_max_label.grid(column=4, row=1, sticky=W, pady=2)
        self.wage_min_input.grid(column=3, row=2, sticky=W, padx=2)
        self.wage_max_input.grid(column=4, row=2, sticky=W, padx=2)
        self.submit_button.grid(column=0, row=3, sticky=W, pady=4)

        # Histogram properties
        self.bin_size = 5
        self.max_weekdays = 262
        self.bins = list(range(0, self.max_weekdays, self.bin_size))
        self.xticks = list(range(0, self.max_weekdays, 20))

        self.histogram_frame = Frame(self.content, bg=DARK_COLOR)
        self.histograms = []
        self.__create_histograms()
        self.histogram_frame.pack(side=TOP, fill=BOTH, expand=True)

    def __create_histograms(self):
        # Create new charts for each simulation
        for sim_num in range(self.simulation_engine.sim_count):
            simulation_data = self.filter_data(self.simulation_engine.get_population_analysis_results(sim_num))
            title = get_sim_name(sim_num)
            histogram = self.create_histogram(simulation_data['cpl'], self.bins, simulation_data['PWGTP'], title,
                                              self.xticks)
            self.histograms.append(histogram)
            chart_container = ChartContainer(self.histogram_frame, histogram, DARK_COLOR)
            chart_container.pack()

    def __update_histograms(self):
        for sim_num in range(self.simulation_engine.sim_count):
            simulation_data = self.filter_data(self.simulation_engine.get_population_analysis_results(sim_num))
            fig = self.histograms[sim_num]
            ax = fig.axes[0]
            ax.cla()
            ax.hist(simulation_data['cpl'], self.bins, weights=simulation_data['PWGTP'], color='#1aff8c', rwidth=0.9)
            self.set_histogram_properties(fig, ax, self.xticks, get_sim_name(sim_num))
            fig.canvas.draw()
            fig.canvas.flush_events()

    def filter_data(self, data):
        if self.gender.get() == 'Male':
            data = data[data['female'] == 0]
        elif self.gender.get() == 'Female':
            data = data[data['female'] == 1]

        if self.age_min.get() is not None:
            data = data[data['age'] >= self.age_min.get()]
        if self.age_max.get() is not None and self.age_max.get() > 0:
            data = data[data['age'] <= self.age_max.get()]

        if self.wage_min.get() is not None:
            data = data[data['wage12'] >= self.wage_min.get()]
        if self.wage_max.get() is not None and self.wage_max.get() > 0:
            data = data[data['wage12'] <= self.wage_max.get()]

        return data

    def create_histogram(self, data, bins, weights, title_str, xticks=None):
        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.hist(data, bins, weights=weights, color='#1aff8c', rwidth=0.9)
        self.set_histogram_properties(fig, ax, xticks, title_str)
        return fig

    def set_histogram_properties(self, fig, ax, xticks, title_str):
        ax.set_ylabel('Number of Days', fontsize=9)
        ax.set_xlabel('Number of Workers', fontsize=9)
        if xticks is not None:
            ax.set_xticks(xticks)
        title = 'State: {}. Leaves Taken under Program. {}'.format(self.winfo_toplevel().parent.general_settings.state,
                                                                   title_str)
        format_chart(fig, ax, title, DARK_COLOR, 'white')


class ResultsSummary(Frame):
    def __init__(self, parent, engine):
        super().__init__(parent)
        print('Creating and saving summary chart')
        self.chart = engine.create_chart(engine.get_cost_df(0), 0)
        self.chart_container = Frame(self)

        print('Creating summary chart canvas')
        canvas = FigureCanvasTkAgg(self.chart, self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        self.chart_container.pack(fill=X, padx=15, pady=15)

        save_button = BorderButton(self.chart_container, text='Save Figure', width=10, pady=1,
                                   command=lambda: self.save_file())
        print('Created save button')
        save_button.pack(side=RIGHT, padx=10, pady=10)

    def save_file(self):
        filename = filedialog.asksaveasfilename(defaultextension='.png', initialdir=os.getcwd(),
                                                filetypes=[('PNG', '.png'), ('PDF', '*.pdf'), ('PGF', '*.pgf'),
                                                           ('EPS', '*.eps'), ('PS', '*.ps'), ('Raw', '*.raw'),
                                                           ('RGBA', '*.rgba'), ('SVG', '*.svg'), ('SVGZ', '*.svgz')])
        if filename is None:
            return
        self.chart.savefig(filename, facecolor='#333333', edgecolor='white')


class ChartContainer(Frame):
    def __init__(self, parent, chart, bg_color):
        super().__init__(parent, bg=bg_color)
        self.chart = chart
        self.bg_color = bg_color
        self.canvas = FigureCanvasTkAgg(chart, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().config(height=300)
        self.canvas.get_tk_widget().pack(side=TOP, fill=X)

        save_button = BorderButton(self, text='Save Figure', width=10, command=lambda: self.save_file())
        save_button.pack(side=LEFT, padx=10, pady=4)

    def save_file(self):
        filename = filedialog.asksaveasfilename(defaultextension='.png', initialdir=os.getcwd(),
                                                filetypes=[('PNG', '.png'), ('PDF', '*.pdf'), ('PGF', '*.pgf'),
                                                           ('EPS', '*.eps'), ('PS', '*.ps'), ('Raw', '*.raw'),
                                                           ('RGBA', '*.rgba'), ('SVG', '*.svg'), ('SVGZ', '*.svgz')])
        if filename is None:
            return
        self.chart.savefig(filename, facecolor=self.bg_color, edgecolor='white')


class ABFResults(ScrollFrame):
    def __init__(self, parent, abf_module, **kwargs):
        super().__init__(parent, **kwargs)

        self.abf_module = abf_module
        abf_output, pivot_tables = self.abf_module.run()
        top_level = self.winfo_toplevel()

        self.abf_summary = ABFResultsSummary(self.content, abf_output)
        self.abf_summary.pack(padx=10, pady=10)
        self.abf_pivot_tables = Frame(self.content, bg=DARK_COLOR)
        self.abf_pivot_tables.pack(fill=X, expand=True)
        self.abf_params_reveal = BorderButton(self, text='ABF Parameters', padx=4, command=self.show_params,
                                              width=16, borderwidth=0, font='-size 12', background='#00e600')
        self.abf_params_reveal.pack(side=BOTTOM, anchor='se', padx=3, pady=2)
        self.abf_params = Frame(self, bg=VERY_LIGHT_COLOR, borderwidth=1, relief='solid', padx=3, pady=3)
        self.abf_params_inputs = Frame(self.abf_params, bg=VERY_LIGHT_COLOR, pady=4)
        self.abf_params_inputs.pack(fill=X, side=TOP)
        self.abf_params_buttons = Frame(self.abf_params, bg=VERY_LIGHT_COLOR, pady=4)
        self.abf_params_buttons.pack(side=BOTTOM, fill=X, expand=True)
        self.abf_params_hide = BorderButton(self.abf_params_buttons, text='Hide', padx=4, command=self.hide_params,
                                            background='#00e600')
        self.abf_params_hide.pack(side=LEFT, pady=3, padx=5)
        self.run_button = BorderButton(self.abf_params_buttons, font='-size 11 -weight bold', text="Run ABF",
                                       command=self.rerun_abf, padx=4)
        self.run_button.pack(side=RIGHT, pady=3, padx=5)

        print('Creating ABF graphs')
        self.display_abf_bar_graphs(pivot_tables)

        tip = 'The payroll tax that will be implemented to fund benefits program.'
        self.payroll_tax_label = TipLabel(self.abf_params_inputs, tip, text='Payroll Tax (%):', bg=VERY_LIGHT_COLOR)
        self.payroll_tax_input = Entry(self.abf_params_inputs, textvariable=top_level.parent.variables['payroll_tax'])

        tip = 'Whether or not program benefits are taxed.'
        self.benefits_tax_input = TipCheckButton(self.abf_params_inputs, tip, text='Benefits Tax',
                                                 variable=top_level.parent.variables['benefits_tax'])

        tip = 'The average tax rate of a selected state.'
        self.average_state_tax_label = TipLabel(self.abf_params_inputs, tip, text='State Average Tax Rate (%):',
                                                bg=VERY_LIGHT_COLOR)
        self.average_state_tax_input = Entry(self.abf_params_inputs,
                                             textvariable=top_level.parent.variables['average_state_tax'])

        tip = 'The maximum amount that a person can be taxed.'
        self.max_taxable_earnings_per_person_label = TipLabel(self.abf_params_inputs, tip,
                                                              text='Maximum Taxable Earnings\nPer Person ($):',
                                                              bg=VERY_LIGHT_COLOR, justify=LEFT)
        self.max_taxable_earnings_per_person_input = \
            Entry(self.abf_params_inputs, textvariable=top_level.parent.variables['max_taxable_earnings_per_person'])

        tip = 'The total earnings that can be taxed.'
        self.total_taxable_earnings_label = TipLabel(self.abf_params_inputs, tip, text='Total Taxable Earnings ($):',
                                                     bg=VERY_LIGHT_COLOR)
        self.total_taxable_earnings_input = Entry(self.abf_params_inputs,
                                                  textvariable=top_level.parent.variables['total_taxable_earnings'])

        self.payroll_tax_label.grid(column=0, row=0, sticky=W, padx=(8, 0))
        self.payroll_tax_input.grid(column=1, row=0, sticky=W)
        self.average_state_tax_label.grid(column=0, row=1, sticky=W, padx=(8, 0))
        self.average_state_tax_input.grid(column=1, row=1, sticky=W)
        self.benefits_tax_input.grid(column=0, row=2, columnspan=2, sticky=W, padx=(16, 0))
        self.max_taxable_earnings_per_person_label.grid(column=0, row=3, sticky=W, padx=(8, 0))
        self.max_taxable_earnings_per_person_input.grid(column=1, row=3, sticky=W)
        # self.total_taxable_earnings_label.grid(column=0, row=4, sticky=W, padx=(8, 0))
        # self.total_taxable_earnings_input.grid(column=1, row=4, sticky=W)

    def show_params(self):
        self.abf_params_reveal.pack_forget()
        self.abf_params.pack(side=BOTTOM, anchor='se', padx=1)

    def hide_params(self):
        self.abf_params.pack_forget()
        self.abf_params_reveal.pack(side=BOTTOM, anchor='se', padx=3, pady=2)

    def display_abf_bar_graphs(self, pivot_tables):
        graphs = self.create_abf_bar_graphs(pivot_tables)
        for graph in graphs:
            chart_container = ChartContainer(self.abf_pivot_tables, graph, DARK_COLOR)
            chart_container.pack()

    def create_abf_bar_graphs(self, pivot_tables):
        graphs = []
        fg_color = 'white'
        bg_color = DARK_COLOR
        # bg_color = '#1a1a1a'

        for pivot_table_category, pivot_table in pivot_tables.items():
            fig_pivot = Figure(figsize=(8, 4))
            ax_pivot = fig_pivot.add_subplot(111)

            categories = pivot_table.index.tolist()
            ind_pivot = np.arange(len(categories))
            width_pivot = 0.5
            ys_pivot = pivot_table[('sum', 'ptax_rev_w')].values / 10 ** 6
            title_pivot = 'State: {}. {} by {}'.format(self.winfo_toplevel().parent.general_settings.state,
                                                       'Total Tax Revenue', pivot_table_category)
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

            format_chart(fig_pivot, ax_pivot, title_pivot, bg_color, fg_color)

            graphs.append(fig_pivot)

        return graphs

    def rerun_abf(self):
        settings = self.winfo_toplevel().parent.create_settings()
        abf_output, pivot_tables = self.abf_module.rerun(settings)
        # self.results_window.update_abf_output(abf_output)
        # self.results_window.update_pivot_tables(pivot_tables)
        self.update_abf_output(abf_output, pivot_tables)

    def update_abf_output(self, abf_output, pivot_tables):
        for graph in self.abf_pivot_tables.winfo_children():
            graph.destroy()

        self.display_abf_bar_graphs(pivot_tables)
        self.abf_summary.update_results(abf_output)


class ABFResultsSummary(Frame):
    def __init__(self, parent, output):
        super().__init__(parent, bg=DARK_COLOR, highlightcolor='white', highlightthickness=1, pady=8, padx=10)

        self.income_label = Label(self, text='Total Income:', bg=DARK_COLOR, fg=LIGHT_COLOR, anchor='e',
                                  font='-size 12 -weight bold')
        self.tax_revenue_label = Label(self, text='Total Tax Revenue:', bg=DARK_COLOR, fg=LIGHT_COLOR, anchor='e',
                                       font='-size 12 -weight bold')
        self.benefits_recouped_label = Label(self, text='Tax Revenue Recouped from Benefits:', bg=DARK_COLOR,
                                             fg=LIGHT_COLOR, anchor='e', font='-size 12 -weight bold')

        self.income_value = Label(self, bg=LIGHT_COLOR, fg=DARK_COLOR, anchor='e', padx=5, font='-size 12')
        self.tax_revenue_value = Label(self, bg=LIGHT_COLOR, fg=DARK_COLOR, anchor='e', padx=5, font='-size 12')
        self.benefits_recouped_value = Label(self, bg=LIGHT_COLOR, fg=DARK_COLOR, anchor='e', padx=5, font='-size 12')

        print('Updating ABF results summary values')
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
    def __init__(self, parent, engine_type='Python', se=None):
        super().__init__(parent)
        self.icon = PhotoImage(file='impaq_logo.gif')
        self.tk.call('wm', 'iconphoto', self._w, self.icon)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.parent = parent
        self.se = se

        self.content = Frame(self, width=100)
        self.content.pack(fill=BOTH, expand=True)
        self.progress = DoubleVar(0)
        self.progress_bar = ttk.Progressbar(self.content, orient=HORIZONTAL, length=100, variable=self.progress,
                                            max=100)
        self.progress_bar.pack(fill=X, padx=10, pady=5)
        self.updates_container = Frame(self.content, height=30, bg=VERY_LIGHT_COLOR)
        self.updates_canvas = Canvas(self.updates_container, bg=VERY_LIGHT_COLOR)
        self.updates = Frame(self.updates_container, bg=VERY_LIGHT_COLOR)
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

    def update_progress(self, q, last_progress=0):
        try:  # Try to check if there is data in the queue
            update = q.get_nowait()
            update_type, last_progress = self.parse_update(update, last_progress)
            if update_type == 'done':
                self.parent.show_results()
            elif update_type == 'error':
                self.parent.run_button.enable()
            else:
                self.after(500, self.update_progress, q, last_progress)
        except queue.Empty:
            self.after(500, self.update_progress, q, last_progress)

    def update_progress_r(self, progress_file, last_progress=0):
        while True:
            line = progress_file.readline()
            if line == '':
                self.after(500, self.update_progress_r, progress_file, last_progress)
            else:
                update = ast.literal_eval(line)
                done, last_progress = self.parse_update(update, last_progress)
                if done:
                    self.parent.show_results()
                else:
                    self.after(500, self.update_progress_r, progress_file, last_progress)

    def parse_update(self, update, last_progress):
        update_type = update['type']
        self.update_idletasks()
        if update_type == 'progress':
            progress = update['value']
            self.progress.set(int(progress))
        elif update_type == 'message':
            self.add_update(update['value'], update['engine'])
        elif update_type == 'error':
            error_message = '{}: {}'.format(type(update['value']).__name__, str(update['value']))
            self.add_update(error_message, update['engine'], fg='#e60000')
        elif update_type == 'warning':
            dependency, dependency_version = update['value']
            warning_message = 'Warning: {} library might need to be updated to at least version {}'\
                .format(dependency, dependency_version)
            self.add_update(warning_message, update['engine'], fg='#ff9900')

        return update_type, last_progress

    def add_update(self, update, sim_num, fg='#006600'):
        # If the update is attached to a simulation number, display that number
        if sim_num is not None:
            sim_name = get_sim_name(sim_num)
            update_text = '{}: {}'.format(sim_name, update)
        else:
            update_text = update

        # Create the update label and add it to parent
        label = Message(self.updates, text=update_text, bg=VERY_LIGHT_COLOR,
                        fg=fg, anchor='w', width=350)
        label.pack(padx=3, fill=X)

        # Update the window to display new widget
        self.update()

        # Update scroll region to account for new widget space
        self.updates_canvas.configure(scrollregion=(0, 0, 0, self.updates.winfo_height()))

        # Move scroll area to bottom
        self.updates_canvas.yview_moveto(1)

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

    def on_close(self):
        self.destroy()


# From StackOverflow: https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # milliseconds
        self.wraplength = 250  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, _=None):
        self.schedule()

    def leave(self, _=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        tooltip_id = self.id
        self.id = None
        if tooltip_id:
            self.widget.after_cancel(tooltip_id)

    def showtip(self, _=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a top level window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left', background="#ffffff", relief='solid',
                      borderwidth=1, wraplength=self.wraplength, padx=4, pady=4)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


class AdvancedFrame(Frame):
    def __init__(self, parent, toggle_function, **kwargs):
        super().__init__(parent, **kwargs)
        tip = 'Reveal advanced simulation parameters'
        self.advanced_label = TipLabel(self, tip, text="Advanced Parameters:", bg=DARK_COLOR, fg=LIGHT_COLOR,
                                       font='-size 9 -weight bold')
        self.button_container = Frame(self, highlightbackground='#FFFFFF', borderwidth=1, relief='flat')
        self.on_button = SubtleToggle(self.button_container, text='On', width=3, command=toggle_function)
        self.off_button = SubtleToggle(self.button_container,  text='Off', width=3, command=toggle_function)
        self.off_button.toggle()

        self.advanced_label.pack(side=LEFT, anchor=E, fill=BOTH)
        self.button_container.pack(side=LEFT, anchor=E)
        self.on_button.pack(side=LEFT)
        self.off_button.pack(side=LEFT)


class SubtleButton(Button):
    def __init__(self, parent=None, foreground='#FFFFFF', font='-size 8 -weight bold', pady=2, padx=3,
                 relief='flat', highlightthickness=0, borderwidth=0, background=DARK_COLOR, **kwargs):
        super().__init__(parent, background=background, activebackground=background, foreground=foreground,
                         activeforeground=foreground, font=font, pady=pady, padx=padx, relief=relief,
                         highlightthickness=highlightthickness, borderwidth=borderwidth, **kwargs)


class SubtleToggle(SubtleButton):
    def __init__(self, parent=None, on_background='#00CC00', off_background=DARK_COLOR, **kwargs):
        super().__init__(parent, **kwargs)

        self.toggled_on = False
        self.on_background = on_background
        self.off_background = off_background

    def toggle(self):
        if self.toggled_on:
            self.toggled_on = False
            self.config(background=self.off_background)
        else:
            self.toggled_on = True
            self.config(background=self.on_background)


# The following classes are used so that style options don't have to be reentered for each widget that should be styled
# a certain way.
class BorderButton(Frame):
    def __init__(self, parent=None, custom=False, background='#0074BF', font='-size 11', width=7, pady=0,
                 foreground='#FFFFFF', activebackground='#FFFFFF', relief='flat', borderwidth=1,
                 highlightbackground=LIGHT_COLOR, **kwargs):
        super().__init__(parent, highlightbackground=highlightbackground, relief=relief, borderwidth=borderwidth)

        if not custom:
            self.button = Button(self, foreground=foreground, background=background, font=font, width=width,
                                 relief='flat', activebackground=activebackground, pady=pady, bd=0,
                                 highlightthickness=1, **kwargs)
            self.button.pack()
        else:
            self.button = None

    def add_content(self, content):
        self.button = content
        content.pack()


class RunButton(BorderButton):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, custom=True)
        button = Button(self, foreground='#FFFFFF', background='#ccebff', font='-size 11 -weight bold', width=8,
                        relief='flat', activebackground='#FFFFFF', disabledforeground='#FFFFFF', state=DISABLED,
                        highlightthickness=0, borderwidth=0, pady=1, **kwargs)
        self.add_content(button)
        
    def enable(self):
        self.button.config(state=NORMAL, bg=THEME_COLOR)

    def disable(self):
        self.button.config(state=DISABLED, bg='#99d6ff')


class GeneralEntry(Entry):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, borderwidth=2, highlightbackground='#FFFFFF', relief='flat',
                         highlightthickness=1, font='-size 11', **kwargs)


class NotebookEntry(Entry):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, borderwidth=2, highlightbackground='#999999', relief='flat',
                         highlightthickness=1, font='-size 11', **kwargs)


class TipLabel(Label):
    def __init__(self, parent=None, tip='', cursor='question_arrow', **kwargs):
        super().__init__(parent, cursor=cursor, **kwargs)
        CreateToolTip(self, tip)


class TipCheckButton(ttk.Checkbutton):
    def __init__(self, parent=None, tip='', cursor='question_arrow', onvalue=True, offvalue=False,
                 style='MSCheckbutton.TCheckbutton', **kwargs):
        super().__init__(parent, cursor=cursor, onvalue=onvalue, offvalue=offvalue, style=style, **kwargs)
        CreateToolTip(self, tip)


def run_engines(se, q):
    se.run()

    q.put({'type': 'done', 'value': 'done'})


def run_engines_r(command, counterfactual_command, policy_command):
    subprocess.call(command, shell=True)
    subprocess.call(counterfactual_command, shell=True)
    subprocess.call(policy_command, shell=True)
