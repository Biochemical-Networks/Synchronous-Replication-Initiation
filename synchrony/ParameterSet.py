import numpy as np
import pandas as pd
import os
import git

class ParameterSet:
    """This class is used to generate the parameter set of a simulation."""
    def __init__(self):
        """Generates the parameter set of a simulation as soon as class is initiated."""
        # These are the parameters of the simulation

        # Make new arrays, used for making heatmap where two parameters are varied
        # hill_coeffs, firing_rate = self.make_two_parameter_arrays_given_two_arrays(np.array([4.5, 5.5, 6.5]), np.array([10, 15, 20, 25, 30, 35, 40, 45, 50]))

        # Make new array, varying both delay and firing rate
        # period_delay, doubling_rate = self.make_two_parameter_arrays_given_two_arrays(np.linspace(0, 0.2, 20),np.array([60/24, 60/28, 60/32, 60/36, 60/40]))
        # print(period_delay, doubling_rate)

        # Make new arrays, used for making heatmap where two parameters are varied
        datA_delay, RIDA_delay = self.make_two_parameter_arrays_given_two_arrays(np.linspace(0, 0.14, 15),np.linspace(0, 0.14, 15))
        print(datA_delay)
        # d2_delay, d1_delay = self.make_two_parameter_arrays_given_two_arrays(np.linspace(0, 0.6, 13),np.linspace(0, 0.6, 13))
        # rate_opening, rate_fire = self.make_two_parameter_arrays_given_two_arrays(np.linspace(15, 1065, 15),np.linspace(15, 1065, 15))
        # print(rate_opening)
        self.n_series = datA_delay.size
        print('total series number: ', self.n_series)
        self.id = np.arange(self.n_series) # makes a numpy array from 0-> n.series for the different simulations

        # Storing git version and commit identifier of code
        try:
            self.git_version = git.Repo('.', search_parent_directories=True).head.object.hexsha
        except:
            print('no git repository for storing git SHA')

        # Parameters of simulation
        self.doubling_rate = np.linspace(1.5, 1.5, self.n_series) #1.5 * np.ones(self.n_series) #in units 1/h, default parameters [low: 0.5, int: 60/35, high: 60/25, max:2.5]
        self.n_cycles = 5000 # number of cell cycles the simulation should do [default: 20]
        self.t_max = self.n_cycles / self.doubling_rate  # maximal time of simulation in units of hours
        self.period_blocked = 0.17 * np.ones(self.n_series) #  0.17 * np.ones(self.n_series)np.array([0, 0.05, 0.1, 0.15, 0.2])#time in hours during which no new DnaA is produced and/or oriC is blocked for initiation [default: 0.17] (is approx 10 minutes)
        self.t_C = 2/3 * np.ones(self.n_series)  # time in hours that it takes to replicate the chromosome, [default: 2/3] (=40 min)
        self.t_D = 1/3 * np.ones(self.n_series)  # time in hours that it takes from the end of replication until division, [default: 1/3] (=20 min)
        self.t_CD = self.t_C + self.t_D  # time in hours that it takes from end of replication until division, [default: 1] (=60 min)
        self.rate_growth = self.doubling_rate * np.log(2)  # growth rate of the cell
        self.rate_rep = 1 / self.t_C  # rate at which a chromosome of length 1 is replicated in units 1/h [default: 1/t_C]
        self.time_step = 0.0001 * np.ones(self.n_series)  # time step in h of simulation time, [default: 0.001], choose such that time_step < 1/maximal_rate_of_simulation
        self.maximal_chromosome_tree_depth = 5 * np.ones(self.n_series)  # If chromosome tree depth larger than this simulation asks whether should abort or simply aborts depending on 'ask_whether_continue_if_too_deep' parameter
        self.ask_whether_continue_if_too_deep = 0 * np.ones(self.n_series) # if 1 -> when chromosome is too deeply nested (depth > 3), asks whether should continue, if 0 -> does not ask but just aborts [default: 0]
        self.print_figures = 0 * np.ones(self.n_series) # if 1 -> prints figures of time traces after the simulation ends, if 0 -> only hdf5 files will be stored
        self.store_time_traces = 0 * np.ones(self.n_series) # if 1 -> saves all time traces to hdf5 file if 0 -> no time traces but only initiation and division evets are stored
        
        # Parameters specifying initial conditions
        self.n_ori_0 = np.ones(self.n_series)  # number of origins at t=0, [default: 1]
        self.v_0 = 0.5 * np.ones(self.n_series)  # initiation volume in cubic micro meters, [default: 0.1]
        self.f_0 = 0.05 * np.ones(self.n_series)  # active fraction at t=0, [default: 0]
        
        # Version of model
        self.version_of_model = 'full_model' 
                                            # "switch"-> replication initiation is triggered when critical concentration of total concentration is reached
                                            # "simple_hill_funct" -> models the activation potential as simple hill function
                                            # "full_model" -> replication is initiated at critical, free ATP-DnaA concentration, initiator is expressed explicitly
        self.cascade_model = 0 * np.ones(self.n_series) # if 1 then cascade model is used, if 0 then no cascade model is used
        self.cascade_correct = 1
        self.additional_open_state = 0 * np.ones(self.n_series)

        # Parameters for initiator potential if self.version_of_model is "simlpe_hill_funct"
        self.v_init_th = 1 * np.ones(self.n_series) # volume at which activation potential is 0.5, [default: 1]
        self.hill_activation_potential = 10 * np.ones(self.n_series) #[default: 10] #hill_coeffs
        self.hill_origin_opening = self.hill_activation_potential # 6.5* np.ones(self.n_series) #np.array([4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]) # 6.5 * np.ones(self.n_series) #10.0 * np.ones(self.n_series) # [default: 1]
        self.n_eff = self.hill_activation_potential * self.hill_origin_opening/2
        self.firing_stochastic = True * np.ones(self.n_series) # [default: True]
        self.n_ori_birth = self.calculate_av_number_origins_at_birth(1/self.doubling_rate, self.t_CD)
        self.v_birth_th = self.calculate_av_V_b_th(self.v_init_th, self.rate_growth, self.t_CD)
        print(self.v_birth_th/ self.n_ori_birth)
        self.origin_firing_rate = 1000 * np.ones(self.n_series) #self.calcuate_max_rate_given_n_eff(self.n_eff, self.rate_growth, self.v_init_th, self.v_birth_th/ self.n_ori_birth) #np.linspace(50, 500, self.n_series) #200 * np.ones(self.n_series) # 1000.0  * np.ones(self.n_series)
        self.origin_open_and_firing_rate = self.origin_firing_rate #1000.0 * np.ones(self.n_series) # [default: 1000]rate_opening #
        print('origin firing rate:', self.origin_open_and_firing_rate)
        # Parameters if self.version_of_model is "switch"
        self.michaelis_const_deact = 5 * np.ones(self.n_series)  # Michaelis constant for DnaA deactivators in units of number per volume [default: 50], for ultra-sensitive regime 5
        self.michaelis_const_act = 5 * np.ones(self.n_series)  # Michaelis constant for DnaA activators in units of number per volume [default: 50], for ultra-sensitive regime 5
        self.total_conc_0 = 400 * np.ones(self.n_series)  # total DnaA concentration in units of number per volume (cubic micrometers^-1) [default: 400]
        self.conc_0 = 400 * np.ones(self.n_series) # total DnaA concentration at the beginning of the simulation (initial condition) [default: 400]
        
        # Parameters for location of datA and DARS1/2 on the chromosome
        self.t_doubling_datA = datA_delay #0.13 * np.ones(self.n_series) #np.linspace(0, 0.2, self.n_series) #0.14 * np.ones(self.n_series) #  time from replication initiation to replication of datA in units of hours [default: 0.05] (=3 min
        self.site_datA = self.t_doubling_datA / self.t_C # relative position of DARS2 on the chromosome
        self.t_doubling_dars2 = 0.25 * np.ones(self.n_series)  # time from replication initiation to replication of DARS2 in units of hours [default: 0.2] (=12 min)
        self.site_dars2 = self.t_doubling_dars2 / self.t_C # relative position of DARS2 on the chromosome
        self.t_doubling_dars1 = 0.4 * np.ones(self.n_series)   # time from replication initiation to replication of DARS1 in units of hours [default: 0.1] (=6 min)
        self.site_dars1 = self.t_doubling_dars1 / self.t_C # relative position of DARS2 on the chromosome

        # Parameter for onset of RIDA
        self.t_onset_RIDA = RIDA_delay  #0.06 * np.ones(self.n_series) # time from replication initiation to replication of DARS1 in units of hours [default: 0.1] (=6 min)
        self.site_onset_RIDA = self.t_onset_RIDA / self.t_C # relative position of RIDA onset on the chromosome
        self.rida_rate_exponent = 1 * np.ones(self.n_series) # for DARS2 and RIDA to have different origin density dependence, include higher exponent in RIDA rate (1 is original model) [default: 1]

        # Parameters for (de)activation rates in LD/LDDR model, respectively
        self.lddr = 1  # if 1 then we use the LDDR model, otherwise parameters of LD model
        if self.lddr == 1:
            self.deactivation_rate_rida = 600 * np.ones(self.n_series)  # RIDA deactivation rate in units per hour [default: 600] (for LDDR model)
            self.deactivation_rate_datA = 600 * np.ones(self.n_series)  # datA deactivation rate (low) in units per hour [default: 600] (for LDDR model)\
            if self.version_of_model == 'full_model':
                self.activation_rate_lipids = 800 * np.ones(self.n_series) # lipid activation rate in units per hour [default: 800] (for LDDR-titration model)
            else:
                self.activation_rate_lipids = 500 * np.ones(self.n_series) # lipid activation rate in units per hour [default: 500] (for LDDR model)
            self.activation_rate_dars2 = 50 * np.ones(self.n_series)  # DARS2 activation rate in units per hour [default: 50] (for LDDR model)
            self.activation_rate_dars1 = 100 * np.ones(self.n_series)  # DARS1 activation rate in units per hour [default: 100] (for LDDR model)
            self.high_rate_datA = 0 * np.ones(self.n_series)  # datA deactivation rate (high) in units per hour [default: 0] (for LDDR model) [total deactivation rate is high_rate_datA+deactivation_rate_datA]
        else:
            self.deactivation_rate_rida = 0 * np.ones(self.n_series)   # RIDA deactivation rate in units per hour [default: 0] (for LD model)
            self.deactivation_rate_datA = 6000 * np.ones(self.n_series)  #deact_rate datA deactivation rate (low) in units per hour [default: 600] (for LD model)
            self.activation_rate_lipids = 8000 * np.ones(self.n_series) #deact_rate/6 * 5  lipid activation rate in units per hour [default: 2755] (for LD model)
            self.activation_rate_dars2 = 0 * np.ones(self.n_series)  # DARS2 activation rate in units per hour [default: 0] (for LD model)
            self.activation_rate_dars1 = 0 * np.ones(self.n_series)  # DARS1 activation rate in units per hour [default: 0] (for LD model)
            self.high_rate_datA = 0 * np.ones(self.n_series)  # datA deactivation rate (high) in units per hour [default: 0] (for LD model)
        self.include_synthesis = 1 * np.ones(self.n_series) # if True, the term \lambda * (1-f) is added when the active fraction is calculated

        # Parameters for cell cycle dependent activation and deactivation rates
        self.t_onset_dars2 = self.t_doubling_dars2  # time in hours from moment of replication initiation to when DARS2 begins being more active [default: t_doubling_dars2]
        self.relative_chromosome_position_onset_dars2 = self.t_onset_dars2 / self.t_C # relative position on the chromosome when DARS2 begins to be more active
        self.t_onset_datA = self.t_doubling_datA  # time in hours from moment of replication initiation to when datA begins being more active [default: 0]
        self.relative_chromosome_position_onset_datA = self.t_onset_datA / self.t_C # relative position on the chromosome when DARS2 begins to be more active
        self.t_offset_datA = 0.2 * np.ones(self.n_series)  # time in hours from moment of replication initiation to when datA stops being more active [default: 0.2]
        self.relative_chromosome_position_offset_datA = self.t_offset_datA / self.t_C # relative position on the chromosome when datA stops to be more active
        self.t_offset_dars2 = self.t_C  # time in hours from moment of replication initdars2iation to when DARS2 stops being more active [default: t_c] (=40 min)
        self.relative_chromosome_position_offset_dars2 = self.t_offset_dars2 / self.t_C # relative position on the chromosome when DARS2 stops to be more active
        
        # titration parameters (only used if version of model is "full_model")
        self.n_c_max_0 = 300.0 * np.ones(self.n_series)
        self.n_origin_sites = self.hill_origin_opening #10  * np.ones(self.n_series)0  * np.ones(self.n_series) #
        self.homogeneous_dist_sites = 1 * np.ones(self.n_series)
        self.rate_synth_sites = self.n_c_max_0 * self.rate_rep  # rate of freeing new sites in 1/h if all titration sites are distributed homogeneously on the chromosome
        self.diss_constant_sites = 1.0 * np.ones(self.n_series) # dissociation constant of DnaA boxes in units per volume (in cubic micrometers^-1) [default: 1]
        
        # initiator gene expression parameters (only used if version of model is "full_model")
        self.michaelis_const_initiator = 200 * np.ones(self.n_series) # dissociation constant of promoter of (ATP-)DDnaA in units per volume (in cubic micrometers^-1) [default: 400]
        self.basal_rate_initiator_0 = 400 * np.ones(self.n_series)  #  basal (ATP-)DnaA expression rate [default: 1200]
        self.basal_rate_initiator = self.basal_rate_initiator_0 * self.rate_growth
        self.hill_coeff_initiator =  5 * np.ones(self.n_series)  # hill coefficient of (ATP-)DDnaA promoter [default:5]
        self.critical_free_active_conc =  50.0 * np.ones(self.n_series) # 30 for cascade, ATP-DnaA concentration in units per volume (in cubic micrometers^-1) at which replication is initiated [default: 200]
        self.finite_dna_replication_rate = 0 # either 1 or 0, if 1 -> effect of finite time to replicate chromosome on gene allocation fraction is included
        self.number_density = 10**6 * np.ones(self.n_series) # number density in growing cell model [default: 10**6]
        self.gene_alloc_fraction_initiator_0 = self.basal_rate_initiator_0 / self.number_density
        self.stochastic_gene_expression = 0 # either 0 or 1, if 1-> gene expression of DnaA is stochastic [default: 0]
        self.noise_strength_total_dnaA = 100 * np.ones(self.n_series) # magnitude of the noise in the rida concentration [default: 100]
        self.block_production = 0 * np.ones(self.n_series) # either 0 or 1, if 1 -> DnaA production is stopped during the blocked period after replication initaition via SeqA [default: 0]
        if self.block_production[0] == 1:
            self.basal_rate_initiator_0 = 600 * np.ones(self.n_series)
            self.basal_rate_initiator = self.basal_rate_initiator_0 * self.rate_growth
        self.block_production_onset = 0.11 * np.ones(self.n_series) #np.linspace(0, 0.18, self.n_series)
        self.block_production_offset = self.period_blocked

        # set the critical fraction depending on the model that is being used
        if self.version_of_model == 'simple_hill_funct':
            self.f_crit = 0.5 * np.ones(self.n_series) # fraction at which activation potential is 0.5
        if self.version_of_model == 'switch':
            self.f_crit = 0.5 * np.ones(self.n_series)
        if self.version_of_model == 'full_model':
            self.f_crit = self.critical_free_active_conc
        
        # determine high_dars2 rate only after f_crit has been set (beause we need f_crit to calculate value of high_dars2_rate)
        self.high_rate_dars2 = self.calculate_dars2_from_rida_rate() # activation rate during the high activity time period of DARS2 

        # parameter that determines whether Dars2 rate becomes independent of chromosome density
        self.dars2_rate_indep_density = 0 * np.ones(self.n_series)

        # Parameters for division
        self.cv_division_position = 0 * np.ones(self.n_series)
        self.independent_division_cycle = 0 * np.ones(self.n_series) # if 1 -> division cycle is not coupled to replication cycle, if 0 -> replication initiation triggers cell division [default: 0]
        self.version_of_independent_division_regulation = 'IDA'  # if division is regulated independently of replication, there are different version to trigger cell division [default: 'IDA']:
                                                        # 'IDA' -> independent double adder, independently added volume from birth to division
                                                        # 'sizer' -> constant division volume independent of replication or birth volume
        self.version_of_coupled_division_regulation = 'cooper'  # if division is triggered by replication initiation, these are the differnt coupling mechanisms: [default: 'cooper']
                                                        # 'cooper' -> cell divides constant tcc after replication initiation
                                                        # 'RDA' -> replication double adder, independently added volume from replication initiation to division

        self.division_volume = self.v_init_th * np.exp(self.rate_growth * self.t_CD) * np.ones(self.n_series) # if replication is controlled independently, then this is the critical added/division size   

    @property
    def parameter_set_df(self):
        """ Returns a data frame of the instance of the ParameterSet class."""
        parameter_set_dict = vars(self)
        return pd.DataFrame.from_dict(parameter_set_dict)
    
    def calculate_dars2_from_rida_rate(self):
        return self.deactivation_rate_rida * self.f_crit * self.total_conc_0 / (self.michaelis_const_deact + self.f_crit * self.total_conc_0) \
               * (self.michaelis_const_act + self.total_conc_0 - self.f_crit * self.total_conc_0) / (self.total_conc_0 - self.f_crit * self.total_conc_0)
    
    def make_two_parameter_arrays_given_two_arrays(self, np_array1, np_array2):
        new_np_array1 = np.array([])
        new_np_array2 = np.array([])
        for idx, value in enumerate(np_array1):
            new_np_array1 = np.append(new_np_array1, np.ones(np_array2.size)* value)
            new_np_array2 = np.append(new_np_array2, np_array2)
        return new_np_array1, new_np_array2

    def calculate_av_V_b_th(self, v_init, growth_rate, T_CC):
        return v_init/2 * np.exp(growth_rate* T_CC)

    def calculate_av_number_origins_at_birth(self, doubling_time, T_CC):
        # this function determines the average number of origins present at birth (depends on relative length of TCC and the doubling time)
        ratio = T_CC / doubling_time
        conditions = [(ratio<1), (ratio>=1)&(ratio<2), (ratio>=2)&(ratio<3), (ratio>=3)&(ratio<4)]
        choiceX = [1, 2, 4, 8]
        n_ori_b = np.select(conditions, choiceX)
        return n_ori_b

    def calcuate_max_rate_given_n_eff(self, n, growth_rate, v_init, v_b):
        return n*growth_rate * np.log(2)/np.log((2 * v_init**n)/(v_b**n+v_init**n))
