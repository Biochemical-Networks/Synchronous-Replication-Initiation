import numpy as np
from treelib import Node, Tree
import pandas as pd
import random
from scipy.optimize import fsolve

from . import PlottingTools as plottingTools
from .Chromosome import Chromosome

class CellCycleCombined:
    def __init__(self, parameter_dict):
        """ This class is used for running several cell cycles with the given parameters obtained by the dictionary parameter_dict.

        Parameters
        ----------
        parameter_dict : dictionary
            The parameter dictionary is used during this entire class to obtain all relevant cell cycle parameters
        """
        self.parameter_dict = parameter_dict

        # changing variables of one cell line, time goes until t_max
        self.time = np.arange(0, self.parameter_dict.t_max, self.parameter_dict.time_step)
        self.volume = np.ones(self.time.size) * self.parameter_dict.v_0
        self.n_ori = np.ones(self.time.size) * self.parameter_dict.n_ori_0
        self.n_datA = np.ones(self.time.size) * self.parameter_dict.n_ori_0
        self.length_total = np.ones(self.time.size) * self.parameter_dict.n_ori_0  # length per chromosome is one
        self.sites_total = self.length_total * self.parameter_dict.n_c_max_0
        self.sites_ori = np.ones(self.time.size) * self.parameter_dict.n_origin_sites * self.parameter_dict.n_ori_0 
        self.activation_potential = np.ones(self.time.size) * self.parameter_dict.f_0
        self.number_open_origins = np.zeros(self.time.size) # state of the origin switching between open (1) and closed (0)
        self.total_conc = np.ones(self.time.size) * self.parameter_dict.total_conc_0
        self.N_init = self.total_conc *  self.parameter_dict.v_0
        self.N_active = self.total_conc *  self.parameter_dict.v_0 * self.parameter_dict.f_0
        self.free_conc = np.ones(self.time.size) # start by calcuating the free concentration at t=0 (n_step is 1 because function calculates conc of previous time step)
        # self.origin_opening_rate = self.parameter_dict.origin_closing_rate * (self.activation_potential / self.parameter_dict.f_crit)** self.parameter_dict.hill_origin_opening
        self.origin_opening_probability = 1/(1+(self.parameter_dict.f_crit / self.activation_potential)** self.parameter_dict.hill_origin_opening)

        # division variables
        self.t_div = [] # counts the number of division events that have occured
        self.next_division_volume = self.parameter_dict.division_volume # if IDA or sizer for cell division, this variables determines next division volume

        # initialize chromosome tree
        chromosome_0 = Chromosome(self.parameter_dict, 
                                    active_replication=False, 
                                    length=1,
                                    sites=self.parameter_dict.n_c_max_0, 
                                    blocked=False)
        self.chromosome_tree = Tree()
        self.chromosome_tree.create_node("Chromosome_0", "chromosome_0", data=chromosome_0)  # root node

        # store initiation events
        self.t_init = []
        self.v_init = []
        self.n_ori_init = []
        self.v_init_per_ori = []
        self.v_b_before_init = []
        self.p_open_init = []
        self.f_init = []

        # store division events
        self.list_of_division_events_vb_vd_td = []
        self.last_v_b = None # if this is none then do not store a tupel of birth and division volume
        self.last_t_b = None
        self.v_b = np.array([]) # list of all birth volumes

        # store initiation pairs (v_init_x, t_init_x, v_init_y, t_init_y)
        self.list_of_initiation_pairs = []
        self.last_v_and_t_init_mother_node = [] # store here last initiation volume, initiation time and the mother node
        self.surviving_branch = None # stores the branch number (0 or 1) of the branch that is chosen for the next daughter cell (the other branch is discarded)

        # used for checking whether depths of chromosome is too deep
        self.ask_whether_continue_if_too_deep = self.parameter_dict.ask_whether_continue_if_too_deep
        self.ask_again = self.parameter_dict.ask_whether_continue_if_too_deep
        self.abort_simulation = False

        # switch variables
        self.active_conc = np.ones(self.time.size) * self.parameter_dict.f_0 * self.parameter_dict.total_conc_0
        self.activation_rate_lipids_tot = np.ones(self.time.size) * self.parameter_dict.activation_rate_lipids
        self.deactivation_rate_datA_tot = np.ones(self.time.size) * self.parameter_dict.deactivation_rate_datA
        self.activation_rate_dars2_tot = np.ones(self.time.size) * self.parameter_dict.activation_rate_dars2
        self.activation_rate_dars1_tot = np.ones(self.time.size) * self.parameter_dict.activation_rate_dars1
        self.deactivation_rate_rida_tot = np.ones(self.time.size) * self.parameter_dict.deactivation_rate_rida
        self.dars2_density_tot = np.ones(self.time.size) / self.parameter_dict.v_0


    def makeDataFrameOfCellCycle(self):
        return pd.DataFrame({"time": self.time,
                             "volume": self.volume,
                             "N_init": self.N_init,
                             "total_conc": self.total_conc,
                             "n_ori": self.n_ori,
                             "length_total": self.length_total,
                             "sites_total": self.sites_total,
                             "sites_ori": self.sites_ori,
                             "free_conc": self.free_conc,
                             "active_conc": self.active_conc,
                             "activation_potential": self.activation_potential,
                             "origin_opening_probability": self.origin_opening_probability,
                             "number_open_origins": self.number_open_origins,
                             "dars2_density_tot": self.dars2_density_tot,
                             "abort_simulation": self.abort_simulation
                             })

    def makeDataFrameOfInitEvents(self):
         df_init=pd.DataFrame({
            "t_init": self.t_init,
            "v_init": self.v_init,
            "n_ori_init": self.n_ori_init,
            "v_init_per_ori": self.v_init_per_ori,
            "v_b_before_init": self.v_b_before_init,
            "abort_simulation": self.abort_simulation,
            "p_open_init": self.p_open_init,
            "f_init": self.f_init
        })
         print('data_frame initiation event', df_init)
         return df_init

    def makeDataFrameOfDivisionEvents(self):
        return pd.DataFrame(self.list_of_division_events_vb_vd_td, columns=['v_b', 'v_d', 't_d'])
    
    def makeDataFrameOfInitiationPairs(self):
        return pd.DataFrame(self.list_of_initiation_pairs, columns=['v_init_x', 't_init_x', 'v_init_y', 't_init_y'])

    def grow_cell(self, n_step, growth_rate=0):
        """ The volume of the n_step is updated according to exponential growth.

        Parameters ----------
        n_step : double
            The nth step of the simulation
        growth_rate: double
            The growth rate is either given, in this case we use the provided growth rate. If no growth rate is
            given, the growth rate from the parameter dictionary is used to update the volume of the cell
        """
        if growth_rate == 0:
            self.volume[n_step] = self.volume[n_step - 1] + self.volume[
                n_step - 1] * self.parameter_dict.rate_growth * self.parameter_dict.time_step
        else:
            self.volume[n_step] = self.volume[n_step - 1] + self.volume[
                n_step - 1] * growth_rate * self.parameter_dict.time_step

    def replicate_chromosomes(self, n_step):
        """ All chromosomes in the cell that are currently being replicated, are replicated in this step.
            We loop over the chromosome tree and check whether the depth of the tree is the maximal depth.
            If a chromosome is a leave of the tree, we take the chromosome stored at this leaf
            and apply the replication method of this instance of the Chromosome class.

        Parameters
        ----------
        n_step : double
            The nth step of the simulation
        """
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            # all leaves of the tree are actual chromosomes that could initiate replication
            if self.node_is_leaf(node_i):
                self.chromosome_tree[node_i].data.replicate(self.time[n_step])
        self.update_chromosome(n_step)

    def node_is_leaf(self, node):
        return len(self.chromosome_tree.children(node))==0

    def update_sites(self, n_step):
        sum_sites = 0
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            if self.node_is_leaf(node_i):
                sum_sites = sum_sites + self.chromosome_tree[node_i].data.sites
        self.sites_total[n_step] = sum_sites

    def update_sites_ori(self, n_step):
        sum_sites_ori = 0
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            if self.node_is_leaf(node_i):
                sum_sites_ori = sum_sites_ori + self.chromosome_tree[node_i].data.count_origin_sites()
        self.sites_ori[n_step] = sum_sites_ori

    def update_length(self, n_step):
        sum_length = 0
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            if self.node_is_leaf(node_i):
                sum_length = sum_length + self.chromosome_tree[node_i].data.length
        self.length_total[n_step] = sum_length
    
    def update_n_ori(self, n_step):
        n_ori_total = 0
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            if self.node_is_leaf(node_i):
                n_ori_total = n_ori_total + 1
        self.n_ori[n_step] = n_ori_total

    def update_datA(self, n_step):
        n_datA_tot = 0
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            if self.node_is_leaf(node_i):
                n_datA_tot = n_datA_tot + self.chromosome_tree[node_i].data.n_datA
        self.n_datA[n_step] = n_datA_tot
    
    def update_blocked_state(self, n_step):
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            if self.node_is_leaf(node_i):
                self.chromosome_tree[node_i].data.is_blocked(self.time[n_step])
                self.chromosome_tree[node_i].data.is_blocked_production(self.time[n_step])
    
    def update_chromosome(self, n_step):
        self.update_n_ori(n_step)
        self.update_blocked_state(n_step)
        self.update_sites(n_step)
        self.update_sites_ori(n_step)
        self.update_length(n_step)
        self.update_datA(n_step)
    

    def update_total_conc(self, n_step):
        if self.parameter_dict.version_of_model == 'full_model':
            self.total_conc[n_step] = self.N_init[n_step] / self.volume[n_step]
        else:
            self.total_conc[n_step] = self.parameter_dict.total_conc_0

    def produce_initiators(self, n_step, time):
        """ The total number of initiator proteins is updated in this step. If the parameter
            stochastic_gene_expression==1, noise is added to the production rate.
            Proteins in this function are produced according to the growing cell gene expression model. 
            Because the production rate is proportional to the gene copy number, we need to loop
            over all existing chromosomes in the cell.

        Parameters
        ----------
        n_step : double
            The nth step of the simulation
        time : double
            The time of the current simulation step

        Returns
        -------
        noise dictionary : dictionary
            This function returns a dictionary of the amount of initiator produced and of the
            noise in this time step in the initiator.
        """
        dn_init = 0  # this is the total number of initiators added in this step
        noise_init = 0  # this is the noise in the number of initiators added in this step
        active_promoters = 0
        self.update_chromosome(n_step)

        # loop over all nodes of chromosome tree
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            if self.node_is_leaf(node_i):
                # if not blocked, increase number of active promoters in cell by one
                if self.parameter_dict.block_production == 1:
                    if self.chromosome_tree[node_i].data.is_blocked_production(time) == False:
                        active_promoters = active_promoters + 1
                else:
                    active_promoters = active_promoters + 1
        
        # calculate gene allocation fraction in two scenarions: 1. DNA replication rate is finite 2. DNA replication rate is infinite and thus number of origins is a good approx for number of chromosomes in cell 
        if self.parameter_dict.finite_dna_replication_rate == 1:
            gene_fraction = active_promoters / self.length_total[n_step]
        else:
            gene_fraction = active_promoters / self.n_ori[n_step]
        
        # calculate basal initiator production rate
        basal_rate_initiator = self.parameter_dict.basal_rate_initiator * gene_fraction * self.volume[n_step]

        if self.parameter_dict.stochastic_gene_expression == 1:
            noise_init = np.random.normal(0, 1) * np.sqrt(
                2 * self.parameter_dict.noise_strength_total_dnaA * self.parameter_dict.time_step)
        
        dn_init = dn_init + basal_rate_initiator / (1 + (self.free_conc[
                    n_step - 1] / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator) \
                    * self.parameter_dict.time_step
        self.N_init[n_step] = self.N_init[n_step - 1] + dn_init + noise_init
        return {"dn_init": dn_init , "noise_init": noise_init}

    def calculate_free_dnaA_concentration(self, n_step):
        """ Calculates the free initiator concentration for a given volume, total number of sites and total number
            of initiators. The formula for this is obtained via the law of mass action and specified in the SI of
            the paper.

        Parameters
        ----------
        n_step : double
            The nth step of the simulation
        """
        if self.parameter_dict.cascade_model == 0:
            sum = self.parameter_dict.diss_constant_sites + self.sites_total[n_step] / self.volume[n_step] + \
                    self.N_init[n_step] / self.volume[n_step]

            self.free_conc[n_step] = self.N_init[n_step] / self.volume[n_step] - (sum) / 2 + np.sqrt(
                sum ** 2 - 4 * self.sites_total[n_step] / self.volume[n_step] * self.N_init[n_step] /
                self.volume[n_step]) / 2
        else:
            # print(self.total_conc[n_step])
            self.free_conc[n_step] =  fsolve(plottingTools.solve_free_conc, 
                                                x0=[1], 
                                                args=(self.total_conc[n_step], 
                                                        self.sites_total[n_step] / self.volume[n_step], 
                                                        self.sites_ori[n_step] / self.volume[n_step], 
                                                        self.active_conc[n_step-1] / self.total_conc[n_step],
                                                        self.parameter_dict.n_origin_sites, 
                                                        self.parameter_dict.diss_constant_sites, 
                                                        self.parameter_dict.critical_free_active_conc,
                                                        self.parameter_dict.cascade_correct))[0]


    def calculate_activation_potential(self, noise_dict, n_step):
        if self.parameter_dict.version_of_model == 'switch':
            self.calculate_activation_potential_via_switch(n_step)
        elif self.parameter_dict.version_of_model == 'simple_hill_funct':
            self.calculate_instantaneous_activation_potential(n_step)
        elif self.parameter_dict.version_of_model == 'full_model':
            self.calculate_activation_potential_via_full_model(noise_dict, n_step)

        else:
            print('version of model is neither `switch` nor `simple_hill_funct`, stop simulations!')
            exit()
    
    def calculate_instantaneous_activation_potential(self, n_step):
        self.activation_potential[n_step] = 1/(1 + (self.parameter_dict.v_init_th / (self.volume[n_step-1]/self.n_datA[n_step-1]))** self.parameter_dict.hill_activation_potential)

    def update_active_conc(self, n_step):
        self.active_conc[n_step] = self.active_conc[n_step - 1] + (self.calculate_activation_rate(n_step)
            - self.calculate_deactivation_rate(n_step)
            + self.calculate_production_rate_synthesis(n_step)) * self.parameter_dict.time_step        

    def calculate_activation_potential_via_switch(self, n_step):
        self.update_active_conc(n_step)
        self.activation_potential[n_step] = self.active_conc[n_step] / self.total_conc[n_step]

    def calculate_activation_potential_via_full_model(self, noise_dict, n_step):
        self.N_active[n_step] = self.N_active[n_step - 1] + (
                self.calculate_activation_rate(n_step)
                - self.calculate_deactivation_rate(n_step)) * self.volume[n_step] \
                                * self.parameter_dict.time_step \
                                + noise_dict['dn_init'] \
                                + noise_dict['noise_init']
        if self.N_active[n_step] < 0:
            self.N_active[n_step] = 0
        self.active_conc[n_step] = self.N_active[n_step] / self.volume[n_step]
        self.calculate_free_dnaA_concentration(n_step)
        self.activation_potential[n_step] = self.active_conc[n_step] / self.total_conc[n_step] * self.free_conc[n_step]

    def calculate_activation_rate(self, n_step):
        rate_dars2_tot = 0
        rate_dars1_tot = 0
        n_dars2_tot = 0
        n_dars1_tot = 0
        n_dars2_high_tot = 0
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            # check whether node is a leaf
            if self.node_is_leaf(node_i):
                n_dars2_tot = n_dars2_tot + self.chromosome_tree[node_i].data.n_dars2
                n_dars1_tot = n_dars1_tot + self.chromosome_tree[node_i].data.n_dars1
                n_dars2_high_tot = n_dars2_high_tot + self.chromosome_tree[node_i].data.n_dars2_high_activity
        rate_dars2_tot_average = (n_dars2_tot * self.parameter_dict.activation_rate_dars2 + n_dars2_high_tot * self.parameter_dict.high_rate_dars2) / n_dars2_tot
        rate_dars2_tot = rate_dars2_tot_average * n_dars2_tot / self.volume[n_step - 1]
        rate_dars1_tot = self.parameter_dict.activation_rate_dars1 * n_dars1_tot / self.volume[n_step - 1]
        if self.parameter_dict.dars2_rate_indep_density:
            rate_dars2_tot = rate_dars2_tot / (n_dars2_tot/ self.volume[n_step - 1])
        self.dars2_density_tot[n_step] = n_dars2_tot/ self.volume[n_step - 1]
        self.activation_rate_dars2_tot[n_step] = rate_dars2_tot
        self.activation_rate_dars1_tot[n_step] = rate_dars1_tot
        self.activation_rate_lipids_tot[n_step] = self.parameter_dict.activation_rate_lipids
        rate_activate_tot = (self.parameter_dict.activation_rate_lipids + rate_dars2_tot + rate_dars1_tot) * (
                self.total_conc[n_step-1] - self.active_conc[n_step - 1]) / (
                       self.parameter_dict.michaelis_const_act + self.total_conc[n_step-1] - self.active_conc[n_step - 1])
        return rate_activate_tot

    def calculate_deactivation_rate(self, n_step):
        rate_datA = 0
        n_rida = 0
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            # check whether node is a leaf
            if self.node_is_leaf(node_i):
                rate_datA = rate_datA + self.chromosome_tree[node_i].data.n_datA * self.parameter_dict.deactivation_rate_datA \
                            + self.chromosome_tree[node_i].data.n_datA_high_activity * self.parameter_dict.high_rate_datA
                n_rida = n_rida + self.chromosome_tree[node_i].data.n_rida
        rate_rida = self.parameter_dict.deactivation_rate_rida * (n_rida / self.volume[n_step - 1]) ** self.parameter_dict.rida_rate_exponent
        self.deactivation_rate_datA_tot[n_step] = rate_datA  / self.volume[n_step - 1]
        self.deactivation_rate_rida_tot[n_step] = rate_rida
        rate = (rate_datA / self.volume[n_step - 1] + rate_rida)  * self.active_conc[n_step - 1] / (
                self.parameter_dict.michaelis_const_deact + self.active_conc[n_step - 1])
        return rate

    def calculate_production_rate_synthesis(self, n_step):
        if self.parameter_dict.include_synthesis == 1:
            rate = self.parameter_dict.rate_growth * (self.total_conc[n_step] - self.active_conc[n_step - 1])
        else:
            rate = 0
        return rate

    def calculate_origin_opening_probability(self, n_step):
        self.origin_opening_probability[n_step] =1/(1+(self.parameter_dict.f_crit / self.activation_potential[n_step-1])** self.parameter_dict.hill_origin_opening)

    def check_whether_initiate(self, n_step):
        tree_before_init = Tree(self.chromosome_tree)
        for node_i in tree_before_init.expand_tree(mode=Tree.WIDTH):
            # check whether node is a leaf
            if self.node_is_leaf(node_i):
                if not self.chromosome_tree[node_i].data.is_blocked(self.time[n_step]):
                    if self.parameter_dict.firing_stochastic:
                        if self.calculate_whether_initiate_given_rate(self.parameter_dict.origin_open_and_firing_rate * self.origin_opening_probability[n_step]):
                            if self.parameter_dict.additional_open_state ==1:
                                if self.calculate_whether_initiate_given_rate(self.parameter_dict.origin_firing_rate):
                                    self.initiate_replication(n_step, node_i)
                            else:
                                self.initiate_replication(n_step, node_i)
                    else:
                        if self.deterministic_initiation_probability(n_step):
                            self.initiate_replication(n_step, node_i)                        

    def deterministic_initiation_probability(self, n_step):
        if self.activation_potential[n_step]>= self.parameter_dict.f_crit:
            return True
        else:
            return False

    def calculate_whether_initiate_given_rate(self, rate):
        prob = rate * self.parameter_dict.time_step
        random_number = random.random()
        return prob >= random_number

    def update_and_store_synchronous_events(self, n_step, node):
        # only store event if it is not the top of the tree
        if self.chromosome_tree.parent(node) is not None:
            # node is not the parent node
            if len(self.last_v_and_t_init_mother_node) == 0:
                # currently no initiation event is stored
                self.last_v_and_t_init_mother_node.append((self.volume[n_step], self.time[n_step], self.chromosome_tree[node].tag))
                # print("Synchronous event stored: ", self.last_v_and_t_init_mother_node)
            else:
                # at least one initiation event is stored
                # print('list not empty', len(self.last_v_and_t_init_mother_node))
                # make hard copy of list to iterate it without it changing
                list_stored_init_events = self.last_v_and_t_init_mother_node.copy()
                # loop over list of stored initiation events
                for item in range(0, len(list_stored_init_events)):
                    # check whether cell has ever divided
                    if self.last_t_b is not None:
                        # the cell has divided at least once
                        # print("the cell has divided at least once")
                        # test whether the stored initiation event i has happened after cell division
                        if list_stored_init_events[item][1] < self.last_t_b:
                            # the cell has divided since this initiation event was stored, complicated case
                            # print("the cell has divided since this initiation event was stored")
                            # test whether the stored node is the mother of this initiation event
                            # print('surviving branch and current node index', self.surviving_branch, self.return_only_last_index_of_node_tag(node))
                            if self.surviving_branch == self.return_only_last_index_of_node_tag(node):
                                # the stored node is the mother of the current node, delete stored initiation event
                                self.last_v_and_t_init_mother_node.pop(item)
                            else:
                                # the stored node is a sibling
                                # save the tupel
                                print('stored node is not the mother')
                        else:
                            # the cell has not divided since this initiation event was stored
                            # print('Test whether sibling', self.chromosome_tree.siblings(node))
                            # test whether the stored initiation event i is a sibling of this node
                            if list_stored_init_events[item][2] == self.chromosome_tree.siblings(node)[0].tag: # take list element 0 of siblings because there is maximally one sibling
                                # the node that just initiated is a sibling of the stored node
                                # print('the stored node is a sibling')
                                # store both together and delete stored node
                                # print('store initiation pair and delete stored node')
                                self.list_of_initiation_pairs.append((list_stored_init_events[item][0], 
                                                                    list_stored_init_events[item][1],
                                                                    self.volume[n_step], 
                                                                    self.time[n_step]))
                                # print('stored init events before pop', self.last_v_and_t_init_mother_node)
                                self.last_v_and_t_init_mother_node.pop(item)
                                # print('stored init events after pop', self.last_v_and_t_init_mother_node)
                                # print('Initiation pairs:', self.list_of_initiation_pairs)
                            else:
                                # the node that just initiated is no sibling of the stored node
                                # store initiation event
                                self.last_v_and_t_init_mother_node.append((self.volume[n_step], self.time[n_step], self.chromosome_tree[node].tag))


                    else:
                        # the cell has never divided
                        # print("the cell has never divided")
                        # print('Test whether sibling', self.chromosome_tree.siblings(node))
                        if self.last_v_and_t_init_mother_node[item][2] == self.chromosome_tree.siblings(node)[0].tag:
                            # the node that just initiated is a sibling of the stored node
                            # store both together and delete stored node
                            # print('store initiation pair and delete stored node')
                            self.list_of_initiation_pairs.append((list_stored_init_events[item][0], 
                                                                list_stored_init_events[item][1],
                                                                self.volume[n_step], 
                                                                self.time[n_step]))
                            # print('stored init events before pop', self.last_v_and_t_init_mother_node)
                            self.last_v_and_t_init_mother_node.pop(item)
                            # print('stored init events after pop', self.last_v_and_t_init_mother_node)
                            # print('Initiation pairs:', self.list_of_initiation_pairs)
                                

    
    def initiate_replication(self, n_step, node):
        print('INITIATE REPLICATION at step ', n_step)
        self.store_initiation_event(n_step)
        print("Tree before initiation: ")
        self.chromosome_tree.show()

        # update and store synchrous events
        self.update_and_store_synchronous_events(n_step, node)

        # make part of tag for daughter nodes
        mother_node_name = self.remove_index_from_node_tag(node, 0)
        print('mother node name:', mother_node_name)

        # make the tags and identifiers of two new chromosome leaves
        daughter_node_tag_0 = "chromosome_" + str(mother_node_name) + '_' + str(0)
        daughter_node_identifier_0 = "Chromosome_" + str(mother_node_name) + '_' + str(0)
        daughter_node_tag_1 = "chromosome_" + str(mother_node_name) + '_' + str(1)
        daughter_node_identifier_1 = "Chromosome_" + str(mother_node_name) + '_' + str(1)

        # make new chromosome and block both chromosomes
        chromosome_0 = self.chromosome_tree[node].data
        t_end_blocked = self.time[n_step] + self.parameter_dict.period_blocked
        t_begin_blocked_production = self.time[n_step] + self.parameter_dict.block_production_onset
        t_end_blocked_production = self.time[n_step] + self.parameter_dict.block_production_offset
        chromosome_0.set_t_end_blocked(t_end_blocked)
        chromosome_0.set_t_begin_blocked_production(t_begin_blocked_production)
        chromosome_0.set_t_end_blocked_production(t_end_blocked_production)
        chromosome_1 = Chromosome(self.parameter_dict, t_end_blocked=t_end_blocked, t_begin_blocked_production=t_begin_blocked_production, t_end_blocked_production=t_end_blocked_production)

        # create two new leaves of the tree
        self.chromosome_tree.create_node(daughter_node_identifier_0, daughter_node_tag_0,
                                                     parent=node, data=chromosome_0)
        self.chromosome_tree.create_node(daughter_node_identifier_1, daughter_node_tag_1,
                                                     parent=node, data=chromosome_1)

        print("Tree after initiation: ")
        self.chromosome_tree.show()

        # check whether depth of tree is greater 3 and ask if simulation should be stopped in that case
        self.check_depth_tree()

        # update chromosome variables
        self.update_chromosome(n_step)

    def store_initiation_event(self, n_step):
        self.t_init.append(self.time[n_step])
        self.v_init.append(self.volume[n_step])
        self.n_ori_init.append(self.n_ori[n_step])
        self.v_init_per_ori.append(self.volume[n_step] / self.n_ori[n_step])
        self.p_open_init.append(self.origin_opening_probability[n_step])
        self.f_init.append(self.activation_potential[n_step])
        try:
            self.v_b_before_init.append(self.v_b[-1])
        except:
            self.v_b_before_init.append(0)
            print('last birth volume could not be obtained')
    
    def return_only_last_index_of_node_tag(self, node):
        node_name_list = self.chromosome_tree[node].tag.split('_')
        return node_name_list[-1]
    
    def remove_index_from_node_tag(self, node, idx):
        node_name_list = self.chromosome_tree[node].tag.split('_')
        del node_name_list[idx]
        return '_'.join(node_name_list)
    
    def check_depth_tree(self):
        stop_simulation = True
        ask_again = True
        if self.chromosome_tree.depth()> self.parameter_dict.maximal_chromosome_tree_depth:
            if self.ask_whether_continue_if_too_deep ==1:
                stop_simulation = input('Depth of tree was '+str(self.chromosome_tree.depth())+' (> maximal_chromosome_tree_depth), should simulation be stopped? (Any character=yes, Enter=No)') or False
                print('stop_simulation', stop_simulation)
                if stop_simulation:
                    self.abort_simulation = True
                    # exit()
                if not stop_simulation:
                    ask_again = input('Should we ask the next time again (Any character=yes, Enter=No)?') or False
                    if not ask_again:
                        self.ask_again = ask_again
            else:
                self.abort_simulation = True

    def check_whether_divide(self,n_step):
        divide = False
        if not self.parameter_dict.independent_division_cycle:
            for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
                if self.node_is_leaf(node_i):
                    if self.time[n_step] >= self.chromosome_tree[node_i].data.division_time:
                        divide = True
                        self.chromosome_tree[node_i].data.division_time = self.parameter_dict.t_max
        else:
            if self.volume[n_step] >= self.next_division_volume:
                divide = True
        return divide


    def divide(self, n_step):
        """This function divides all cell variables by two and throws away one of the two chromosomes at random. We only follow one cell which means that at division we
        need to discard one out of two. 

        Parameters
        ----------
        n_step : int
            the simulation step
        """
        print('DIVIDE at time ', self.time[n_step])
        print('Tree before division: ')
        self.chromosome_tree.show()

        # save division event
        self.t_div.append(self.time[n_step])
        self.save_tuple_v_b_v_d(n_step)

        # divide cell volume
        rel_div_position_error = self.divide_volume(n_step)

        # store new birth parameters
        self.store_new_birth_volume(n_step)

        # divide number of initiators
        self.divide_n_initiator(n_step, rel_div_position_error)

        #  split tree in two decide at random which of the two subtrees will be kept for next cycle
        self.surviving_branch = np.random.random_integers(low=0, high=1) # generates random interger of either 0 or 1
        print('surviving branch: ', self.surviving_branch)
        # make new tree containing only the selected chromosome
        self.chromosome_tree = self.chromosome_tree.remove_subtree(self.chromosome_tree.root+'_'+str(self.surviving_branch))

        # rename nodes in tree
        self.rename_all_nodes_after_division()

        # update chromosome variables of cell cycle
        self.update_chromosome(n_step)

        # set next division volume if independent division cycle
        if self.parameter_dict.independent_division_cycle:
            self.calculate_new_division_volume()
        
        # print tree after division
        print('Tree after division: ')
        self.chromosome_tree.show()

    def save_tuple_v_b_v_d(self, n_step):
        if self.last_v_b is None:
            return
        else:
            self.list_of_division_events_vb_vd_td.append((self.last_v_b, self.volume[n_step], self.time[n_step]))

    def divide_volume(self, n_step):
        rel_div_position_error = 0

        # divide volume by two if no error in division volume
        if self.parameter_dict.cv_division_position == 0:
            self.volume[n_step] = self.volume[n_step] / 2
        else:
            if self.parameter_dict.single_division_error == 1:
                if len(self.t_div) == self.parameter_dict.cycle_with_error:
                    rel_div_position_error = self.parameter_dict.cv_division_position
                    print('length of t_div was one and division volume with error', self.t_div)
                    self.volume[n_step] = self.volume[n_step] / 2 + rel_div_position_error * self.volume[n_step]
                else:
                    print('length of t_div was not one and division volume without error', self.t_div)
                    self.volume[n_step] = self.volume[n_step] / 2
            else:
                rel_div_position_error = self.return_division_position_error()
                self.volume[n_step] = self.volume[n_step] / 2 + rel_div_position_error * self.volume[n_step]
        return rel_div_position_error

    def divide_n_initiator(self, n_step, relative_division_error):
        self.N_init[n_step] = self.N_init[n_step] / 2 + relative_division_error * self.N_init[n_step]
        self.N_active[n_step] = self.N_active[n_step] / 2 + relative_division_error * self.N_active[n_step]


    def return_division_position_error(self):
        if self.parameter_dict.cv_division_position == 0:
            return 0
        else:
            return np.random.normal(0, self.parameter_dict.cv_division_position)

    def return_rand_binomial_distributed_number(self, prob, n_tot):
        k_rand = np.random.binomial(n_tot, prob)
        return k_rand

    def store_new_birth_volume(self, n_step):
        self.v_b = np.append(self.v_b, self.volume[n_step])
        self.last_v_b = self.volume[n_step]
        self.last_t_b = self.time[n_step]

    def rename_all_nodes_after_division(self):
        # loop over tree, rename all tags, make dictionary with identifiers
        identifier_dict = {}
        for node_i in self.chromosome_tree.expand_tree(mode=Tree.WIDTH):
            # rename tag
            self.chromosome_tree[node_i].tag = self.remove_index_from_node_tag(node_i, 1)

            # make dictionary for identifiers
            identifier_list = self.chromosome_tree[node_i].identifier.split("_")
            del identifier_list[1]
            new_identifier = '_'.join(identifier_list)

            identifier_dict[self.chromosome_tree[node_i].identifier] = new_identifier

        # loop over identifier dictionary and rename all identifiers
        for key in identifier_dict:
            self.chromosome_tree.update_node(key, identifier=identifier_dict[key])

    def calculate_new_division_volume(self):
        if self.parameter_dict.version_of_independent_division_regulation == 'sizer':
            self.next_division_volume  = self.parameter_dict.division_volume
        elif self.parameter_dict.version_of_independent_division_regulation == 'IDA':
            self.next_division_volume = self.last_v_b + self.calculate_noisy_added_volume()
        else:
            print('Error, independent division control, but neither sizer nor IDA!')
            exit()

    def run_cell_cycle(self):
        """ This is the main function of this class and runs the cell cycle by updating all variables for every time step. """
        # iterate over the entire time array, starting with the second entry (fist entry is initial condition)
        for n_step in range(1, self.time.size): 
            # update cell volume
            self.grow_cell(n_step)
            
            # replicate chromosomes and update all cell cycle parameters related to the chromosome
            self.replicate_chromosomes(n_step)

            # update total number of initiators in cell
            noise_dict =self.produce_initiators(n_step, self.time[n_step])

            # update the total DnaA concentration
            self.update_total_conc(n_step)
            # self.calculate_free_dnaA_concentration(n_step)

            # compute activation potential, should be done AFTER new volume was calculated
            self.calculate_activation_potential(noise_dict, n_step)
            self.calculate_origin_opening_probability(n_step)
            
            # check whether initiate and if yes the initiate new round of replication
            self.check_whether_initiate(n_step) 

            # check whether divide and if yes divide
            if self.check_whether_divide(n_step):
                self.divide(n_step)
            if self.abort_simulation:
                print('cell cycle simulation is aborted now')
                break
        return

