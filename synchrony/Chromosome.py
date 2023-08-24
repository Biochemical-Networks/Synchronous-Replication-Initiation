import numpy as np

class Chromosome:
    def __init__(self, parameter_dict, t_end_blocked=0, t_begin_blocked_production=0, t_end_blocked_production=0, active_replication=True, length=0, sites=0, blocked=True, blocked_production = False):
        self.parameter_dict = parameter_dict

        # changing chromosome variables
        self.t_end_blocked = t_end_blocked # if not blocked then 0, if blocked then this gives the time when blocked period is over
        self.t_begin_blocked_production = t_begin_blocked_production # if not blocked then 0, if blocked then this gives the time when blocked production period is starting
        self.t_end_blocked_production = t_end_blocked_production # if not blocked then 0, if blocked then this gives the time when blocked production period is over
        self.length = length
        if self.parameter_dict.homogeneous_dist_sites == 0:
            self.sites = self.parameter_dict.n_c_max_0  # if distribution is inhomogeneous, all sites are created instantaneously after the origin has been replicated
        else:
            self.sites = sites  # if distribution is homogeneous, the number of sites depends on how the chromosome is initiated, either fully replicated or being actively replicated
        self.active_replication = active_replication
        self.division_time = self.parameter_dict.t_max
        if self.parameter_dict.period_blocked > 0:
            self.blocked = blocked # parameter that specifies whether origin is blocked; if False-> not blocked, if True-> blocked
        else:
            self.blocked = False
        self.blocked_production = blocked_production

        # determine whether origin sites are available or not
        if self.blocked:
            self.n_ori_sites = 0
        else:
            self.n_ori_sites = self.parameter_dict.n_origin_sites

        # changing ultrasensitivity parameters
        if self.active_replication ==0:
            self.n_rida = 0
            self.n_dars2 = 1
            self.n_dars1 = 1
            self.n_datA = 1
        else:
            self.n_rida = 0
            self.n_dars2 = 0
            self.n_dars1 = 0
            self.n_datA = 0
        self.n_dars2_high_activity = 0
        self.n_datA_high_activity = 0

    def replicate(self, time):
        if self.active_replication == 1:
            if self.length < 1:
                # update chromosome length
                self.length = self.length + self.parameter_dict.rate_rep * self.parameter_dict.time_step

                # update number of titration sites
                if self.parameter_dict.homogeneous_dist_sites == 1:
                    self.sites = self.sites + self.parameter_dict.rate_synth_sites * self.parameter_dict.time_step
                else:
                    self.sites =  self.parameter_dict.n_c_max_0

                # if length is longer than 1 or number of total sites larger than n_c_max_0, then set length to 1 and total number to maximal total number
                if self.length > 1 or self.sites > self.parameter_dict.n_c_max_0:
                    self.length = 1
                    self.sites = self.parameter_dict.n_c_max_0

                # Check whether datA, dars1 and dars2 are being doubled
                if self.length >= self.parameter_dict.site_datA: #as soon as we reach position of site of datA, the number of datA sites goes from 0 to 1
                    self.n_datA = 1
                if self.length >= self.parameter_dict.site_dars2: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    self.n_dars2 = 1
                if self.length >= self.parameter_dict.site_dars1: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    self.n_dars1 = 1
                if self.length >= self.parameter_dict.site_onset_RIDA: #as soon as we reach position of site of datA, the number of datA sites goes from 0 to 1
                    self.n_rida = 2

                # Check whether datA and dars2 are in high or low activity state
                if self.length >= self.parameter_dict.relative_chromosome_position_onset_dars2: #as soon as we reach position where dars2 becomes more active, the number of high activity dars2 goes up to two
                    self.n_dars2_high_activity = 2
                if self.length >= self.parameter_dict.relative_chromosome_position_onset_datA: #as soon as we reach position where datA becomes more active, the number of high activity datA goes up to two
                    self.n_datA_high_activity = 2
                if self.length >= self.parameter_dict.relative_chromosome_position_offset_dars2: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    self.n_dars2_high_activity = 0
                if self.length >= self.parameter_dict.relative_chromosome_position_offset_datA: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    self.n_datA_high_activity = 0
            else:
                # replication is not active anymore and therefore rida is switched off
                self.active_replication = False
                self.n_rida = 0
                self.division_time = time + self.parameter_dict.t_D
                print('replication was finished, division time was set: ', self.division_time)

    #  checks whether blocked, if condition correct unblocks, returns 0 if not and 1 if it is blocked
    def is_blocked(self, time):
        if time >= self.t_end_blocked:
            self.blocked = False
            return self.blocked
        else:
            self.blocked = True
            return self.blocked

    def set_t_end_blocked(self, t_end_blocked):
        self.t_end_blocked = t_end_blocked

    #  checks whether production is blocked, if condition correct unblocks, returns 0 if not and 1 if it is blocked
    def is_blocked_production(self, time):
        if time >= self.t_end_blocked_production:
            self.blocked_production = False
            return self.blocked_production
        else:
            if  time < self.t_begin_blocked_production:
                self.blocked_production = False
                return self.blocked_production
            else:
                self.blocked_production = True
                return self.blocked_production

    def set_t_end_blocked_production(self, t_end_blocked_production):
        self.t_end_blocked_production = t_end_blocked_production

    def set_t_begin_blocked_production(self, t_begin_blocked_production):
        self.t_begin_blocked_production = t_begin_blocked_production

    def set_division_time(self, division_time):
        self.division_time = division_time
    
    def count_origin_sites(self):        
        if self.blocked:
            self.n_ori_sites = 0
        else:
            self.n_ori_sites = self.parameter_dict.n_origin_sites
        return self.n_ori_sites