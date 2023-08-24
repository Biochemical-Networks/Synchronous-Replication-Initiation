from cgi import test
from tkinter import N
import unittest   # The test framework
import numpy as np
from treelib import Node, Tree

from synchrony.CellCycleCombined import CellCycleCombined
from synchrony.ParameterSet import ParameterSet

class Test_CellCycleCombined(unittest.TestCase):
    def test_grow_cell_volume_doubled(self):
        test_parameter_set = ParameterSet()
        parameter_dict = test_parameter_set.parameter_set_df.iloc[0]
        test_cell_cycle = CellCycleCombined(parameter_dict)
        v_0 = test_cell_cycle.volume[0]
        indx_doubled = np.where(test_cell_cycle.time == 1/parameter_dict.doubling_rate)[0][0]
        for n_step in range(1, indx_doubled+1): # iterate over the entire time array, starting with the second entry (fist entry is initial condition)
            test_cell_cycle.grow_cell(n_step)  # update cell volume
        message = "First volume after doubling time is not equal (to at least 3 integers after comma) to two times the initiation volume!"
        # assertEqual() to check equality of first & second value
        print('expected volume and obtained volume:', 2*v_0, np.round(test_cell_cycle.volume[indx_doubled], 3))
        self.assertEqual(2*v_0, np.round(test_cell_cycle.volume[indx_doubled], 3), message)

    def test_replicate_chromosome(self):
        test_parameter_set = ParameterSet()
        parameter_dict = test_parameter_set.parameter_set_df.iloc[0]
        test_cell_cycle = CellCycleCombined(parameter_dict)
        initial_tree = Tree(test_cell_cycle.chromosome_tree)
        for node_i in initial_tree.expand_tree(mode=Tree.WIDTH):
            if test_cell_cycle.node_is_leaf(node_i):
                test_cell_cycle.initiate_replication(n_step=1,node=node_i)
        # print(parameter_dict.t_C, test_cell_cycle.time)
        # indx_TC = np.where(test_cell_cycle.time == parameter_dict.t_C)
        time_cut = test_cell_cycle.time[test_cell_cycle.time < parameter_dict.t_C]
        for n_step in range(2, time_cut.size+2):
            test_cell_cycle.replicate_chromosomes(n_step)
            test_cell_cycle.update_length(n_step)
        message = "After TC the total length of the chromosome is not equal (to at least 3 integers after comma) to two!"
        # assertEqual() to check equality of first & second value
        print('expected total DNA length and obtained lenth:', 2, np.round(test_cell_cycle.length_total[time_cut.size+1], 3))
        self.assertEqual(2, np.round(test_cell_cycle.length_total[time_cut.size+1], 3), message)

    def test_calculate_probability_given_rate(self):
        print('in calculate_probability_given_rate')
        test_parameter_set = ParameterSet()
        parameter_dict = test_parameter_set.parameter_set_df.iloc[0]
        test_cell_cycle = CellCycleCombined(parameter_dict)
        sample_size = 1000
        rate = 10
        waiting_times = np.ones(sample_size)
        error_allowed = 0.05
        # loop over how many times we want to obtain stochastic waiting time
        for index in range(0, sample_size):
            continue_loop = True
            event_time = 0
            while continue_loop:
                event_time = event_time + test_cell_cycle.parameter_dict.time_step
                if test_cell_cycle.calculate_probability_given_rate(rate):
                    # print('event happened at time', event_time)
                    continue_loop= False
            waiting_times[index] = event_time
        print('all waiting times:', waiting_times)
        rel_error = abs((rate - 1/np.mean(waiting_times))/rate)
        print('measured error and allowed error:', rel_error, error_allowed)
        # error message in case if test case got failed
        message = "measured error of waiting time is not less that allowed error."
        self.assertLess(rel_error, error_allowed, message)

    




if __name__ == '__main__':
    unittest.main()
