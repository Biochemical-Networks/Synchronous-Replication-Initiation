from .CellCycleCombined import CellCycleCombined
from . import PlottingTools as plottingTools
from .ParameterSet import ParameterSet
from . import DataAnalysis as dataAnalysis
from . import DataStorage as dataStorage
from . import FilePathTools as filePathTools

import numpy as np
import pandas as pd
import json
import sys


def start_simulation(file_path, confirm):
    """Checks whether a new directory was made. If yes, then simulation will start. If not, then ask user whether he wants to overwrite existing simulation and 
    start a new one. If user types simply enter, then a new simulation will start. If user types any character and then enter, the simulation will stop.

    Args:
        file_path (string): path to where new directory should be made

    Returns:
        boolean: if True -> any existing folders will be overwritten and a new simulation starts
        if False-> stop simulation
    """
    # If this parameter is set to True, simulation is stopped [default: False]
    start_simulation = True
    new_directory_was_made = filePathTools.make_directory(file_path)
    if not new_directory_was_made:
        print('simulation exists already')
        # if no external confirm parameter was given then ask via terminal
        if confirm == None:
            stop_simulation = input('Should simulation start? (Enter=yes, any character=no)') or False      
            start_simulation = not stop_simulation
        else:
            start_simulation = confirm # if confirm is given then set start simulation to this value
    return start_simulation


def data_frame_from_existing_parameter_set(input_file_path):
    return pd.read_csv(input_file_path, sep=';')

def data_frame_from_new_parameter_set():
    # make an instance of parameterSet class containing all parameters
    parameter_set = ParameterSet()
    return parameter_set.parameter_set_df

def run_simulations(data_frame_params, confirm, file_name, file_path, parameter_path):
    # Based on given root_path, file_name makes paths for storing data
    series_names = filePathTools.make_series_names(file_name, data_frame_params["id"])
    series_paths = filePathTools.make_series_paths(file_path, series_names)
    dataset_paths = filePathTools.make_dataset_paths(series_paths)

    # Check whether there is an output directory already and that the additional confirm is true
    if start_simulation(file_path, confirm):
        print('start with simulation')
        # print parameter set to file
        filePathTools.print_parameter_set_to_file(file_path, parameter_path, data_frame_params)

        # Loop over all series and run a cell cycle for each series
        for i_series in range(data_frame_params["n_series"][0]): 

            # make a dictionary from the parameters set with index i_series
            parameter_dict = data_frame_params.iloc[i_series]  

            run_series_store_data_and_plot(dataset_paths[i_series], series_paths[i_series], parameter_dict)

    else:
        print('No new simulation was started')

def run_series_store_data_and_plot(dataset_series_path, series_path, parameter_dict):
    # create the directory for the simulation i_series and save parameters as json
    filePathTools.make_directory(series_path)
    filePathTools.print_dict_to_json(parameter_dict, series_path)

    # make new instance of cell cycle class and run it
    myCellCycle = CellCycleCombined(parameter_dict)
    myCellCycle.run_cell_cycle()

    # make data frames out of data arrays and store data in hdf5 file for each series
    if parameter_dict["store_time_traces"] == 1:
        dataStorage.saveDataFrameToHdf5(dataset_series_path, myCellCycle.makeDataFrameOfCellCycle(), 'dataset_time_traces')
    dataStorage.saveDataFrameToHdf5(dataset_series_path, myCellCycle.makeDataFrameOfInitEvents(), 'dataset_init_events')
    v_b_v_d_dataframe = myCellCycle.makeDataFrameOfDivisionEvents()
    dataStorage.saveDataFrameToHdf5(dataset_series_path, v_b_v_d_dataframe, 'dataset_div_events')
    # plot a few variables as a function of the simulation time
    if parameter_dict["print_figures"] == 1:    
        dataAnalysis.plot_time_trace_activation_potential(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_origin_opening_probability(series_path, myCellCycle, parameter_dict)
        plot_time_traces_of_simulation(parameter_dict, series_path, myCellCycle)

def plot_time_traces_of_simulation(parameter_dict, series_path, myCellCycle):

    # if parameter_dict.version_of_model == 'simple_hill_funct':
        # dataAnalysis.plot_time_trace_switch_titration_combined(series_path, myCellCycle, parameter_dict)
        # dataAnalysis.plot_time_trace_number_initiators(series_path, myCellCycle, parameter_dict)
    
    if parameter_dict["version_of_model"] == 'switch':
        dataAnalysis.plot_time_trace_dars2_density(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_dars2_rate(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_rida_rate(series_path, myCellCycle, parameter_dict)
        # dataAnalysis.plot_time_trace_synchronized(series_path, myCellCycle, parameter_dict)
        
    elif parameter_dict["version_of_model"] == 'full_model':
        dataAnalysis.plot_time_trace_total_number_and_conc_init(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_titration_site_conc(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_full_model(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_sites_ori(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_concentration_ATP_DnaA_bound(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_concentration_ATP_DnaA_bound_delay(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_oric_bound_DnaA(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_free_conc(series_path, myCellCycle, parameter_dict)
    plottingTools.plot_two_arrays(series_path, myCellCycle.t_init, myCellCycle.v_init_per_ori, r'$t$', r'$v^\ast$', 'init_volume_over_time')

def extract_variables_from_input_params_json(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    return data

def run():
    file_path_input_params_json = sys.argv[1] # path to input_params.json file specifying paths of simulation
    main(file_path_input_params_json)

def run_cluster():
    parameter_path = sys.argv[1] # path to parameters.csv file, including file name (e.g. path/parameters.csv)
    job_id = sys.argv[2] # simulation name 
    series_id = sys.argv[3] # index of simulation
    root_path = sys.argv[4] # location to which simulation output should be stored, if there is already a file with the same name it will be overwritten
    data_frame_params = data_frame_from_existing_parameter_set(parameter_path)
    parameter_name = filePathTools.get_parameter_file_name(parameter_path)
    file_name = '_'.join([job_id, parameter_name])
    file_path = filePathTools.make_file_path(root_path, file_name)
    series_name = '_'.join([job_id, parameter_name, series_id])
    series_path = filePathTools.make_file_path(file_path, series_name)
    dataset_series_path = filePathTools.make_file_path(series_path, 'dataset.h5')
    # data_frame_params = data_frame_from_existing_parameter_set(parameter_path) 
    print('series id:', series_id)
    parameter_dict = data_frame_params.iloc[int(series_id)]
    filePathTools.make_directory(file_path) 
    filePathTools.print_parameter_set_to_file(file_path, parameter_path, data_frame_params)
    run_series_store_data_and_plot(dataset_series_path, series_path, parameter_dict)
    print('simulation finished')

def main(file_path_input_params_json, confirm=None):
    input_param_dict = extract_variables_from_input_params_json(file_path_input_params_json)
    # Based on given root_path and file_name makes paths for storing data
    file_path = filePathTools.make_file_path(input_param_dict["ROOT_PATH"], input_param_dict["FILE_NAME"])
    parameter_path = filePathTools.make_parameter_path(file_path)
    if input_param_dict["MAKE_NEW_PARAMETER_SET"]:
        data_frame_params = data_frame_from_new_parameter_set()
    else:
        print('input file being used')
        data_frame_params = data_frame_from_existing_parameter_set(input_param_dict["INPUT_FILE_PATH"])
    run_simulations(data_frame_params, confirm, input_param_dict["FILE_NAME"], file_path, parameter_path)