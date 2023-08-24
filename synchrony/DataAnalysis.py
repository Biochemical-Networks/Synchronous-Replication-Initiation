import numpy as np
from scipy.optimize import fsolve

from . import PlottingTools as plottingTools
from . import DataStorage as dataStorage

pinkish_red = (247 / 255, 109 / 255, 109 / 255)
green = (0 / 255, 133 / 255, 86 / 255)
dark_blue = (36 / 255, 49 / 255, 94 / 255)
light_blue = (168 / 255, 209 / 255, 231 / 255)
blue = (55 / 255, 71 / 255, 133 / 255)
yellow = (247 / 255, 233 / 255, 160 / 255)
dark_yellow = (235 / 255, 201 / 255, 27 / 255)

def plot_time_trace_activation_potential(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$f(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.activation_potential]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='activation_potential',
                          title=title,
                          h_line_plot_3 = parameter_dict.f_crit,
                          period_blocked = parameter_dict.period_blocked)

def plot_time_trace_origin_opening_probability(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$p_{o}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.origin_opening_probability]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='origin_opening_probability',
                          title=title,
                          h_line_plot_3 = 0.5)

def plot_time_trace_dars2_density(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'[Dars2](t)']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.dars2_density_tot]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3)) + r'$\langle [D2]\rangle = $' + str(np.round(np.average(myCellCycle.dars2_density_tot), 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='dars2_density',
                          title=title)


def plot_time_trace_dars2_rate(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}(t)$', r'$\alpha_{d2}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.activation_rate_dars2_tot]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='dars2_rate',
                          title=title)


def plot_time_trace_rida_rate(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}(t)$', r'$\beta_{rida}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.deactivation_rate_rida_tot]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='rida_rate',
                          title=title)

def plot_time_trace_active_conc(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$[D]_{ATP}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.active_conc]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='active_conc',
                          title=title,
                          h_line_plot_3 = parameter_dict.f_crit)

def plot_time_trace_number_open_origins(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$n_{open}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.number_open_origins]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='number_open_origins',
                          title=title)

def determine_frac_synchronised(number_origins):
    fraction_synchronised = np.zeros(number_origins.size)
    n_ori_of_synchronous_cell = [2, 4, 8, 16, 32, 64]
    for index, value in enumerate(number_origins):
        fraction_synchronised[index] = value in n_ori_of_synchronous_cell
    return fraction_synchronised

def plot_time_trace_synchronized(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$f_{sync}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    fraction_synchronised = determine_frac_synchronised(myCellCycle.n_ori)
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, fraction_synchronised]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))+ r', fraction synchronized: $\langle f_{sync} \rangle =$'+ str(np.round(np.mean(fraction_synchronised), 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='fraction_synchronised',
                          title=title,
                          h_line_plot_3 = 0.5)

def plot_time_trace_total_number_and_conc_init(filepath_series, myCellCycle, parameter_dict):
    label_list = [r'$V(t) \, [\mu m^3]$', r'$N(t)$', r'$[p]_{\rm T}(t)$', r'$[p](t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.N_init, myCellCycle.total_conc, myCellCycle.free_conc]
    color_list = [green, dark_blue, pinkish_red, blue]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))+ r', average total conc: $\langle [D_{T}] \rangle =$'+ str(np.round(np.mean(myCellCycle.total_conc), 3))
    plottingTools.plot_four_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='total_number_and_conc_init',
                          title=title,
                          extra_second_row= myCellCycle.sites_total)

def plot_time_trace_titration_site_conc(filepath_series, myCellCycle, parameter_dict):
    label_list = [r'$V(t) \, [\mu m^3]$', r'$N_{\rm s}(t)$', r'$[p]_{\rm T}(t)$', r'$\rho_{\rm sites}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.sites_total, myCellCycle.total_conc, myCellCycle.sites_total/myCellCycle.volume]
    color_list = [green, dark_blue, pinkish_red, blue]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_four_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='chromosome_density',
                          title=title)

def plot_time_trace_full_model(filepath_series, myCellCycle, parameter_dict):
    label_list = [r'$V(t) \, [\mu m^3]$', r'$[p]_{\rm f}(t)$', r'$f(t)$', r'$[D]_{ATP,f}(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.free_conc, myCellCycle.active_conc / myCellCycle.total_conc, myCellCycle.activation_potential]
    color_list = [green, dark_blue, light_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_four_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='free_active_conc_full_model',
                          title=title)


def plot_time_trace_sites_ori(filepath_series, myCellCycle, parameter_dict):
    label_list = [r'$V(t) \, [\mu m^3]$', r'$[p]_{\rm f}(t)$', r'$f(t)$', r'$n_{\rm s}^{\rm ori}$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.free_conc, myCellCycle.active_conc / myCellCycle.total_conc, myCellCycle.sites_ori]
    color_list = [green, dark_blue, light_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_four_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='origin_sites_cascade_model',
                          title=title)

def plot_time_trace_concentration_ATP_DnaA_bound(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$ATP-D-oriC(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.origin_opening_probability * myCellCycle.sites_ori / myCellCycle.volume]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='origin_bound_ATP_DnaA',
                          title=title)

def plot_time_trace_concentration_ATP_DnaA_bound_delay(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$ATP-D-oriC(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.sites_ori / myCellCycle.volume * (myCellCycle.activation_potential)**parameter_dict.n_origin_sites/(20.0**parameter_dict.n_origin_sites + (myCellCycle.activation_potential)**parameter_dict.n_origin_sites)]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='origin_bound_ATP_DnaA_delay',
                          title=title)

def plot_time_trace_free_conc(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$ATP-D-oriC(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, myCellCycle.free_conc]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='free_conc',
                          title=title)

def plot_time_trace_oric_bound_DnaA(filepath_series, myCellCycle, parameter_dict):
    label_list =[r'$V(t) \, [\mu m^3]$', r'$n_{ori}$', r'$ATP-D-oriC(t)$']
    time_array = myCellCycle.time
    x_axes_list = [time_array, time_array, time_array]
    y_axes_list = [myCellCycle.volume, myCellCycle.n_ori, 1/(1+(parameter_dict.f_crit / myCellCycle.free_conc)**parameter_dict.hill_origin_opening) * myCellCycle.sites_ori / myCellCycle.volume]
    color_list = [green, dark_blue, pinkish_red]
    title = r'doubling rate: $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))
    plottingTools.plot_three_subplots_paper_style(label_list, 
                          x_axes_list, 
                          y_axes_list, 
                          color_list, 
                          myCellCycle.t_div, 
                          myCellCycle.t_init, 
                          parameter_dict.v_init_th, 
                          filepath_series,
                          file_name='origin_bound_DnaA',
                          title=title)