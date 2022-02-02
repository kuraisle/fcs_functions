'''Calibration
This module contains the basic functions for calibrating FCS experiments with a standard

Examples:

Attributes:
    given_d: a dictionary containing literature values for the diffusion coefficients of fluorophores used for calibration of the confocal volume
    conc_units: a dictionary allowing users to specify which units of concentration they would like returned, for convenience.
'''

from math import pi, sqrt


given_d = {
    'Rhodamine 6G': 2.8*10**(-10),
    'Cy5':3.15*10**(-10)
}

conc_units = {
    'M': 1,
    'mM': 1000,
    'uM': 1000000,
    'nM': 1000000000
}

k_b = 1.38*10**-23

mu_water = 1.0016

def confocal_widths(td: float, d: float, sp: float) -> tuple:
    '''
    '''
    w1 = sqrt(4*td*d)
    w2 = w1*sp
    return (w1, w2)

def confocal_volume(w1: float, w2: float) -> float:
    '''
    '''
    return pi**(3/2) * w1**2 * w2 * 1000

def calibrate_fcs(measured_td: float, measured_sp:float, calibration_label) -> float:
    return confocal_volume(
        *confocal_widths(
            measured_td,
            given_d[calibration_label],
            measured_sp
        )
    )

def calibrated_conc(measured_n, confocal_volume, units = 'nM'):
    return (measured_n/(confocal_volume*6.023*10**23))*conc_units[units]

def calc_coeff(w1, t_d):
    return w1**2/(4*t_d)

def calc_hydrodynamic_radius(diff_co: float, viscosity: float, temperature: float) -> float:
    return (k_b*temperature)/(6*pi*viscosity*diff_co)