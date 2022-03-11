'''Calibration
This module contains the basic functions for calibrating FCS experiments with a standard

Examples:

Attributes:
    given_d: a dictionary containing literature values for the diffusion coefficients of fluorophores used for calibration of the confocal volume
    conc_units: a dictionary allowing users to specify which units of concentration they would like returned, for convenience.
    k_b: Boltzmann's constant
    mu_water: Viscosity of water
'''

from math import pi, sqrt

# The values for the diffusion coefficients of Rhodamine 6G and Cy5 used at the UoN
given_d = {
    'Rhodamine 6G': 2.8*10**(-10),
    'Cy5':3.15*10**(-10)
}

# A dictionary of string: integer pairs for ease of converting concentration units
conc_units = {
    'M': 1,
    'mM': 1000,
    'uM': 1000000,
    'nM': 1000000000
}

# Boltzmann's constant
k_b = 1.38*10**-23

# The viscosity of water
mu_water = 1.0016

def confocal_widths(td: float, d: float, sp: float) -> tuple:
    """Calculates the widths of a confocal volume

    Parameters
    ----------
    td: float
        Recorded dwell time for the calibration species
    d: float
        Known diffusion coefficient for the calibration species
    sp: float
        The recorded structural parameter

    Returns
    -------
    A tuple with the first and second widths of the confocal volume
    """
    w1 = sqrt(4*td*d)
    w2 = w1*sp
    return (w1, w2)

def confocal_volume(w1: float, w2: float) -> float:
    """Calculates the confocal volume

    Parameters
    ----------
    w1: float
        The first width of the confocal volume
    w2: float
        The second width of the confocal volume

    Returns
    -------
    A float of the calculated confocal volume
    """
    return pi**(3/2) * w1**2 * w2 * 1000

def calibrate_fcs(measured_td: float, measured_sp:float, calibration_label) -> float:
    """Calculates the confocal volume from measured parameters
    
    Parameters
    ----------
    measured_td: float
        Recorded dwell time for the calibration species
    measured_sp: float
        Recorded structural parameter
    calibration label
        Either
            A string matching a key in the calibration_label dictionary
        Or
            A float of the desired calibration species' diffusion coefficient
    
    Returns
    -------
    A float of the calculated confocal volume
    """
    if calibration_label in given_d.keys():
        calibration_d = given_d[calibration_label]
    else:
        calibration_d = calibration_label
    return confocal_volume(
        *confocal_widths(
            measured_td,
            calibration_d,
            measured_sp
        )
    )

def calibrated_conc(measured_n: float, confocal_volume: float, units: str = 'nM'):
    """Calculates the concentration of a species
    
    Parameters
    ----------
    measured_n: float
        The measured number of molecules in the confocal volume
    confocal_volume: float
        The calculated confocal volume
    units: str
        A string matching a key in the dictionary of units

    Returns
    -------
    A float of the concentration of the species in the given units
    """
    return (measured_n/(confocal_volume*6.023*10**23))*conc_units[units]

def calc_coeff(w1: float, t_d: float) -> float:
    """Calculate the diffusion coefficient from dwell time
    
    Parameters
    ----------
    w1: float
        width 1 of the confocal volume
    t_d: float
        measured dwell time

    Returns
    -------
    A float of the diffusion coefficient in m^2/second
    """
    return w1**2/(4*t_d)

def calc_hydrodynamic_radius(diff_co: float, viscosity: float, temperature: float) -> float:
    """Calculate the hydrodynamic radius of a diffusing species

    Parameters
    ----------
    diff_co: float
        The diffusion coefficient
    viscosity: float
        The viscosity of the solvent
    temperature: float
        The temperature of the measurement in kelvin
    """
    return (k_b*temperature)/(6*pi*viscosity*diff_co)