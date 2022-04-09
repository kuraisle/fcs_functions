'''FCS Objects

Classes
    - FcsData
    - FcsFit
    - Confocor3Fcs
Values
    - numeric_parameters
Functions
    - break_tab
    - gen_field
    - create_fields
    - make_array
    - average_time_series

TODO:
    - Add raw file integration
    - implement a class for fits (FcsFit)
'''
import numpy as np
import matplotlib.pyplot as plt
from . import calibration
from . import raw_functions

def break_tab(field: str) -> list:
    """Break a field by its tabs
    
    ConfoCor3 fcs files are broken into fields by indentation. This is a recursive function that will break a string down by indentation level.
    Each run, a 'field' will be broken by lines. Each time there's a further indentation level, the field will be broken into sub-fields. Anything else will be a str
    
    Parameters
    ----------
    field: str
        A string to be broken by indentation
        
    Returns
    -------
    A list of fields
    """
    tabs = [len(x) - len(x.lstrip('\t')) for x in field]
    if any(map(lambda x: x > 0, tabs)):
        subfields = []
        indented = []
        for index, indent in enumerate(tabs):
            if indent == 0:
                if indented:
                    tab_removed = [x[1:] for x in indented]
                    subfields[-1].append(break_tab(tab_removed))
                    indented = []
                subfields.append([field[index]])
            else:
                indented.append(field[index])
        if indented:
            tab_removed = [x[1:] for x in indented]
            subfields[-1].append(break_tab(tab_removed))
        return subfields
    else:
        return field

def gen_field(field_string: str, field_dict: dict) -> None:
    """Generates a dictionary entry from a string

    Values in ConfoCor3 fcs files are shown by 'Key= Value'. This adds a key: value pair for this to the target dictionary

    Parameters
    ----------
    field_string: str
        A string to be converted into a dictionary
    field_dict: dict
        A dictionary for the key:value pair to be added to

    Returns
    -------
    None
    """
    name, field = field_string.split('=')
    field_dict[name[:-1]] = field[:-1].lstrip()

def create_fields(field_list: str) -> dict:
    """Creates a dictionary from ConfoCor3 values

    Parameters
    ----------
    field_list: str
        A list of strings for conversion to a dictionary
    
    Returns
    -------
    A dictionary of the values in the field list
    """
    fields = {}
    for item in field_list:
        if len(item) == 1:
            if '=' in item[0]:
                gen_field(item[0], fields)
    return fields

def make_array(array_list: list) -> 'np.array':
    """Converts a list of strings into an array
    
    Parameters
    ----------
    array_list: list
        A list of strings where each string is a row of an array, with values, separated by tabs, convertible to floats
    
    Returns
    -------
    A numpy array with a row for each item of the original list
    """
    out = []
    for item in array_list:
        item_str = item[0][:-3].split('\t')
        item_flt = [float(x) for x in item_str]
        out.append(item_flt)
    return np.array(out)

def average_time_series(data):
    """
    """
    all_times = [set(rep[:,0]) for rep in data]
    times = all_times[0]
    for rep in all_times[1:]:
        times = times.intersection(rep)
    times_array = np.array(sorted(times))
    new_times = []
    for rep in data:
        new_times.append(rep[np.isin(rep[:, 0], times_array), 1])
    
    new_times = np.array(new_times)

    return {
        'time': times_array,
        'mean': np.mean(new_times, axis = 0),
        'stddev': np.std(new_times, axis = 0)
    }

numeric_parameters = ['UnitFactor', 'Precision', 'Minimum', 'Maximum', 'StartValue', 'LinkIndex', 'ResultValid', 'Result', 'StandardDeviation']

class FcsData(object):
    def __init__(self, entry: list) -> None:
        entry_data_dict = create_fields(entry)
        labels = []

        for index, item in enumerate(entry):
            if len(item) == 1:
                label = item[0].split(' ')[0]
                if label[-5:] == 'Array':
                    array_length = int(item[0].split(' ')[-2])
                    array_list = entry[index+1:index+1+array_length]
                    array = make_array(array_list)
                    entry_data_dict[label] = array
                    labels.append(label[:-5])
            else:
                if item[0][:17] == 'BEGIN Acquisition':
                    entry_acquisition_dict = create_fields(item[1][0][1])

                elif item[0][:9] == 'BEGIN Fit':
                    entry_fit_dict = create_fields(item[1][0][1])
                    parameters_dict = {}
                    for entry in item[1][0][1]:
                        if entry[0][:15] == 'BEGIN Parameter':
                            listified = [[x] for x in entry[1]]
                            parameter = create_fields(listified)
                            parameters_dict[parameter['Identifier']] = parameter
                    for id, par in parameters_dict.items():
                        for numeric in numeric_parameters:
                            if numeric in par.keys():
                                parameters_dict[id][numeric] = float(parameters_dict[id][numeric])
                    entry_fit_dict['Parameters'] = parameters_dict
        
        self.data = entry_data_dict
        self.datalabels = labels
        self.acquisition = entry_acquisition_dict
        self.fit = entry_fit_dict
    
    def get_fit_parameters(self):
        if self.fit['Parameters']:
            return dict([(x, y['Result']) for x,y in self.fit['Parameters'].items()])
        else:
            return 'No model fitted'
    
    premade_plots = [
        'ACF',
        'PCH'
    ]

    def plot(self, axis = None, plot_type = 'CorrelationArray', **kwargs):
        if plot_type in self.premade_plots:
            if plot_type == 'ACF':
                fig, ax = plt.subplots(nrows=2, figsize = [8,6], gridspec_kw={'height_ratios': [3,1]})

                acf_data = self.data['CorrelationArray']
                cr_data = self.data['CountRateArray']

                ax[0].plot(acf_data[:,0], acf_data[:,1])
                ax[1].plot(cr_data[:,0], cr_data[:,1])

                ax[0].set_xscale('log')

                plt.tight_layout()
        else:
            plot_data = self.data[plot_type]
            axis.plot(plot_data[:,0], plot_data[:,1], **kwargs)

    def link_raw(self, path: str) -> None:
        self.raw = raw_functions.RawFile(path)



class FcsFit(object):
    pass

class Confocor3FCS(object):
    def __init__(self, path: str) -> None:
        with open(path, 'r') as f:
            data_in = f.readlines()
        tab_broken = break_tab(data_in)
        if tab_broken[0] == ['Carl Zeiss ConfoCor3 - measurement data file - version 3.0 ANSI\n']:
            data = tab_broken[1][1]
            top_level_fields = create_fields(data)

            self.info = {
                'Name': top_level_fields['Name'],
                'Comment': top_level_fields['Comment'],
                'Average Flags': top_level_fields['AverageFlags'].split('|'),
                'Sort Order': top_level_fields['SortOrder'].split('-')
            }

            entries = [x[1][0][1] for x in data if x[0][:5] == 'BEGIN']

            self.data = {}
            self.average = FcsData(entries[-1])

            for entry_no, entry in enumerate(entries[:-1]):
                self.data['Repeat ' + str(entry_no+1)] = FcsData(entry)
            
            self.fits = dict([(entry_id, entry.get_fit_parameters()) for entry_id, entry in self.data.items()])
            self.fits['Average'] = self.average.get_fit_parameters()
            self.confocal_volume = None
            
        else:
            print('Not a Confocor3 FCS file')
    
    def link_raw(self, repeat: str, path: str) -> None:
        self.data[repeat].link_raw(path)

    def calibrate_by(self, calibration_label, verbose = False):
        self.confocal_volume = calibration.calibrate_fcs(
            self.average.fit['Parameters']['Translation diffusion time species 1']['Result'],
            self.average.fit['Parameters']['Translation structural parameter']['Result'],
            calibration_label
        )
        self.confocal_widths = calibration.confocal_widths(
            self.average.fit['Parameters']['Translation diffusion time species 1']['Result'],
            calibration.given_d[calibration_label],
            self.average.fit['Parameters']['Translation structural parameter']['Result']
        )

    def calibrate(self, calibration_read, units = 'nM'):
        self.confocal_volume = calibration_read.confocal_volume
        self.confocal_widths = calibration_read.confocal_widths

        self.calibrated_concs = {}

        self.diffusion_coefficients = {}

        for entry_id, entry in self.fits.items():
            if type(entry) == dict:
                self.calibrated_concs[entry_id] = calibration.calibrated_conc(entry['Number of molecules'], self.confocal_volume, units)
                diff_species = [(id, calibration.calc_coeff(self.confocal_widths[0], par)) for id, par in entry.items() if 'diffusion time' in id]
                if len(diff_species) > 1:
                    self.diffusion_coefficients[entry_id] = dict(diff_species)
                else:
                    self.diffusion_coefficients[entry_id] = diff_species[0][1]
            else:
                self.diffusion_coefficients[entry_id] = 'No model fit'
        
        if type(self.average.fit) == dict:
            self.calibrated_concs['Average'] = calibration.calibrated_conc(self.average.fit['Parameters']['Number of molecules']['Result'], self.confocal_volume, units)
            diff_species = [(id, calibration.calc_coeff(self.confocal_widths[0], par['Result'])) for id, par in self.average.fit['Parameters'].items() if 'diffusion time' in id]
            if len(diff_species) > 1:
                self.diffusion_coefficients['Average'] = dict(diff_species)
            else:
                self.diffusion_coefficients['Average'] = diff_species[0][1]
    
    def calc_hydrodynamic_radii(self, viscosity: float = calibration.mu_water, temperature: float = 297.15):
        if self.calibrated_concs:
            self.hydrodynamic_radii = {}
            for entry_id, diff_co in self.diffusion_coefficients.items():
                if type(diff_co) == dict:
                    self.hydrodynamic_radii[entry_id] = dict([(x, calibration.calc_hydrodynamic_radius(y, viscosity, temperature)) for x,y in diff_co.items()])
                elif type(diff_co) == float:
                    self.hydrodynamic_radii[entry_id] = calibration.calc_hydrodynamic_radius(diff_co, viscosity, temperature)
                else:
                    self.hydrodynamic_radii[entry_id] = 'No model fit'
        else:
            print('Calibrate diffusion coefficients first')
    
    premade_plots = ['ACF', 'PCH']

    def plot_all_repeats(self, plot_type):
        # Add ability to omit some repeats, maybe even by filter e.g. max(rep['CorrelationArray'][:,1]) < x
        if plot_type in self.premade_plots:
            if plot_type == 'ACF':
                fig, ax = plt.subplots(nrows = 2, figsize = [8,6], gridspec_kw={'height_ratios': [3,1]}, dpi = 600)

                for repeat, repeat_data in self.data.items():
                    plot_kwargs = {
                        'label': repeat,
                        'alpha': 0.3
                    }
                    repeat_data.plot(axis = ax[0], plot_type = 'CorrelationArray', **plot_kwargs)
                    repeat_data.plot(axis = ax[1], plot_type = 'CountRateArray', **plot_kwargs)
                
                av_acf = average_time_series([rep.data['CorrelationArray'] for rep in self.data.values()])
                ax[0].plot(av_acf['time'], av_acf['mean'], label = 'Average', color = 'black')

                ax[0].set_xscale('log')
                ax[0].legend(loc = 'upper right')
                plt.tight_layout()
        # Add other plot types, e.g. 'ACF with fit'
        else:
            pass

class Experiment(object):
    def __init__(self, calibration):
        self.calibration = calibration
        self.data = dict()
        self._trace_count = 1
        if type(calibration.confocal_volume) == float:
            self.confocal_volume = calibration.confocal_volume
            self.confocal_widths = calibration.confocal_widths

    def add_run(self, trace, **kwargs):
        # Add check for trace class being appropriate
        if 'trace_name' in kwargs:
            self.data[kwargs.trace_name] = trace
        else:
            self.data['trace'+str(self._trace_count)] = trace
        self._trace_count += 1
    
    def calibrate_by(self, calibration_label):
        self.confocal_volume = calibration.calibrate_fcs(
            self.calibration.average.fit['Parameters']['Translation diffusion time species 1']['Result'],
            self.calibration.average.fit['Parameters']['Translation structural parameter']['Result'],
            calibration_label
        )
        self.confocal_widths = calibration.confocal_widths(
            self.calibration.average.fit['Parameters']['Translation diffusion time species 1']['Result'],
            calibration.given_d[calibration_label],
            self.calibration.average.fit['Parameters']['Translation structural parameter']['Result']
        )
        # Add calibration of traces
    
    def multiplot(self, rows: str, col: str, plot_type: str = 'ACF', **kwargs):
        pass

    def get_average_parameters(self, parameters):
        pass

