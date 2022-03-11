"""RAW functions
Classes:
    - RawConfoCor3
Values:
    - zen_standard_acf
Functions:
    - bin_times
    - acf
TODO:
    - Add standard deviation estimation to acf function for fitting
    - Find out why Zen output does not exactly match the computed ACF
"""
import numpy as np
from math import ceil
import struct
from numba import njit

# The x-axis values from Zen's default CountRateArray
zen_standard_acf = np.array([2.0000000e-07, 4.0000000e-07, 6.0000000e-07, 8.0000000e-07,
       1.0000000e-06, 1.2000000e-06, 1.4000000e-06, 1.6000000e-06,
       1.8000000e-06, 2.0000000e-06, 2.2000000e-06, 2.4000000e-06,
       2.6000000e-06, 2.8000000e-06, 3.0000000e-06, 3.2000000e-06,
       3.6000000e-06, 4.0000000e-06, 4.4000000e-06, 4.8000000e-06,
       5.2000000e-06, 5.6000000e-06, 6.0000000e-06, 6.4000000e-06,
       7.2000000e-06, 8.0000000e-06, 8.8000000e-06, 9.6000000e-06,
       1.0400000e-05, 1.1200000e-05, 1.2000000e-05, 1.2800000e-05,
       1.4400000e-05, 1.6000000e-05, 1.7600000e-05, 1.9200000e-05,
       2.0800000e-05, 2.2400000e-05, 2.4000000e-05, 2.5600000e-05,
       2.8800000e-05, 3.2000000e-05, 3.5200000e-05, 3.8400000e-05,
       4.1600000e-05, 4.4800000e-05, 4.8000000e-05, 5.1200000e-05,
       5.7600000e-05, 6.4000000e-05, 7.0400000e-05, 7.6800000e-05,
       8.3200000e-05, 8.9600000e-05, 9.6000000e-05, 1.0240000e-04,
       1.1520000e-04, 1.2800000e-04, 1.4080000e-04, 1.5360000e-04,
       1.6640000e-04, 1.7920000e-04, 1.9200000e-04, 2.0480000e-04,
       2.3040000e-04, 2.5600000e-04, 2.8160000e-04, 3.0720000e-04,
       3.3280000e-04, 3.5840000e-04, 3.8400000e-04, 4.0960000e-04,
       4.6080000e-04, 5.1200000e-04, 5.6320000e-04, 6.1440000e-04,
       6.6560000e-04, 7.1680000e-04, 7.6800000e-04, 8.1920000e-04,
       9.2160000e-04, 1.0240000e-03, 1.1264000e-03, 1.2288000e-03,
       1.3312000e-03, 1.4336000e-03, 1.5360000e-03, 1.6384000e-03,
       1.8432000e-03, 2.0480000e-03, 2.2528000e-03, 2.4576000e-03,
       2.6624000e-03, 2.8672000e-03, 3.0720000e-03, 3.2768000e-03,
       3.6864000e-03, 4.0960000e-03, 4.5056000e-03, 4.9152000e-03,
       5.3248000e-03, 5.7344000e-03, 6.1440000e-03, 6.5536000e-03,
       7.3728000e-03, 8.1920000e-03, 9.0112000e-03, 9.8304000e-03,
       1.0649600e-02, 1.1468800e-02, 1.2288000e-02, 1.3107200e-02,
       1.4745600e-02, 1.6384000e-02, 1.8022400e-02, 1.9660800e-02,
       2.1299200e-02, 2.2937600e-02, 2.4576000e-02, 2.6214400e-02,
       2.9491200e-02, 3.2768000e-02, 3.6044800e-02, 3.9321600e-02,
       4.2598400e-02, 4.5875200e-02, 4.9152000e-02, 5.2428800e-02,
       5.8982400e-02, 6.5536000e-02, 7.2089600e-02, 7.8643200e-02,
       8.5196800e-02, 9.1750400e-02, 9.8304000e-02, 1.0485760e-01,
       1.1796480e-01, 1.3107200e-01, 1.4417920e-01, 1.5728640e-01,
       1.7039360e-01, 1.8350080e-01, 1.9660800e-01, 2.0971520e-01,
       2.3592960e-01, 2.6214400e-01, 2.8835840e-01, 3.1457280e-01,
       3.4078720e-01, 3.6700160e-01, 3.9321600e-01, 4.1943040e-01,
       4.7185920e-01, 5.2428800e-01, 5.7671680e-01, 6.2914560e-01,
       6.8157440e-01, 7.3400320e-01, 7.8643200e-01, 8.3886080e-01,
       9.4371840e-01, 1.0485760e+00, 1.1534336e+00, 1.2582912e+00,
       1.3631488e+00, 1.4680064e+00, 1.5728640e+00, 1.6777216e+00,
       1.8874368e+00, 2.0971520e+00, 2.3068672e+00, 2.5165824e+00,
       2.7262976e+00, 2.9360128e+00, 3.1457280e+00, 3.3554432e+00])


class RawConfoCor3(object):
    """
        A class for importing raw fcs files from ConfoCor 3

        ...

        Attributes
        ----------
        identifier : str
            The file header from the raw file
        measurement_id : list?
            Four integers from the header
        measurement_pos : int
            The measurement position recorded by the instrument
        kinetic_index : int
            The kinetic index recorded by the instrument
        repetition_number : int
            The repetition number recorded by the instrument
        sampling_frequency : int
            The sampling frequency of the instrument (in Hz)
        pulse_distances : list
            The raw output of the file. Each value is the clock time recorded between each pulse
        detector_times : numpy array
            The times from the start of recording at which each pulse occurs in the detector's clock times
        absolute_times : numpy array
            The detector times converted to seconds
        
        Optional Attributes
        -------------------
        CountRateArray : numpy array
            Created by the bin method. An array of times (in seconds) and the count rate array for that time
        acf : numpy array
            Created by the make_acf method. An array of time delays (in seconds) and the average autocorrelation of count rate for that delay
        PhotonCountHistogram: numpy array
            Created by the make_pch method. An array of count rate bins and the density of count rates that come under that bin
        
        Methods
        -------
        bin(bin_size):
            Add a CountRateArray attribute. This splits your absolute times into bins of width bin_size.
        make_acf(bin_size = 2*10**-7, autocorr_times = zen_standard_acf):
            Add an acf attribute. Calculates an autocorrelation function at the time delays given in autocorr_times after binning the data with the bin_size provided
        make_pch(bin_size = 2*10**5, pch_bins = np.arange(0,160000, 50000)):
            
    """

    def __init__(self, path: str) -> None:
        """
        Parameters
        ----------
        path: str
            The path leading to the raw file to read in
        """

        # The file is binary
        with open(path, 'rb') as f:
            bytes = f.read()
        # The first 64 bytes make a string of ASCII characters showing the file header
        self.identifier = bytes[:64].decode('ASCII')
        # The next 32 bytes are 4-byte integers. This can also be seen in the default file name
        self.measurement_id = struct.unpack_from('<4i', bytes[64:80])
        # The next 16 bytes are 4-byte integers encoding the measurement position, kinetic index, repetition number, and sampling frequency
        self.measurement_pos, self.kinetic_index, self.repetition_number, self.sampling_frequency = struct.unpack_from('<4I', bytes[80:96])
        # The remaining bytes (after a 32 byte gap) are integers showing the clock times of recorded pulses
        self.pulse_distances = [x[0] for x in struct.iter_unpack('<I', bytes[128:])]
        # To convert from pulse_distances to detector times, take the cumulative sum.
        self.detector_times = np.cumsum(self.pulse_distances)
        # To convert from detector times to real times, divide by the sampling frequency (e.g. clock time 15000 at 150000 Hz is 1 second)
        self.absolute_times = self.detector_times/self.sampling_frequency
    
    def bin(self, bin_size: int) -> None:
        """Splits the file's detected photons into count rate

        Uses the function bin_times.
        Adds a CountRateArray attribute to the object.

        Parameters
        ----------
        bin_size: int
            The size of bins in which to put the data, in seconds
        """

        self.CountRateArray = bin_times(self.absolute_times, bin_size)
    
    def make_acf(self, bin_size: int = 2*10**-7, autocorr_times = zen_standard_acf) -> None:
        """Computes an autocorrelation function from the file's pulse times
        
        Uses the function acf.
        Adds an acf attribute to the object

        Parameters
        ----------
        bin_size: int
            The bin size to pass to bin_times. Note, this bins data separately from the CountRateArray attribute
        autocorr_times: numpy array
            The time delays (tau) at which to calculate the autocorrelation function. See documentation for the acf function for details
        """

        binned = bin_times(self.absolute_times, bin_size)
        intervals = np.array(autocorr_times/bin_size, dtype = int)
        self.acf = np.array([autocorr_times, acf(binned, intervals)])
    
    def make_pch(self, bin_size:int = 2*10**-5, pch_bins: 'np.array' = np.arange(0, 160000, 50000)) -> None:
        """Creates a photon counting histogram from the file's pulse times

        Adds a PhotonCountHistogram attribute to the object

        Parameters
        ----------
        bin_size: int
            The binning for the pulse times. Note, this is not the bins for the PCH
        pch_bins: numpy array
            The bins for the PCH
        """

        binned = bin_times(self.absolute_times, bin_size)
        self.PhotonCountHistogram = np.histogram(binned, bins = pch_bins)

def bin_times(time_array: 'np.array', bin_size: int) -> 'np.array':
    """Bins an array of times into the bin sizes provided

    Parameters
    ----------
    time_array: numpy array
        The times of recorded responses from the detector
    bin_size: int
        The size of bins into which time_array should be split

    Returns
    -------
    A 2-Dimensional numpy array with the binned times and the count rates within those bins
    """
    # Compute the bin boundaries for the time array
    bins = np.arange(bin_size, time_array[-1], bin_size)
    # Count the responses within each bin, then divide it by the bin size to convert this into a count rate
    binned = np.bincount(np.digitize(time_array, bins))/bin_size
    # Return a 2-Dimensional array of the bin times and the binned data. The last bin is trimmed off as it may not be full length (this is what Zen seems to do, so I copied it)
    return np.array([bins, binned[:-1]])

# Calculating the ACF is **very** slow without JIT compiling and parallel processing
@njit(parallel = True)
def acf(count_rate_array, autocorr_interval):
    """Computes an autocorrelation function for the provided count rate array
    
    Parameters
    ----------
    count_rate_array: numpy array
        An array of count rates with the times in the first row and count rates in the second
    autocorr_interval: numpy array
        An array of times at which to calculate the autocorrelation function.
        The times must be a multiple of the count rate array bin sizes.
        The times provided are the default times for a 10 second trace saved in Zen. The calculation is pretty slow, so I would recommend keeping it around this size.
    Returns
    -------
    A numpy array of the autocorrelation times and the mean autocorrelation at those times
    """

    intensity = count_rate_array[1,:]
    mean_sq = np.mean(intensity)**2
    # Compute the ACF based on the FCS ACF definition G(tau) = (<I(t)*I(t+tau)>/<I**2>)
    ac_mean = np.array([np.mean(intensity[:-interval]*intensity[interval:]) for interval in autocorr_interval])/mean_sq

    return ac_mean

