'''RAW functions
Classes:
    - RawConfoCor3
Values:
    - zen_standard_acf
Functions:
    - bin_times
    - acf
TODO:
    - Add standard deviation estimation to acf function for fitting

'''
import numpy as np
from math import ceil
import struct
from numba import njit

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
    def __init__(self, path) -> None:
        with open(path, 'rb') as f:
            bytes = f.read()

        self.identifier = bytes[:64].decode('ASCII')
        self.measurement_id = struct.unpack_from('<4i', bytes[64:80])
        self.measurement_pos, self.kinetic_index, self.repetition_number, self.sampling_frequency = struct.unpack_from('<4I', bytes[80:96])

        self.pulse_distances = [x[0] for x in struct.iter_unpack('<I', bytes[128:])]
        self.detector_times = np.cumsum(self.pulse_distances)
        self.absolute_times = self.detector_times/self.sampling_frequency
    
    def bin(self, bin_size):
        self.CountRateArray = bin_times(self.absolute_times, bin_size)
    
    def make_acf(self, bin_size = 2*10**-7, autocorr_times = zen_standard_acf):
        binned = bin_times(self.absolute_times, bin_size)
        intervals = np.array(autocorr_times/bin_size, dtype = int)
        self.acf = np.array([autocorr_times, acf(binned, intervals)])

def bin_times(time_array, bin_size):
    bins = np.arange(bin_size, time_array[-1], bin_size)
    binned = np.bincount(np.digitize(time_array, bins))/bin_size
    return np.array([bins, binned[:-1]])

@njit(parallel = True)
def acf(count_rate_array, autocorr_interval):
    intensity = count_rate_array[1,:]
    mean = np.mean(intensity)

    ac_mean = np.array([np.mean(intensity[:-interval]*intensity[interval:]) for interval in autocorr_interval])/mean**2

    return ac_mean

