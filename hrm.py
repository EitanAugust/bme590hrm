import numpy as np
import pandas as pd
import math
import logging
import json
from scipy import signal
import sys

logging.basicConfig(filename="hrm.log",
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


class hrm:

    def __init__(self, csvdata):
        self.csvdata = csvdata
        self.mean_hr_bpm = None
        self.voltage_extremes = None
        self.duration = None
        self.num_beats = None
        self.beats = None
        self.run_hrm()

    def run_hrm(self):
        '''Function to coordinate steps in finding meaningful ECG parameters'''

        data = self.load_csv()
        self.voltage_extremes = self.calc_voltage_extremes(data)
        self.duration = self.calc_duration(data)
        data_norm = self.normalize_data(data)
        [threshold, data_cutoff] = self.cut_data(data_norm)
        [self.mean_hr_bpm, self.num_beats, self.beats, voltages] \
            = self.calc_beats(data_norm, data, threshold, data_cutoff)
        self.write_json()

    def load_csv(self):
        '''Function to load csv file and convert it to a numpy array
         and checks data for discrepancies

            :return: A two column Numpy array with time data in first
             column and voltage data in second
            :rtype: array
            :raises ImportError: if a pandas is not installed
            '''
        try:
            datapd = pd.read_csv(self.csvdata)
            data = datapd.as_matrix()
            logging.info('file found and array created')
            first = True
            for i in range(0, data.shape[0]):
                if isinstance(data[i, 0], str) or isinstance(data[i, 1], str):
                    try:
                        data[i, 0] = float(data[i, 0])
                        data[i, 1] = float(data[i, 1])
                    except ValueError:
                        data[i, 0] = data[i-1, 0]
                        data[i, 1] = data[i-1, 1]
                        print('non digit string in data')
                        logging.warning('string or empty space in data '
                                        'replaced with previous row''s values')
                if data[i, 0] is None or data[i, 1] == 1:
                    data[i, 0] = data[i-1, 0]
                    data[i, 1] = data[i-1, 1]
                if data[i, 1] > 300 and first is True:
                    logging.warning('ECG voltage greater than 300 mV')
                    print('ECG voltage exceeds 300 mV')
                    first = False
            data = np.array(data, dtype=np.float32)
            return data
        except ImportError:
            print('pandas installation required or csv file does not exist')
            logging.debug('Required package was not installed')

    def calc_voltage_extremes(self, data):
        '''Function to find the maximum and minimum voltages in ECG signal

            :param array data: contains two columns of numerical lists
            :return: a tuple of the minimum and maximum voltages in signal
            :rtype: tuple
            :raises ValueError: if either column of data is not a
             numerical list
            :raises TypeError: if csv file is not correctly formatted
            :raises ImportError: if numpy is not installed
            '''
        try:
            minV = np.amin(data, axis=0)[1]
            maxV = np.amax(data, axis=0)[1]
            voltage_extremes = (minV, maxV)
            return voltage_extremes
        except ImportError:
            print('numpy installation required')
            logging.debug('Required package was not installed')
        except TypeError:
            print('csv file is not correct format')
        except ValueError:
            print('csv data must have numbers in both columns')

    def calc_duration(self, data):
        '''Function to find the duration of ECG signal

            :param array data: contains two columns of numerical lists
            :return: the duration in seconds of the ECG signal
            :rtype: float
            :raises ImportError: if a required package was not loaded
            '''
        try:
            duration = np.amax(data, axis=0)[0]
        except ImportError:
            print('numpy installation required')
        return duration

    def normalize_data(self, data):
        '''Function to autocorrelate data

            :param array data: contains two columns of numerical lists
            :return: autocorrelated data, a voltage threshold to signal
             a beat, and the autocorrelated data with less
             significant voltage sign data as 0's.
            :rtype: array, float and array respectively
            :raises IndexError: if file has less than 10 relative minimums
             or maximums
            :raises ImportError: if a required package was not loaded
            '''
        data_avg = []
        sample_rate = data[5, 0] - data[4, 0]

        try:
            window_size = math.floor(0.5 / sample_rate)
        except ImportError:
            print('must have math package imported')
            logging.debug('math package was not imported')
        if window_size % 2 != 0:
            window_size = window_size - 1
        for i in range(0, data.shape[0]):
            if i < window_size/2:
                data_avg.append(np.average(data[i:i+window_size, 1]))
            elif i > data.shape[0] - window_size/2:
                data_avg.append(np.average(data[i-window_size:i, 1]))
            else:
                data_avg.append(np.average(data[int(i-window_size/2):
                                                int(i+window_size/2), 1]))

        data_normal = data[:, 1] - data_avg
        logging.info('data was successfully autocorrelated')
        return data_normal

    def cut_data(self, data_normal):
        '''Function to replace less significant signed voltage
         data (positive or negative) with 0's.

                    :param array data_normal: single dimension numerical list
                    :return: a voltage threshold to signal a beat,
                    and the autocorrelated data with less
                     significant voltage sign data as 0's.
                    :rtype: float and array respectively
                    :raises IndexError: if file has less than 10 relative
                     minimums or maximums
                    :raises ImportError: if a required package was not loaded
                    '''
        try:

            maximums_loc = signal.argrelextrema(data_normal, np.greater)
            minimums_loc = signal.argrelextrema(data_normal, np.less)

            maximums = data_normal[maximums_loc[0]]
            minimums = np.absolute(data_normal[minimums_loc[0]])

            maximums.sort()
            minimums.sort()

            min_norm = np.absolute(minimums[-10])
            max_norm = np.absolute(maximums[-10])

        except ImportError:
            print('must have scipy.signal package installed')
            logging.debug('Required package was not installed')
        except IndexError:
            print('there are less than 10 relative maximum or'
                  ' minimum in CSV file')
            logging.warning('file contains a small amount of data')

        threshold = None

        data_cutoff = data_normal.copy()

        if max_norm > min_norm:
            threshold = max_norm
            for i in range(0, data_normal.shape[0]):
                if data_normal[i] < 0:
                    data_cutoff[i] = 0
        else:
            threshold = min_norm
            for i in range(0, data_normal.shape[0]):
                if data_normal[i] > 0:
                    data_cutoff[i] = 0

        data_cutoff = np.absolute(data_cutoff)
        logging.info('voltage data was successfully cut and threshold found')
        return threshold, data_cutoff

    def calc_beats(self, data_norm, data, threshold, data_cutoff):
        '''Function to calculate beat identifying related parameters
         like bpm, number of beats, and time of beats

                    :param array data_norm: list of correlated voltage data
                    :param array data: two columns of numerical lists
                    :param float threshold: 65% if this value signals a beat
                    :param array data_cutoff: list of correlated voltage
                    data with less significant sign voltages replaced with 0
                    :return: average bpm, the number of total beats,
                     time of beats, and voltages in correlated
                     data at these times.
                    :rtype: float, int, array and array respectively
                    '''
        threshold = 0.65*threshold
        beats = []
        voltages = []
        maximums = signal.argrelextrema(data_cutoff, np.greater)
        avg = np.average(data_cutoff[maximums[0]])
        for i in range(0, maximums[0].shape[0]):
            if data_cutoff[maximums[0][i]] > threshold and not beats:
                beats.append(data[maximums[0][i], 0])
                voltages.append(data_norm[maximums[0][i]])
            elif data_cutoff[maximums[0][i]] > threshold and \
                    data[maximums[0][i], 0]-beats[len(beats)-1] > 0.3:
                beats.append(data[maximums[0][i], 0])
                voltages.append(data_norm[maximums[0][i]])

        mean_hr_bpm = len(beats)/beats[len(beats)-1] * 60.0
        num_beats = len(beats)
        np.array(beats)
        logging.info('all beat identifying calculations have been made')
        return mean_hr_bpm, num_beats, beats, voltages

    def write_json(self):
        '''Function to save 5 attributes for ECG in json file

                    :raises ImportError: if JSON has not been imported
                            '''
        d = {'mean_hr_bpm': self.mean_hr_bpm, 'voltage_extremes':
             [self.voltage_extremes], 'duration':
             [self.duration], 'num_beats': [self.num_beats],
             'beats': [self.beats]}
        df = pd.DataFrame(data=d)
        df.to_json(self.csvdata[:-3]+'json')
        print(self.csvdata[:-3]+'json created')
