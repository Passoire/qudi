# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:35:51 2015
This file contains a class for standalone analysis of fast counter data.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2015 Nikolas Tomek nikolas.tomek@uni-ulm.de
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class PulseAnalysis:

    def __init__(self, counter_binwidth):
        self.counter_binwidth = counter_binwidth
        # std. deviation of the gaussian filter.
        # Too small and the filtered data is too noisy to analyze; too big and the pulse edges are
        # filtered out...
        self.conv_std_dev = 10
        # set windows for signal and normalization of the laser pulses in seconds
        self.signal_start = 0
        self.signal_width = 200e-9
        self.norm_start = 500e-9
        self.norm_width = 200e-9
        # total number of laser pulses in the sequence
        self.number_of_lasers = 50

    def _gated_extraction(self, count_data):
        """ This method detects the rising flank in the gated timetrace data and extracts just the laser pulses
          @param 2D numpy.ndarray count_data: the raw timetrace data from a gated fast counter (dimensions 0: gate number, 1: time bin)
          @return  2D numpy.ndarray: The extracted laser pulses of the timetrace (dimensions 0: laser number, 1: time bin)
        """
        # sum up all gated timetraces to ease flank detection
        timetrace_sum = np.sum(count_data, 1)
        # apply gaussian filter to remove noise and compute the gradient of the timetrace sum
        conv_deriv = self._convolve_derive(timetrace_sum, self.conv_std_dev)
        # get indices of rising and falling flank
        rising_ind = conv_deriv.argmax()
        falling_ind = conv_deriv.argmin()
        # slice the data array to cut off anything but laser pulses
        laser_arr = count_data[rising_ind:falling_ind, :].transpose()
        return laser_arr

    def _ungated_extraction(self, count_data, num_of_lasers):
        """ This method detects the laser pulses in the ungated timetrace data and extracts them
          @param 1D numpy.ndarray count_data: the raw timetrace data from an ungated fast counter
          @param int num_of_lasers: The total number of laser pulses inside the pulse sequence
          @return 2D numpy.ndarray: The extracted laser pulses of the timetrace (dimensions 0: laser number, 1: time bin)
        """
        # apply gaussian filter to remove noise and compute the gradient of the timetrace
        conv_deriv = self._convolve_derive(count_data, self.conv_std_dev)
        # initialize arrays to contain indices for all rising and falling flanks, respectively
        rising_ind = np.empty([num_of_lasers], int)
        falling_ind = np.empty([num_of_lasers], int)
        # Find as many rising and falling flanks as there are laser pulses in the timetrace
        for i in range(num_of_lasers):
            # save the index of the absolute maximum of the derived timetrace as rising flank position
            rising_ind[i] = np.argmax(conv_deriv)
            # set this position and the sourrounding of the saved flank to 0 to avoid a second detection
            if rising_ind[i] < 2*self.conv_std_dev:
                del_ind_start = 0
            else:
                del_ind_start = rising_ind[i] - 2*self.conv_std_dev
            if (conv_deriv.size - rising_ind[i]) < 2*self.conv_std_dev:
                del_ind_stop = conv_deriv.size-1
            else:
                del_ind_stop = rising_ind[i] + 2*self.conv_std_dev
            conv_deriv[del_ind_start:del_ind_stop] = 0

            # save the index of the absolute minimum of the derived timetrace as falling flank position
            falling_ind[i] = np.argmin(conv_deriv)
            # set this position and the sourrounding of the saved flank to 0 to avoid a second detection
            if falling_ind[i] < 2*self.conv_std_dev:
                del_ind_start = 0
            else:
                del_ind_start = falling_ind[i] - 2*self.conv_std_dev
            if (conv_deriv.size - falling_ind[i]) < 2*self.conv_std_dev:
                del_ind_stop = conv_deriv.size-1
            else:
                del_ind_stop = falling_ind[i] + 2*self.conv_std_dev
            conv_deriv[del_ind_start:del_ind_stop] = 0
        # sort all indices of rising and falling flanks
        rising_ind.sort()
        falling_ind.sort()
        # find the maximum laser length to use as size for the laser array
        laser_length = np.max(falling_ind-rising_ind)
        # initialize the empty output array
        laser_arr = np.zeros([num_of_lasers, laser_length],int)
        # slice the detected laser pulses of the timetrace and save them in the output array
        for i in range(num_of_lasers):
            if (rising_ind[i]+laser_length > count_data.size):
                lenarr = count_data[rising_ind[i]:].size
                laser_arr[i, 0:lenarr] = count_data[rising_ind[i]:]
            else:
                laser_arr[i] = count_data[rising_ind[i]:rising_ind[i]+laser_length]
        return laser_arr

    def _convolve_derive(self, data, std_dev):
        """ This method smoothes the input data by applying a gaussian filter (convolution) with
            specified standard deviation. The derivative of the smoothed data is computed afterwards and returned.
            If the input data is some kind of rectangular signal containing high frequency noise,
            the output data will show sharp peaks corresponding to the rising and falling flanks of the input signal.
          @param 1D numpy.ndarray timetrace: the raw data to be smoothed and derived
          @param float std_dev: standard deviation of the gaussian filter to be applied for smoothing
          @return 1D numpy.ndarray: The smoothed and derived data
        """
        conv = ndimage.filters.gaussian_filter1d(data, std_dev)
        conv_deriv = np.gradient(conv)
        return conv_deriv

    def extract_laser_data(self, raw_data, number_of_lasers=None):
        """

        @param 1D/2D numpy.ndarray raw_data:
        @return 2D numpy.ndarray: The extracted laser data
        """
        if number_of_lasers is None:
            number_of_lasers = self.number_of_lasers

        if raw_data.ndim == 1:
            return self._ungated_extraction(raw_data, number_of_lasers)
        elif raw_data.ndim == 2:
            return self._gated_extraction(raw_data)
        raise TypeError('Timetrace raw data must be numpy.ndarray with either 1 or 2 dimensions.')

    def analyze_laser_data(self, laser_data, signal_start=None, signal_width=None, norm_start=None,
                           norm_width=None):
        """

        @param laser_data:
        @return:
        """
        # laser_data = laser_data.transpose()

        if signal_start is None:
            signal_start = self.signal_start
        if signal_width is None:
            signal_width = self.signal_width
        if norm_start is None:
            norm_start = self.norm_start
        if norm_width is None:
            norm_width = self.norm_width

        # set start and stop indices for the analysis
        norm_start_bin = int(np.rint(norm_start / self.counter_binwidth))
        norm_end_bin = int(np.rint((norm_start + norm_width) / self.counter_binwidth))
        signal_start_bin = int(np.rint(signal_start / self.counter_binwidth))
        signal_end_bin = int(np.rint((signal_start + signal_width) / self.counter_binwidth))
        # loop over all laser pulses and analyze them. Save analyzed data points in array.
        y_data = np.zeros(len(laser_data), dtype=float)
        for i, trace in enumerate(laser_data):
            # calculate the mean of the data in the normalization window
            norm_mean = trace[norm_start_bin:norm_end_bin].mean()
            # calculate the mean of the data in the signal window
            signal_mean = (trace[signal_start_bin:signal_end_bin] - norm_mean).mean()
            # update the signal plot y-data
            y_data[i] = 1. + (signal_mean / norm_mean)
        return y_data

    def analyze_raw_data(self, raw_data):
        """
        This method captures the fast counter data and extracts the laser pulses.

        @param int num_of_lasers: The total number of laser pulses inside the pulse sequence
        @return 2D numpy.ndarray: The extracted laser pulses of the timetrace
                                  (dimensions 0: laser number, 1: time bin)
        @return 1D/2D numpy.ndarray: The raw timetrace from the fast counter
        """
        # Extract laser pulses
        laser_data = self.extract_laser_data(raw_data)

        # Analyze laser data
        data = self.analyze_laser_data(laser_data)
        return data


if __name__ == "__main__":
    tool = PulseAnalysis(False, 1e-9)
    data = np.loadtxt('FastComTec_demo_timetrace.asc')
    analyzed_data = tool.analyze_raw_data(data)
    plt.plot(analyzed_data)
    plt.show()
