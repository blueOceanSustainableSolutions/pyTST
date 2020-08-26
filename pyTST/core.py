#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from matplotlib import pyplot


class pyTST:
    def load_data_array(self, signal_array, time_array=None, tstep=1):
        """
        Load time signal data from python array

        Parameters
        ----------
        signal_array : array of float
            signal to use for the TST

        time_array : array of float
            time stamps assiciated with signal_array

        tstep : float
            time step used when time_array is not provided
        """

        self.signal_array = signal_array
        if time_array is None:
            self.time_array = (np.array(range(len(self.signal_array)))+1)*tstep
        else:
            self.time_array = time_array


    def load_data_file(self, filename, signal_column, time_column=None, tstep=1, **kwargs):
        """
        Load time signal data text file

        Parameters
        ----------
        filename : string
            filename where the data is located

        signal_column : int
            index of the column where the signal is located

        time_column : int
            index of the time column in the file

        tstep : float
            multiplier for the time_column, useful to convert a counter column to real time steps,
            or timestep used when time_column is not provided

        kwargs
            any other parameter is passed directly to numpy.loadtxt
        """

        if time_column is None:
            usecols = (signal_column, )
        else:
            usecols = (signal_column, time_column)

        timedata = np.loadtxt(filename, usecols=usecols, **kwargs)
        self.signal_array = timedata[:, 0]

        if time_column is None:
            self.time_array = (np.array(range(len(self.signal_array)))+1)*tstep
        else:
            self.time_array = timedata[:, 1]*tstep


    def compute_TST(self, step_size=10, analyse_end=False, nproc=None):
        """
        Actual computation of the Transient Scanning Technique 

        Parameters
        ----------
        step_size : int
            size of the steps for the TST, data length/step_size computations will be performed

        analyse_end : Boolean
            analyse the end of the signal instead of the begining, (TST-B instead of TST-A)

        nproc : int
            number of process to use for the parallel computation, 
            if not provided the maximum available will be used

        """

        if analyse_end:
            data = self.signal_array[::-1]
            time = self.time_array[::-1]
        else:
            data = self.signal_array
            time = self.time_array

        if nproc is None:
            nproc = mp.cpu_count()

        if step_size is None:
            step_size = 10

        nb_slices = int(np.ceil(len(data)/step_size))
        print("{} slices to compute".format(nb_slices))

        pool = mp.Pool(processes=nproc)
        result = pool.map(variance_stats,
                          [ data[step_size*istart:] for istart in range(nb_slices) ],
                          chunksize=max(2, int(nb_slices/nproc/10)))
        pool.close()
        pool.join()

        self.mean_array = [ row[0] for row in result ]
        self.u95_array = [ row[1] for row in result ]
        self.step_time_array = np.array([ time[step_size*istart] for istart in range(nb_slices) ])

        if analyse_end:
            self.step_time_array = self.step_time_array[::-1]

    def export_to_txt(self, filename):
        """
        Export computed data to text file, can be loaded with import_from_txt

        Parameters
        ----------
        filename : string
            filename of the file to save
            
        """
 
        export_array = np.column_stack((self.step_time_array, self.u95_array, self.mean_array),)
        np.savetxt(filename, export_array,
                   header="t, u95, mean")

    def import_from_txt(self, filename):
        """
        Import data that was exported with export_to_txt

        Parameters
        ----------
        filename : string
            filename of the file to import
            
        """
 
        timedata = np.loadtxt(filename)
        self.step_time_array = timedata[:, 0]
        self.u95_array = timedata[:, 1]
        self.mean_array = timedata[:, 2]

    def plot(self, filename=None, show_cursor=True):
        """
        Plot the TST results previously computed

        Parameters
        ----------

        filename : string
            if provided, the plot will be exported to filename

        show_cursor : Boolean
            True if a cursor is ploted. Double clicking on the plot will move it

        """

        fig, ax = pyplot.subplots()
        pyplot.loglog(self.step_time_array, self.u95_array[::-1])


        # Display the grid (t, 1/t)
        grid_t = np.array([self.step_time_array[0]/2, self.step_time_array[-1]*2])
        for i in range(-20,20):
            factor = 10**(i/2)
            pyplot.loglog(grid_t,
                          factor/grid_t,
                          color='grey', alpha=0.5, linewidth=0.5)

        if show_cursor:
            def update_cursor(index):
                min_u95 = self.u95_array[-index-1]
                discard_time = self.step_time_array[-1] - self.step_time_array[index]

                hline.set_ydata(min_u95)
                vline.set_xdata(self.step_time_array[index])
                text.set_text('u95={:2e}\nt={}'.format(min_u95, discard_time))
                print("t={}, mean={:e} ± {:e}".format(discard_time, self.mean_array[-index-1], min_u95))


            def onclick(event):
                # only act on double click
                if not event.dblclick:
                    return

                index = min(np.searchsorted(self.step_time_array, event.xdata), len(self.step_time_array) - 1)
                update_cursor(index)
                pyplot.draw()

            hline = ax.axhline(color='k', lw=0.8, ls='--', alpha=0.6)
            vline = ax.axvline(color='k', lw=0.8, ls='--', alpha=0.6)

            text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
            update_cursor(np.argmin(self.u95_array[::-1]))

            cid = fig.canvas.mpl_connect('button_press_event', onclick)


        pyplot.ylim(top=np.max(self.u95_array)*2,
                    bottom= np.min(self.u95_array)/2)
        pyplot.xlim(right=self.step_time_array[-1]*2,
                    left=self.step_time_array[0]/2)
        pyplot.xlabel("t")
        pyplot.ylabel("95% uncertainty (u95)")


        if filename is None:
            pyplot.show()
        else:
            print("Figure exported to {}".format(filename))
            pyplot.savefig(filename)





def variance_stats(data):
    """
        Calculate variance statistics based on
        Brouwer, J., Tukker, J., & van Rijsbergen, M. (2013). Uncertainty Analysis of Finite Length Measurement Signals. The 3rd International Conference on Advanced Model Measurement Technology for the EU Maritime Industry, February.
        Brouwer, J., Tukker, J., & van Rijsbergen, M. (2015). Uncertainty Analysis and Stationarity Test of Finite Length Time Series Signals. 4th International Conference on Advanced Model Measurement Technology for the Maritime Industry.
        Brouwer, J., Tukker, J., Klinkenberg, Y., & van Rijsbergen, M. (2019). Random uncertainty of statistical moments in testing: Mean. Ocean Engineering, 182(April), 563–576. https://doi.org/10.1016/j.oceaneng.2019.04.068


        Parameters
        ----------
        data : array of float
            time signal to get the variance uncertainty from

        Returns
        ----------
        mean: float
            mean of the signal

        u95: float
            95% confidence bound (1.96* expected standard deviation of the mean)
    """

    N = len(data)
    mean = np.mean(data)

    # Estimate autocovariance
    Sxx = (np.abs(np.fft.fft(data - mean, N*2))**2)/N # autospectral density function
    Cxx = np.fft.ifft(Sxx)                            # autocovariance function (Wiener-Khinchine)
    Cxx = np.real(Cxx[0:N])                           # crop till Nyquist point

    # Variance estimators
    iArr     = np.abs(range(1-N,N))                       # indexing for -T to T integral
    var_x_av = 0.5/N*np.sum((1.0 - iArr*1.0/N)*Cxx[iArr]) # variance estimate for mean value (including factor 0.5 for bias correction)

    # expanded uncertainty factor for normal distribution with 95% confidence
    u95 = 1.96*np.sqrt(var_x_av)

    return mean, u95

