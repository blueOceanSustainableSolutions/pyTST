#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot
from pyTST import pyTST


tst = pyTST()


tst.load_data_file("example_data_filename", signal_column=1, time_column=0, tstep=0.05)

# fix to prevent recursive spawning under Windows
if __name__ == '__main__':
  tst.compute_TST(step_size=4)
tst.plot()

# Plot input signal data
pyplot.figure()
idx = 332 # result of tst analysis

pyplot.axvline(tst.time_array[idx], color='k', lw=0.8, ls='--', alpha=0.6)
pyplot.plot(tst.time_array[0:idx], tst.signal_array[0:idx], color='C1', alpha=0.8)
pyplot.plot(tst.time_array[idx:], tst.signal_array[idx:], color='C0')

pyplot.xlabel("t")
pyplot.ylabel("signal")
pyplot.show()

