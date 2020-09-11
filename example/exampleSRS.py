#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot
from pyTST import pyTST


# fix to prevent recursive spawning under Windows
if __name__ == '__main__':
    tst = pyTST()
    tst.load_data_file("SRSexample_data_file", signal_column=0, time_column=None, tstep=0.05)

    tst.compute_TST(step_size=8)
    tst.plot(filename='TST_exampleSRS.eps')

    # Plot input signal data
    pyplot.figure()
    idx = 2000 # result of tst analysis

    pyplot.axvline(tst.time_array[idx], color='k', lw=0.8, ls='--', alpha=0.6)
    pyplot.plot(tst.time_array[0:idx], tst.signal_array[0:idx], color='C1', alpha=0.8)
    pyplot.plot(tst.time_array[idx:], tst.signal_array[idx:], color='C0')

    pyplot.xlabel("$t / $s",fontsize=16)
    pyplot.ylabel("$\phi$",fontsize=16)


    pyplot.savefig('signalExampleSRS.eps',dpi=1000)
    pyplot.show()
