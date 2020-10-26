#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot
from pyTST import pyTST


# fix to prevent recursive spawning under Windows
if __name__ == '__main__':
    tst = pyTST()
    tst.load_data_file("example_data_filename", signal_column=1, time_column=0, tstep=0.05)

    tst.compute_TST(step_size=4)
    tst.plot()

