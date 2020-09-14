#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot
from pyTST import pyTST

selectTimeDiscard = True
step_size = 500


# fix to prevent recursive spawning under Windows
if __name__ == '__main__':
    tst = pyTST()
    tst.load_data_file("SRSexample_data_file", signal_column=0, time_column=None, tstep=0.05)

    tst.compute_TST(step_size=step_size)
    tst.plot(filename='TST_exampleSRS',fileFormat='.eps',\
                    selectTimeDiscard=selectTimeDiscard,step_size=step_size)

# EOF
