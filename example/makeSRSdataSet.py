#!/usr/bin/env python
# -*- coding: utf-8 -*-

# A slightly more realistic input files

import numpy as np
from matplotlib import pyplot
from pyTST import pyTST
import random


t = np.linspace(0.,24999.,25000)

# Sine 1
a = 4.
b = 0.5
period = 500.
c = 2. * np.pi/period
phi = a+b * np.sin(c*t)

# Sine 2
b2 = 0.1
period2 = 3000.
c2 = 2. * np.pi/period2

phi = phi + b2 * np.sin(c2*t)




# Add start-up
for i in range(0,3000):
  #phi[i] = phi[i] + a * np.sin(2.*np.pi/30*(i-3/2*np.pi))*(1.-(i-100)/300.)
  phi[i] = phi[i] + 3 * np.sin(2.*np.pi/300*(i-3/2*np.pi))*(0.5+0.5*np.cos(i*2.*np.pi/6000.))




# Random noise
for i in range(0,5000):
  phi[5*i] = phi[5*i] + random.randint(-1000, 1000) / 6000.




pyplot.plot(t,phi)
pyplot.show()

np.savetxt('SRSexample_data_file',phi,fmt='%.10E')
