# pyTST

This module performs the "Transient Scanning Technique" presented in:

* Brouwer, J., Tukker, J., & van Rijsbergen, M. (2013). Uncertainty Analysis of Finite Length Measurement Signals. 3rd International Conference on Advanced Model Measurement Technology for the EU Maritime Industry.
       
* Brouwer, J., Tukker, J., & van Rijsbergen, M. (2015). Uncertainty Analysis and Stationarity Test of Finite Length Time Series Signals. 4th International Conference on Advanced Model Measurement Technology for the Maritime Industry.

It allows to easily detect transient portion of signal data.

## Installation
Can be installed and use like any python package, for example:
```
pip3 install --user https://github.com/WavEC-Offshore-Renewables/pyTST/archive/master.zip
```



## Usage
This package provides both a command line tool as well as a python library (for more flexibility).  

### Command line usage
If the signal data looks like:
```
# time   signal
  1     0.280910708014E-03 
  2     0.280910708014E-03
  3     0.345576259768E-03
...
```

the following can be used

```
TST-cli --time-col=0 --signal-col=1 example_data_filename
```
   
See `TST-cli -h` for more details on the possibilities available.

### Library examples
Signal data can be loaded from a file
```
from pyTST import pyTST

tst = pyTST()

tst.load_data_file("example_data_filename", signal_column=1, time_column=0, tstep=0.05)

tst.compute_analysis(step_size=10)
tst.export_to_txt("TST_analysis.dat")
# tst.import_from_txt("TST_analysis.dat")
tst.plot()
```

Or data can be directly provided:
```
import numpy as np
from pyTST import pyTST

# Signal creation
t = np.linspace(1,1000, 5000)

signal = np.sin(t)

# Add initial transiant effect
signal[0:100] += np.linspace(1,0, 100)


tst = pyTST()
tst.load_data_array(signalay=t)

tst.compute_analysis(step_size=10)
tst.export_to_txt("TST_analysis.dat")
# tst.import_from_txt("TST_analysis.dat")
tst.plot()
```
