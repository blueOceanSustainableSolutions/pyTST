# pyTST

This module performs the "Transient Scanning Technique" presented in:

        Brouwer, J., Tukker, J., & van Rijsbergen, M. (2013). Uncertainty Analysis of Finite Length Measurement Signals. The 3rd International Conference on Advanced Model Measurement Technology for the EU Maritime Industry, February.
and
        Brouwer, J., Tukker, J., & van Rijsbergen, M. (2015). Uncertainty Analysis and Stationarity Test of Finite Length Time Series Signals. 4th International Conference on Advanced Model Measurement Technology for the Maritime Industry.


## Installation
Can be installed and use like any python package, for example:



## Usage
This package provides both a command line tool as well as a python library (for more flexibility). 
Usage examples for the library are provided in `example` folder. 

### Command line examples
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
When loading data from a file
```
from pyTST import pyTST

tst = pyTST()

tst.load_data_file("example_data_filename", signal_column=1, time_column=0, tstep=0.05)

tst.compute_analysis(step_size=10)
tst.export_to_txt("TST_analysis.dat")
# tst.import_from_txt("TST_analysis.dat")
tst.plot()
```
