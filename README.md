# metrology-data-analysis

This is a loose collection of mostly Jupyter notebooks but also some Python scripts for analysing metrology data acquired by the scripts in the [dmm-logging repo](https://github.com/alson/dmm-logging).

## Shared code

`common_plotting.py` module with commonly used functions for importing, processing and plotting data.

`common_analysis.py` module with commonly used functions for analysing data.

## Experiments

### Measurement tests

Testing various theories, mostly relating to the HPAK 3458A

Scripts:
* `measurement-tests.ipynb`

### SR104 stability

Quick-and-dirty test that of an SR104 against a HP 3458A (aka testing the stability of the 3458A 😂).

Scripts:
`plot-sr104.py`: Plot results

### Datron 4910 stability

Quick-and-dirty test (spotting a pattern yet) of Datron 4910 against Fluke 732A and EDC MV106. Various DMMs are used as nullmeters to compare the 10V output of the several references. It measures the four individual 10V reference outputs of the 4910 against the average output, and the Fluke 732A and EDC MV106 against the 4910 average output.

Scripts:
* `plot-F732A-mv106-d4910.py`: Plot the results (in [metrology-data-analysis](https://github.com/alson/metrology-data-analysis)).

### Voltage reference comparisons

Comparing the stability of various voltage references against each other and against a 3458A.

Scripts:
* `vrefs.ipynb`: Comparison of all historical measurements trying to establish drift.
* `old_vref_comparison.ipynb`: Older analysis

### Resistance transfers

Comparing the stability of various resistance standards and calibrators against each other and against a 3458A.

Scripts:
* `resistance-decade.ipynb`: Testing a resistance decade
* `resistance.ipynb`: Measuring the resistance and voltage tempco of my 3458As.
* `Resistance transfers.ipynb`:
* `sr104-results.ipynb`: Comparing SR104 measurements against 3458A

### Environmental monitoring

Monitoring environmental conditions (temperature, humidity and/or pressure).

Scripts:
`view_pt100.ipynb`: Plot PT100 measurements
`view_recent_thp.ipynb`: Plot recent temperature, humidity and pressure measurements

### Meter comparisons

Comparing the stability of various meters against each other.

Scripts:
* `w4950-w4920-ks3458a-comparison-sweeps.ipynb`: Comparing HPAK 3458A, Wavetek 4950 and Wavetek 4920

### xDevs nanovolt data

Just playing around with some xDevs nanovolt data to visualize change in behaviour over time.

Scripts:
* `time_series_histogram.ipynb`

### Events

#### Metrology Meet 2022 (organized by branadic)

All my results from MM 2022

Scripts:
* `MM2022.ipynb`: Comparing results from MM 2022
* `MM2022.html`: HTML export of this notebook
* `MM2022-*.png`: Plots from this notebook
