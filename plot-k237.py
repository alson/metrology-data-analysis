#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import sys


curve = pd.read_csv(sys.argv[1], parse_dates=['datetime'])
curve.plot.scatter('set_voltage', 'measure_current')
plt.show()

