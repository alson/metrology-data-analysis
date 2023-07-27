#!/usr/bin/python3

import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.legend_handler as lh
import matplotlib.lines as mlines
from itertools import chain

THP_FN = 'thp_log.csv'
VOLT_FNS = [
    '3458A-2-MV106.csv',
    '3458A-MV106.csv',
    '3458A-x2-MV106.csv',
    'k199-x2-3456A-3458A-F730A-6031A-3245A-log.csv',
    'k199-x2-3456A-3458A-F730A-log.csv',
    'k199-x2-3456A-3458A-x2-F730A-6031A-3245A-8200-log.csv',
    'k199-x2-3456A-3458A-x2-F730A-6031A-3245A-log.csv',
    'k199-x2-3456A-3458A-x2-F730A-6031A-8200-log.csv',
    'k199-x2-3458A-x2-F730A-6031A-8200-log.csv',
    'k199-x2-3458A-x2-F730A-MV106-PM2534-ch3-only-log.csv',
    'k199-x2-3458A-x2-F730A-MV106-PM2534-log.csv',
    'k199-x2-3458A-x2-F730A-MV106-ch3-only-log.csv',
    'k199-x2-3458A-x2-F730A-PM2534-log.csv',
    'k199-x2-3458A-x2-F732A-MV106-log.csv',
    'k2000-3456A-3458A-VI700-log.csv',
    ]
ISO8601_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
VOLT_COLUMNS = { 'last_acal_1_cal72', 'last_acal_2_cal72' }


days = mdates.DayLocator(interval=7)
days7 = mdates.DayLocator(interval=7)
months = mdates.MonthLocator()
daysFmt = mdates.DateFormatter('%Y-%m-%d')
monthsFmt = mdates.DateFormatter('%Y-%m')

def parse_iso8601_date(date_str):
    if '.' not in date_str:
        date_str += '.0'
    return datetime.datetime.strptime(date_str, ISO8601_FORMAT)

def fix_last_acal_cal72(row):
    if 'last_acal_cal72' in row:
        row['last_acal_1_cal72'] = row['last_acal_cal72']
        del row['last_acal_cal72']
    return row

def parse_datetime_volts(row):
    return { k: (parse_iso8601_date(v) if k ==
        'datetime' else float(v) if k in VOLT_COLUMNS else None) for k, v in
        fix_last_acal_cal72(row).items()}

def parse_datetime(row):
    return { k: (parse_iso8601_date(v) if k ==
        'datetime' else float(v)) for k, v in row.items()}

def read_data():
    volt_rows = []
    for fn in VOLT_FNS:
        with open(fn, 'r', newline='') as volt_file:
            voltr = csv.DictReader(volt_file)
            volt_rows.extend(voltr)
    with open(THP_FN, 'r', newline='') as thp_file:
        thpr = csv.DictReader(thp_file)
        volt_values = [row for row in (parse_datetime_volts(row) for row in
            volt_rows) if row.get('last_acal_1_cal72',
            row.get('last_acal_2_cal72', -1)) > 0.0]
        min_date = min((row['datetime'] for row in volt_values))
        max_date = max((row['datetime'] for row in volt_values))
        thp_values = [row for row in (parse_datetime(row) for row in thpr) if
                min_date <= row['datetime'] <= max_date]
        return (thp_values, volt_values)

def get_column_from_dictlist(dictlist, column):
    return (r[column] for r in dictlist if column in r)

def plot_volt(dictlist, ax):
    datetimes = list(get_column_from_dictlist(dictlist, 'datetime'))
    time_interval = datetimes[-1] - datetimes[0]
    lns = []
    y_min = float('+inf')
    y_max = float('-inf')
    columns = set()
    for row in dictlist:
        columns |= set(row.keys())
    for column in sorted(columns):
        if column == 'datetime': continue
        if column not in VOLT_COLUMNS: continue
        volt = np.array(list(get_column_from_dictlist(dictlist, column)))
        dates = list(map(mdates.date2num, get_column_from_dictlist((r for r in
            dictlist if column in r), 'datetime')))
        print("{0}: min/max: {1}/{2}, std (ppm of mean): {3}".format(column,
                min(volt), max(volt), volt.std() / volt.mean() * 1e6))
        print("Average drift per day: {0} ppm/day".format(
            ((volt[-1]-volt[0]) * 1e6) / (volt[0] *
                (time_interval.total_seconds() / (24 * 3600)))))
        ppm = (volt - volt.mean()) / volt.mean() * 1e6
        lns.append(ax.plot(dates, ppm, ',', label=column)[0])
        y_min = min(y_min, np.percentile(ppm, 1))
        y_max = max(y_max, np.percentile(ppm, 99))
    ax.set_xlabel("Time")
    ax.set_ylabel("ppm")
    ax.set_ylim([y_min, y_max])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(days7)
    ax.legend(handles=lns, numpoints=10, loc='best')

def plot_thp(thp_values, ax):

    par1 = ax.twinx()
    par2 = ax.twinx()

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    par1.set_ylabel("Humidity (%)")
    par2.set_ylabel("Pressure (Pa)")

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(0.8)

    dates = list(map(mdates.date2num, get_column_from_dictlist(thp_values, 'datetime')))
    p1, = ax.plot(dates, list(get_column_from_dictlist(thp_values, 'temperature')), ',', color=color1, label="Temperature")
    p2, = par1.plot(dates, list(get_column_from_dictlist(thp_values, 'humidity')), ',', color=color2, label="Humidity")
    p3, = par2.plot(dates, list(get_column_from_dictlist(thp_values, 'pressure')), ',', color=color3, label="Pressure")

    lns = [p1, p2, p3]
    ax.legend(handles=lns, numpoints=10, loc='best')

    # right, left, top, bottom
    par2.spines['right'].set_position(('outward', 60))
    # no x-ticks                 
    #par2.xaxis.set_ticks([])
    # Sometimes handy, same for xaxis
    #par2.yaxis.set_ticks_position('right')

    ax.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

def plot(thp_values, volt_values):
    fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True)
    plot_volt(volt_values, axarr[0])
    plot_thp(thp_values, axarr[1])

thp_values, volt_values = read_data()
plot(thp_values, volt_values)
plt.show()
