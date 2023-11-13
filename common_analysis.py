import pandas as pd
import numpy as np
from IPython.display import display


def add_dut_and_setting_group(data):
    data_groups = (
        (data[["dut", "dut_setting"]].apply(tuple, axis=1) != data[["dut", "dut_setting"]].shift().apply(tuple, axis=1))
        .cumsum()
        .rename("group")
    )
    return data.join(data_groups)


def combine_stds_sum(stds):
    return np.sqrt(np.sum(stds**2))


def combine_stds_mean(stds):
    return np.sqrt(np.sum(stds**2) / np.size(stds))


def combine_stds_ratio_product(product_or_ration_value, mean1, sem1, mean2, sem2):
    return np.abs(product_or_ration_value) * np.sqrt((sem1 / mean1) ** 2 + (sem2 / mean2) ** 2)


def rel_data_cut_index_last(data, group, dut_neg_lead, dut_pos_lead, timedelta, before=None):
    cut = data[(data.group == group) & (data.dut_neg_lead == dut_neg_lead) & (data.dut_pos_lead == dut_pos_lead)]
    return (
        (data.group == group)
        & (data.dut_neg_lead == dut_neg_lead)
        & (data.dut_pos_lead == dut_pos_lead)
        & (data.index < ((before if before else cut.index[-1]) - timedelta))
    )


def rel_data_cut_index_first(data, group, dut_neg_lead, dut_pos_lead, timedelta, after=None):
    cut = data[(data.group == group) & (data.dut_neg_lead == dut_neg_lead) & (data.dut_pos_lead == dut_pos_lead)]
    return (
        (data.group == group)
        & (data.dut_neg_lead == dut_neg_lead)
        & (data.dut_pos_lead == dut_pos_lead)
        & (data.index > ((after if after else cut.index[0]) + timedelta))
    )


def abs_data_cut_index_last(data, group, dut, dut_setting, timedelta, before=None):
    cut = data[(data.group == group) & (data.dut == dut) & (data.dut_setting == dut_setting)]
    return (
        (data.group == group)
        & (data.dut == dut)
        & (data.dut_setting == dut_setting)
        & (data.index < ((before if before else cut.index[-1]) - timedelta))
    )


def abs_data_cut_index_first(data, group, dut, dut_setting, timedelta, after=None):
    cut = data[(data.group == group) & (data.dut == dut) & (data.dut_setting == dut_setting)]
    return (
        (data.group == group)
        & (data.dut == dut)
        & (data.dut_setting == dut_setting)
        & (data.index > ((after if after else cut.index[0]) + timedelta))
    )


def std_minus_first(series):
    return np.std(series.iloc[:-1])


def std_minus_last(series):
    return np.std(series.iloc[1:])


def display_full_df(df):
    orig_max_rows = pd.get_option("display.max_rows")
    pd.set_option("display.max_rows", None)
    display(df)
    pd.set_option("display.max_rows", orig_max_rows)


def filter_acal_points(data, ks3458a_number):
    is_acal = data[f"last_acal_{ks3458a_number}"] != data[f"last_acal_{ks3458a_number}"].shift(1)
    return data[is_acal]