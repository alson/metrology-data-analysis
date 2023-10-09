from scipy.optimize import fsolve

from common_analysis import add_dut_and_setting_group


def analyse_ohms(
    data,
    meter="ag3458a_2",
    remove_first_and_last=True,
    temperature_columns=["temperature"],
):
    data_with_groups = add_dut_and_setting_group(data)
    if remove_first_and_last:
        cleaned = data_with_groups.groupby("group").apply(lambda x: x.iloc[1:-1]).droplevel(0)
    else:
        cleaned = data_with_groups
    return analyse_grouped_ohms(cleaned, meter, temperature_columns)

def analyse_grouped_ohms(data, meter="ag3458a_2", temperature_columns=["temperature"]):
    agg_dict = {
        f"{meter}_ohm": ["mean", "std", "sem", "count"],
        "dut": "last",
        "dut_setting": "last",
        f"{meter}_range": "last",
    }
    for temp_col in temperature_columns:
        agg_dict[temp_col] = ["mean", "std", "sem", "count"]
    data_grouped_by_dut_setting = data.groupby("group").agg(agg_dict)
    data_with_dut_group = data_grouped_by_dut_setting.copy()
    data_with_dut_group.columns = ["_".join(col) for col in data_with_dut_group.columns.values]
    data_grouped_by_dut = data_with_dut_group[
        [
            "dut_last",
            "dut_setting_last",
            f"{meter}_ohm_count",
            f"{meter}_ohm_mean",
            f"{meter}_ohm_sem",
            f"{meter}_ohm_std",
            f"{meter}_range_last",
        ]
        + [f"{temp_col}_mean" for temp_col in temperature_columns]
    ]
    data_grouped_by_dut.columns = [
        "dut",
        "dut_setting",
        "count",
        "ohm_mean",
        "ohm_sem",
        "ohm_std",
        "range",
    ] + [f"{t}_mean" for t in temperature_columns]
    return data_grouped_by_dut


def add_sr104_temp(data, ohm_column_name):
    base_name, _, unit = ohm_column_name.rpartition("_")
    assert unit == "ohm"
    temp_column_name = f"{base_name}_degC"
    data[temp_column_name] = Trh(data[ohm_column_name])


def add_pt100_temp(data, ohm_column_name):
    base_name, _, unit = ohm_column_name.rpartition("_")
    assert unit == "ohm"
    temp_column_name = f"{base_name}_degC"
    data[temp_column_name] = data[ohm_column_name].apply(lambda resistance: fsolve(PT385_eq, 25, args=(resistance,))[0])


def correct_sr104(data, uncorrected_column_name, temperature_column):
    base_name, _, unit = uncorrected_column_name.rpartition("_")
    relative_column_name = f"{base_name}_ppm"
    nominal_sr104 = Rt(data[temperature_column])
    data[relative_column_name] = (data[uncorrected_column_name] / nominal_sr104 - 1) * 1e6


def make_sr104_relative(data, absolute_column_name):
    base_name, unit, _ = absolute_column_name.rsplit("_", 2)
    relative_column_name = f"{base_name}_ppm"
    data[relative_column_name] = (data[absolute_column_name] - Rs23) / Rs23 * 1e6


def interpolate_temp(data, temp_column_name):
    interpolated_column_name = temp_column_name + "_interpolated"
    data[interpolated_column_name] = data[temp_column_name].interpolate()


# From https://us.flukecal.com/pt100-calculator for PT-385
_R0 = 100
_A = 3.9083e-3
_B = -5.775e-7
PT385_eq = lambda t, Rt: _R0 * (1 + _A * t + _B * (t**2)) - Rt

# From my SR104 data
Rth = lambda T: (0.1e-2 * 10e3) * (T - 23) + ((1 + 0.003e-2) * 10e3)
Trh = lambda Rt: (Rt - ((1 + 0.003e-2) * 10e3)) / (0.1e-2 * 10e3) + 23
Rs23 = 10e3 * (1 - 1.4e-6)
Rt = lambda T: Rs23 * (1 - 0.04e-6 * (T - 23) - 0.025e-6 * (T - 23) ** 2)
