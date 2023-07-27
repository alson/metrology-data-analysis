import pandas as pd
import numpy as np


def analyse_ohms(
    data,
    meter="ag3458a_2",
    remove_first_and_last=True,
    temperature_columns=["temperature"],
):
    data_with_groups = add_dut_and_setting_group(data)
    if remove_first_and_last:
        cleaned = (
            data_with_groups.groupby("group").apply(lambda x: x.iloc[1:-1]).droplevel(0)
        )
    else:
        cleaned = data_with_groups
    analyse_grouped_ohms(cleaned, meter, temperature_columns)


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
    data_with_dut_group.columns = [
        "_".join(col) for col in data_with_dut_group.columns.values
    ]
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


def analyse_dcv(absolute_data, relative_data, meter_absolute, meter_relative):
    absolute_results, absolute_ratios_in_ppm = analyse_dcv_absolute(
        absolute_data, "D4910avg", meter_absolute
    )
    f7001_value = absolute_results[absolute_results.index == "F7001bat"][
        ["dcv_mean"]
    ].iloc[0, 0]
    relative_results_in_ppm = analyse_dcv_relative(
        relative_data, "F7001bat", f7001_value, "D4910avg", meter_relative
    )
    return dcv_combine_absolute_and_relative(
        absolute_ratios_in_ppm, relative_results_in_ppm
    )


def dcv_combine_absolute_and_relative(ratios_ppm, relative_results_in_ppm):
    combined = ratios_ppm.join(relative_results_in_ppm)
    combined.columns = ["temperature", "abs_mean", "abs_sem", "rel_mean", "rel_sem"]
    return combined


def analyse_dcv_relative(
    relative_data, reference_name, reference_value, new_reference_name, meter
):
    relative_dcv_add_polarity(relative_data, reference_name, meter)
    relative_results_in_ppm = relative_results_to_ppm(
        relative_data, reference_name, reference_value, new_reference_name
    )
    return relative_results_in_ppm


def analyse_dcv_absolute(absolute_data, reference_name, meter):
    absolute_data_with_groups = add_dut_and_setting_group(absolute_data)
    absolute_data = analyse_group_quality(absolute_data_with_groups, meter)
    absolute_data_first_and_last_in_group_removed = clean_groups(
        absolute_data_with_groups, meter
    )
    cleaned_absolute_data = aggregate_absolute_data_by_group(
        absolute_data_first_and_last_in_group_removed, meter
    )
    # display(cleaned_absolute_data)
    absolute_grouped_by_dut_group = aggregate_absolute_data_by_dut_group(
        cleaned_absolute_data, meter
    )
    absolute_results = absolute_grouped_by_dut_group.groupby("dut").agg(
        {"dcv_mean": "mean", "dcv_sem": combine_stds_sum, "temperature_mean": "mean"}
    )
    ratios_from_absolute = dcv_calculate_ratios(
        absolute_grouped_by_dut_group, reference_name
    )
    ratios_in_ppm = absolute_results_to_ppm(ratios_from_absolute)
    return absolute_results, ratios_in_ppm


def combine_stds_sum(stds):
    return np.sqrt(np.sum(stds**2) / np.size(stds))


def combine_stds_ratio_product(product_or_ration_value, mean1, sem1, mean2, sem2):
    return np.abs(product_or_ration_value) * np.sqrt(
        (sem1 / mean1) ** 2 + (sem2 / mean2) ** 2
    )


def add_dut_and_setting_group(data):
    data_groups = (
        (
            data[["dut", "dut_setting"]].apply(tuple, axis=1)
            != data[["dut", "dut_setting"]].shift().apply(tuple, axis=1)
        )
        .cumsum()
        .rename("group")
    )
    return data.join(data_groups)


def add_dut_neg_and_pos_group(data):
    data_groups = (
        (
            data[["dut_neg_lead", "dut_pos_lead"]].apply(tuple, axis=1)
            != data[["dut_neg_lead", "dut_pos_lead"]].shift().apply(tuple, axis=1)
        )
        .cumsum()
        .rename("group")
    )
    return data.join(data_groups)


def std_minus_first(series):
    return np.std(series.iloc[:-1])


def std_minus_last(series):
    return np.std(series.iloc[1:])


def analyse_group_quality(data, meter):
    # groups = data.groupby('group').agg({f'{meter}_dcv': ['std', std_minus_first, std_minus_last]})
    return data


def clean_groups(data, meter):
    groups = data.groupby("group").apply(lambda x: x.iloc[1:-1]).droplevel(0)
    quality = groups.groupby("group").agg({f"{meter}_dcv": "std", "dut": "last"})
    bad_group_index = quality[f"{meter}_dcv"] >= 1e-6
    if bad_group_index.any():
        bad_groups = quality[bad_group_index]
        display("Found bad groups:")
        display(bad_groups)
        return groups[~groups.group.isin(bad_groups.index)]
    return groups


def aggregate_absolute_data_by_group(data, meter):
    return (
        data.reset_index()
        .groupby("group")
        .agg(
            {
                f"{meter}_dcv": ["mean", "std", "sem", "count"],
                "temperature": ["mean", "std", "sem", "count"],
                "dut": "last",
                "dut_setting": "last",
                "datetime": "mean",
            }
        )
    )


def aggregate_absolute_data_by_dut_group(absolute_dcv_data, meter):
    data_with_dut_group = absolute_dcv_data.copy()
    # display(absolute_dcv_data)
    data_with_dut_group["dut_group"] = (
        data_with_dut_group["dut"]["last"]
        != data_with_dut_group["dut"]["last"].shift(1)
    ).cumsum()
    # display(    data_with_dut_group)
    # display((data_with_dut_group['dut']['last'] != data_with_dut_group['dut']['last'].shift(-1)))
    data_with_dut_group.columns = [
        "_".join(col) for col in data_with_dut_group.columns.values
    ]
    data_grouped_by_dut = data_with_dut_group.groupby("dut_group_").agg(
        {
            "dut_last": "last",
            f"{meter}_dcv_mean": lambda v: np.mean(np.abs(v)),
            (f"{meter}_dcv_sem"): combine_stds_sum,
            "temperature_mean": "mean",
            "datetime_mean": "mean",
        }
    )
    data_grouped_by_dut.columns = [
        "dut",
        "dcv_mean",
        "dcv_sem",
        "temperature_mean",
        "datetime",
    ]
    # display(data_grouped_by_dut)
    # display(    data_with_dut_group)
    return data_grouped_by_dut


def dcv_calculate_ratios(grouped_by_dut, reference):
    refs = grouped_by_dut[grouped_by_dut.dut == reference]
    duts = grouped_by_dut[grouped_by_dut.dut != reference]
    ratio_input = duts.apply(
        lambda x: dcv_add_prev_and_next_refs(refs, grouped_by_dut, x.name), axis=1
    )

    ratios_before_input = ratio_input[~ratio_input.dut_before.isna()].copy()
    ratios_before_input["ratio"] = (
        ratios_before_input.dcv_mean / ratios_before_input.dcv_mean_before
    )
    ratios_before_input["ratio_sem"] = combine_stds_ratio_product(
        ratios_before_input.ratio,
        ratios_before_input.dcv_mean,
        ratios_before_input.dcv_sem,
        ratios_before_input.dcv_mean_before,
        ratios_before_input.dcv_sem_before,
    )

    ratios_after_input = ratio_input[~ratio_input.dut_after.isna()].copy()
    ratios_after_input["ratio"] = (
        ratios_after_input.dcv_mean / ratios_after_input.dcv_mean_after
    )
    ratios_after_input["ratio_sem"] = combine_stds_ratio_product(
        ratios_after_input.ratio,
        ratios_after_input.dcv_mean,
        ratios_after_input.dcv_sem,
        ratios_after_input.dcv_mean_after,
        ratios_after_input.dcv_sem_after,
    )

    ratios_before_and_after = pd.concat(
        [
            ratios_before_input[["dut", "ratio", "ratio_sem", "temperature_mean"]],
            ratios_after_input[["dut", "ratio", "ratio_sem", "temperature_mean"]],
        ]
    )
    ratios_from_absolute = ratios_before_and_after.groupby("dut").agg(
        {"ratio": "mean", "ratio_sem": combine_stds_sum, "temperature_mean": "mean"}
    )
    ratios_from_absolute = pd.concat(
        [
            ratios_from_absolute,
            pd.DataFrame(
                {"ratio": 1, "ratio_sem": 0, "temperature_mean": np.nan},
                index=(reference,),
            ),
        ]
    )
    return ratios_from_absolute


def dcv_add_prev_and_next_refs(refs, duts, dut_index):
    refs_with_dut = refs.copy()
    refs_with_dut.loc[dut_index] = duts.loc[dut_index]
    refs_with_dut.sort_index(inplace=True)
    return pd.concat(
        [
            refs_with_dut,
            refs_with_dut.shift(1).add_suffix("_before"),
            refs_with_dut.shift(-1).add_suffix("_after"),
        ],
        axis=1,
    ).loc[dut_index]


def relative_dcv_add_polarity(data, reference_name, meter):
    data["polarity"] = data.dut_neg_lead.apply(
        lambda dut: "positive" if dut == reference_name else "negative"
    )
    data["dut"] = data.apply(
        lambda row: row.dut_neg_lead
        if row.polarity == "negative"
        else row.dut_pos_lead,
        axis=1,
    )
    data.loc[data["polarity"] == "positive", "corrected_value"] = data[f"{meter}_dcv"]
    data.loc[data["polarity"] == "negative", "corrected_value"] = -data[f"{meter}_dcv"]


def relative_results_to_ppm(
    relative_data, reference_name, reference_value, new_reference_name
):
    grouped_by_dut_polarity = (
        relative_data.groupby(["dut", "polarity"])
        .agg({"corrected_value": ["mean", "std", "sem", "count"]})
        .droplevel(0, axis=1)
        .reset_index()
    )
    relative_results = grouped_by_dut_polarity.groupby("dut").agg(
        {"mean": "mean", "sem": combine_stds_sum}
    )
    relative_results_in_ppm = pd.DataFrame()
    relative_results_in_ppm.index = relative_results.index
    relative_results_in_ppm["mean_in_ppm"] = (
        relative_results["mean"] / reference_value
    ) * 1e6
    relative_results_in_ppm["sem_in_ppm"] = (
        relative_results["sem"] / reference_value
    ) * 1e6
    relative_results_in_ppm = pd.concat(
        [
            relative_results_in_ppm,
            pd.DataFrame({"mean_in_ppm": 0, "sem_in_ppm": 0}, index=(reference_name,)),
        ]
    )
    relative_results_in_ppm["mean_in_ppm"] = (
        relative_results_in_ppm[
            relative_results_in_ppm.index == new_reference_name
        ].mean_in_ppm.iloc[0]
        - relative_results_in_ppm.mean_in_ppm
    )
    relative_results_in_ppm["sem_in_ppm"] = np.sqrt(
        relative_results_in_ppm[
            relative_results_in_ppm.index == new_reference_name
        ].sem_in_ppm.iloc[0]
        ** 2
        + relative_results_in_ppm.sem_in_ppm**2
    )
    return relative_results_in_ppm


def absolute_results_to_ppm(ratios):
    ratios_ppm = ratios.copy().drop(["ratio", "ratio_sem"], axis=1)
    ratios_ppm["ppm_diff"] = (1 - ratios.ratio) * 1e6
    ratios_ppm["ppm_sem"] = ratios.ratio_sem * 1e6
    return ratios_ppm


def rel_data_cut_index_last(data, group, dut_neg_lead, dut_pos_lead, timedelta):
    cut = data[
        (data.group == group)
        & (data.dut_neg_lead == dut_neg_lead)
        & (data.dut_pos_lead == dut_pos_lead)
    ]
    return (
        (data.group == group)
        & (data.dut_neg_lead == dut_neg_lead)
        & (data.dut_pos_lead == dut_pos_lead)
        & (data.index < (cut.index[-1] - timedelta))
    )


def rel_data_cut_index_first(data, group, dut_neg_lead, dut_pos_lead, timedelta):
    cut = data[
        (data.group == group)
        & (data.dut_neg_lead == dut_neg_lead)
        & (data.dut_pos_lead == dut_pos_lead)
    ]
    return (
        (data.group == group)
        & (data.dut_neg_lead == dut_neg_lead)
        & (data.dut_pos_lead == dut_pos_lead)
        & (data.index > (cut.index[0] + timedelta))
    )


def filter_acal_points(data, ks3458a_number):
    is_acal = data[f"last_acal_{ks3458a_number}"] != data[
        f"last_acal_{ks3458a_number}"
    ].shift(1)
    return data[is_acal]


def add_sr104_temp(data, ohm_column_name):
    base_name, _, unit = ohm_column_name.rpartition("_")
    assert unit == "ohm"
    temp_column_name = f"{base_name}_degC"
    data[temp_column_name] = Trh(data[ohm_column_name])


def add_pt100_temp(data, ohm_column_name):
    base_name, _, unit = ohm_column_name.rpartition("_")
    assert unit == "ohm"
    temp_column_name = f"{base_name}_degC"
    data[temp_column_name] = data[ohm_column_name].apply(
        lambda resistance: fsolve(PT385_eq, 25, args=(resistance,))[0]
    )


def correct_sr104(data, uncorrected_column_name, temperature_column):
    base_name, _, unit = uncorrected_column_name.rpartition("_")
    relative_column_name = f"{base_name}_ppm"
    nominal_sr104 = Rt(data[temperature_column])
    data[relative_column_name] = (
        data[uncorrected_column_name] / nominal_sr104 - 1
    ) * 1e6


def make_sr104_relative(data, absolute_column_name):
    base_name, unit, _ = absolute_column_name.rsplit("_", 2)
    relative_column_name = f"{base_name}_ppm"
    data[relative_column_name] = (data[absolute_column_name] - Rs23) / Rs23 * 1e6


def interpolate_temp(data, temp_column_name):
    interpolated_column_name = temp_column_name + "_interpolated"
    data[interpolated_column_name] = data[temp_column_name].interpolate()


# From https://us.flukecal.com/pt100-calculator for PT-385
R0 = 100
A = 3.9083e-3
B = -5.775e-7
PT385_eq = lambda t, Rt: R0 * (1 + A * t + B * (t**2)) - Rt

# From my SR104 data
Rth = lambda T: (0.1e-2 * 10e3) * (T - 23) + ((1 + 0.003e-2) * 10e3)
Trh = lambda Rt: (Rt - ((1 + 0.003e-2) * 10e3)) / (0.1e-2 * 10e3) + 23
Rs23 = 10e3 * (1 - 1.4e-6)
Rt = lambda T: Rs23 * (1 - 0.04e-6 * (T - 23) - 0.025e-6 * (T - 23) ** 2)
