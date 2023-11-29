import pandas as pd
import numpy as np
from IPython.display import display

from common_analysis import (
    add_dut_and_setting_group,
    combine_stds_mean,
    combine_stds_ratio_product,
    combine_stds_sum,
    std_minus_first,
    std_minus_last,
)


def analyse_dcv(absolute_data, relative_data, meter_absolute, meter_relative):
    absolute_results, absolute_ratios_in_ppm = analyse_dcv_absolute(absolute_data, "D4910avg", meter_absolute)
    f7001_value = absolute_results[absolute_results.index == "F7001bat"][["dcv_mean"]].iloc[0, 0]
    relative_results_in_ppm = analyse_dcv_relative(relative_data, "F7001bat", f7001_value, "D4910avg", meter_relative)
    return _dcv_combine_absolute_and_relative(absolute_ratios_in_ppm, relative_results_in_ppm)


def analyse_dcv_k182(relative_data_k182, substract_short_offset=False):
    return analyse_dcv_k182_for_other_reference(relative_data_k182, "F7001bat", 10, substract_short_offset)


def analyse_dcv_k182_for_other_reference(relative_data_k182, reference_name, reference_value=10, substract_short_offset=False):
    if substract_short_offset:
        short_voltage = relative_data_k182[relative_data_k182.dut_pos_lead != "short"]["k182_dcv"].mean()
    else:
        short_voltage = 0
    relative_data = relative_data_k182[(relative_data_k182.dut_pos_lead != "short")]
    results = analyse_dcv_relative(relative_data, reference_name, reference_value, reference_name, "k182", short_voltage)
    return results


def analyse_dcv_relative(relative_data, reference_name, reference_value, new_reference_name, meter, short_offset=0):
    filtered_data = relative_data[
        (relative_data.dut_neg_lead == reference_name) | (relative_data.dut_pos_lead == reference_name)
    ].copy()
    filtered_data = _relative_dcv_substract_offset(filtered_data, meter, short_offset)
    relative_dcv_add_polarity(filtered_data, reference_name, meter)
    relative_results_in_ppm = relative_results_to_ppm(
        filtered_data, reference_name, reference_value, new_reference_name
    )

    if reference_name != new_reference_name:
        relative_results_in_ppm = _retarget_reference(relative_results_in_ppm, reference_name, new_reference_name)

    return relative_results_in_ppm


def analyse_dcv_absolute(absolute_data, reference_name, meter, skip_bad_groups=False, with_pressure_and_humidity=False):
    absolute_data_with_groups = add_dut_and_setting_group(absolute_data)
    # display(analyse_group_quality(absolute_data_with_groups, f'{meter}_dcv'))
    absolute_data_first_and_last_in_group_removed = clean_groups(absolute_data_with_groups, meter, skip_bad_groups)
    cleaned_absolute_data = aggregate_absolute_data_by_group(
        absolute_data_first_and_last_in_group_removed, meter, with_pressure_and_humidity
    )
    absolute_grouped_by_dut_group = aggregate_absolute_data_by_dut_group(
        cleaned_absolute_data, meter, with_pressure_and_humidity
    )
    if with_pressure_and_humidity:
        agg = {
            "dcv_mean": "mean",
            "dcv_sem": combine_stds_mean,
            "dcv_std": combine_stds_mean,
            "temperature_mean": "mean",
            "pressure_mean": "mean",
            "humidity_mean": "mean",
            "datetime": "mean",
        }
    else:
        agg = {"dcv_mean": "mean", "dcv_sem": combine_stds_mean, "dcv_std": combine_stds_mean, "temperature_mean": "mean", "datetime": "mean"}
    absolute_results = absolute_grouped_by_dut_group.groupby("dut").agg(agg)
    ratios_from_absolute = dcv_calculate_ratios(absolute_grouped_by_dut_group, reference_name)
    ratios_in_ppm = _absolute_results_to_ppm(ratios_from_absolute)
    return absolute_results, ratios_in_ppm


def dcv_calculate_ratios(grouped_by_dut, reference):
    refs = grouped_by_dut[grouped_by_dut.dut == reference]
    duts = grouped_by_dut[grouped_by_dut.dut != reference]
    ratio_input = duts.apply(lambda x: _dcv_add_prev_and_next_refs(refs, grouped_by_dut, x.name), axis=1)
    if len(duts) == 0:
        return pd.DataFrame({"ratio": 1, "ratio_sem": 0, "ratio_std": 0, "temperature_mean": np.nan}, index=(reference,))
    ratios_before_input = ratio_input[~ratio_input["dut_before"].isna()].copy()
    ratios_before_input["ratio"] = ratios_before_input.dcv_mean / ratios_before_input.dcv_mean_before
    ratios_before_input["ratio_sem"] = combine_stds_ratio_product(
        ratios_before_input.ratio,
        ratios_before_input.dcv_mean,
        ratios_before_input.dcv_sem,
        ratios_before_input.dcv_mean_before,
        ratios_before_input.dcv_sem_before,
    )
    ratios_before_input["ratio_std"] = combine_stds_ratio_product(
        ratios_before_input.ratio,
        ratios_before_input.dcv_mean,
        ratios_before_input.dcv_std,
        ratios_before_input.dcv_mean_before,
        ratios_before_input.dcv_std_before,
    )

    ratios_after_input = ratio_input[~ratio_input.dut_after.isna()].copy()
    ratios_after_input["ratio"] = ratios_after_input.dcv_mean / ratios_after_input.dcv_mean_after
    ratios_after_input["ratio_sem"] = combine_stds_ratio_product(
        ratios_after_input.ratio,
        ratios_after_input.dcv_mean,
        ratios_after_input.dcv_sem,
        ratios_after_input.dcv_mean_after,
        ratios_after_input.dcv_sem_after,
    )
    ratios_after_input["ratio_std"] = combine_stds_ratio_product(
        ratios_after_input.ratio,
        ratios_after_input.dcv_mean,
        ratios_after_input.dcv_std,
        ratios_after_input.dcv_mean_after,
        ratios_after_input.dcv_std_after,
    )

    ratios_before_and_after = pd.concat(
        [
            ratios_before_input[["dut", "ratio", "ratio_sem", "ratio_std", "temperature_mean"]],
            ratios_after_input[["dut", "ratio", "ratio_sem", "ratio_std", "temperature_mean"]],
        ]
    )
    ratios_from_absolute = ratios_before_and_after.groupby("dut").agg(
        {"ratio": "mean", "ratio_sem": combine_stds_mean, "ratio_std": combine_stds_mean, "temperature_mean": "mean"}
    )
    ratios_from_absolute = pd.concat(
        [
            ratios_from_absolute,
            pd.DataFrame({"ratio": 1, "ratio_sem": 0, "ratio_std": 0, "temperature_mean": np.nan}, index=(reference,)),
        ]
    )
    return ratios_from_absolute


def aggregate_absolute_data_by_group(data, meter, with_pressure_and_humidity=False):
    if with_pressure_and_humidity:
        agg = {
            f"{meter}_dcv": ["mean", "std", "sem", "count"],
            "temperature": ["mean", "std", "sem", "count"],
            "pressure": ["mean", "std", "sem", "count"],
            "humidity": ["mean", "std", "sem", "count"],
            "dut": "last",
            "dut_setting": "last",
            "datetime": "mean",
        }
    else:
        agg = {
            f"{meter}_dcv": ["mean", "std", "sem", "count"],
            "temperature": ["mean", "std", "sem", "count"],
            "dut": "last",
            "dut_setting": "last",
            "datetime": "mean",
        }

    return data.reset_index().groupby("group").agg(agg)


def aggregate_absolute_data_by_dut_group(absolute_dcv_data, meter, with_pressure_and_humidity=False):
    data_with_dut_group = absolute_dcv_data.copy()
    data_with_dut_group["dut_group"] = (
        data_with_dut_group["dut"]["last"] != data_with_dut_group["dut"]["last"].shift(1)
    ).cumsum()
    data_with_dut_group.columns = ["_".join(col) for col in data_with_dut_group.columns.values]
    if with_pressure_and_humidity:
        agg = {
            "dut_last": "last",
            f"{meter}_dcv_mean": lambda v: np.mean(np.abs(v)),
            (f"{meter}_dcv_sem"): combine_stds_mean,
            (f"{meter}_dcv_std"): combine_stds_mean,
            "temperature_mean": "mean",
            "pressure_mean": "mean",
            "humidity_mean": "mean",
            "datetime_mean": "mean",
        }
        columns = [
            "dut",
            "dcv_mean",
            "dcv_sem",
            "dcv_std",
            "temperature_mean",
            "pressure_mean",
            "humidity_mean",
            "datetime",
        ]
    else:
        agg = {
            "dut_last": "last",
            f"{meter}_dcv_mean": lambda v: np.mean(np.abs(v)),
            (f"{meter}_dcv_sem"): combine_stds_sum,
            (f"{meter}_dcv_std"): combine_stds_sum,
            "temperature_mean": "mean",
            "datetime_mean": "mean",
        }
        columns = [
            "dut",
            "dcv_mean",
            "dcv_sem",
            "dcv_std",
            "temperature_mean",
            "datetime",
        ]

    data_grouped_by_dut = data_with_dut_group.groupby("dut_group_").agg(agg)
    data_grouped_by_dut.columns = columns
    return data_grouped_by_dut


def analyse_group_quality(data, column):
    return data.groupby("group").agg({column: ["std", std_minus_first, std_minus_last]})


def clean_groups(data, meter, skip_bad_groups=False):
    groups = data.groupby("group").apply(lambda x: x.iloc[1:-1]).droplevel(0)
    quality = groups.groupby("group").agg({f"{meter}_dcv": "std", "dut": "last"})
    if skip_bad_groups:
        bad_group_index = quality[f"{meter}_dcv"] >= 1e-6
        if bad_group_index.any():
            bad_groups = quality[bad_group_index]
            display("Found bad groups:")
            display(bad_groups)
            return groups[~groups.group.isin(bad_groups.index)]
    return groups


def relative_dcv_add_polarity(data, reference_name, meter):
    data["polarity"] = data.dut_neg_lead.apply(lambda dut: "positive" if dut == reference_name else "negative")
    data["dut"] = data.apply(lambda row: row.dut_neg_lead if row.polarity == "negative" else row.dut_pos_lead, axis=1)
    data.loc[data["polarity"] == "positive", "corrected_value"] = data[f"{meter}_dcv"]
    data.loc[data["polarity"] == "negative", "corrected_value"] = -data[f"{meter}_dcv"]
    _check_sign(data, meter)


def relative_results_to_ppm(relative_data, reference_name, reference_value, new_reference_name):
    grouped_by_dut_polarity = (
        relative_data.reset_index()
        .groupby(["dut", "polarity"])
        .agg(
            {
                "corrected_value": ["mean", "std", "sem", "count"],
                "datetime": "mean",
                "temperature": "mean",
                "pressure": "mean",
                "humidity": "mean",
            }
        )
        .reset_index()
    )
    grouped_by_dut_polarity.columns = (
        "dut",
        "polarity",
        "mean",
        "std",
        "sem",
        "count",
        "datetime",
        "temperature",
        "pressure",
        "humidity",
    )
    relative_results = grouped_by_dut_polarity.groupby("dut").agg(
        {
            "mean": "mean",
            "sem": combine_stds_mean,
            "std": combine_stds_mean,
            "datetime": "mean",
            "temperature": "mean",
            "pressure": "mean",
            "humidity": "mean",
        }
    )
    relative_results_in_ppm = pd.DataFrame()
    relative_results_in_ppm.index = relative_results.index
    relative_results_in_ppm["mean_in_ppm"] = (relative_results["mean"] / reference_value) * 1e6
    relative_results_in_ppm["sem_in_ppm"] = (relative_results["sem"] / reference_value) * 1e6
    relative_results_in_ppm["std_in_ppm"] = (relative_results["std"] / reference_value) * 1e6
    relative_results_in_ppm["datetime"] = relative_results["datetime"]
    relative_results_in_ppm["temperature"] = relative_results.temperature
    relative_results_in_ppm["pressure"] = relative_results.pressure
    relative_results_in_ppm["humidity"] = relative_results.humidity
    return relative_results_in_ppm


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


def _relative_dcv_substract_offset(relative_data, meter, short_offset):
    relative_data[f"{meter}_dcv"] -= short_offset
    return relative_data


def _retarget_reference(relative_results_in_ppm, reference_name, new_reference_name):
    relative_results_in_ppm = pd.concat(
        [relative_results_in_ppm, pd.DataFrame({"mean_in_ppm": 0, "sem_in_ppm": 0, "std_in_ppm": 0}, index=(reference_name,))]
    )
    relative_results_in_ppm["mean_in_ppm"] = (
        relative_results_in_ppm[relative_results_in_ppm.index == new_reference_name].mean_in_ppm.iloc[0]
        - relative_results_in_ppm.mean_in_ppm
    )
    relative_results_in_ppm["sem_in_ppm"] = np.sqrt(
        relative_results_in_ppm[relative_results_in_ppm.index == new_reference_name].sem_in_ppm.iloc[0] ** 2
        + relative_results_in_ppm.sem_in_ppm**2
    )
    relative_results_in_ppm["std_in_ppm"] = np.sqrt(
        relative_results_in_ppm[relative_results_in_ppm.index == new_reference_name].std_in_ppm.iloc[0] ** 2
        + relative_results_in_ppm.std_in_ppm**2
    )
    return relative_results_in_ppm


def _check_sign(data, meter):
    data["sign"] = data[f"{meter}_dcv"] / data[f"{meter}_dcv"].abs()
    check_sign_data = data.reset_index().groupby(["dut", "polarity"]).agg({"sign": "unique", "datetime": "first"})
    check_sign_data["sign_length"] = check_sign_data["sign"].apply(lambda r: len(r))
    sign_failures = check_sign_data[check_sign_data.sign_length > 1]
    if not sign_failures.empty:
        display("Sign flip in measurement with same reported polarity and dut")
        display(sign_failures)
        for dut in sign_failures.reset_index().dut.unique():
            display(data[data.dut == dut])


def _absolute_results_to_ppm(ratios):
    ratios_ppm = ratios.copy().drop(["ratio", "ratio_sem", "ratio_std"], axis=1)
    ratios_ppm["ppm_diff"] = (1 - ratios.ratio) * 1e6
    ratios_ppm["ppm_sem"] = ratios.ratio_sem * 1e6
    ratios_ppm["ppm_std"] = ratios.ratio_std * 1e6
    return ratios_ppm


def _dcv_add_prev_and_next_refs(refs, duts, dut_index):
    refs_with_dut = refs.copy()
    refs_with_dut.loc[dut_index] = duts.loc[dut_index]
    refs_with_dut.sort_index(inplace=True)
    return pd.concat(
        [refs_with_dut, refs_with_dut.shift(1).add_suffix("_before"), refs_with_dut.shift(-1).add_suffix("_after")],
        axis=1,
    ).loc[dut_index]


def _dcv_combine_absolute_and_relative(ratios_ppm, relative_results_in_ppm):
    combined = ratios_ppm.join(relative_results_in_ppm)
    combined.columns = [
        "abs_temperature",
        "abs_mean",
        "abs_sem",
        "abs_std",
        "rel_mean",
        "rel_sem",
        "rel_std",
        "rel_datetime",
        "rel_temperature",
        "rel_pressure",
        "rel_humidify",
    ]
    return combined
