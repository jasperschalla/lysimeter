#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :lysi_check.py
@Time     :2023/10/25 10:40:01
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""

import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import statsmodels.api as sm
import numpy as np
import datetime
import json
import pathlib
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb import InfluxDBClient as InfluxDBClientOld
from penmon import Station
import warnings
from random import randint


# Ignore warnings for concatenate empty pandas dataframes
warnings.simplefilter(action="ignore", category=FutureWarning)

# Open LysiCheck image
image = Image.open("./lysi_check_text.png")

# Configure pae layout, title and icon
st.set_page_config(
    layout="wide", page_title="LysiCheck", page_icon="./favicon_check.ico"
)

src_path = "./"

# Options for sidebar user input
types = ["gap_check", "postprocess"]
locations = ["FE1", "FE2", "FE3", "GW", "RB1", "RB2"]
data_type = ["additional", "balance"]

# Open configuration indicating which lysimeter has which partner lysimeter
with open(os.path.join(src_path, "lysimeter_matches.json"), "r") as f:
    lysimeter_matches = json.loads(f.read())

# Default lysimeter matches

default_lysimeter_matches = {
    "FE1": {"1": 5, "2": 5, "3": 6, "4": 6, "5": 2, "6": 5},
    "FE2": {"1": 6, "2": 6, "3": 5, "4": 1, "5": 3, "6": 1},
    "FE3": {"1": 2, "2": 4, "3": 4, "4": 6, "5": 6, "6": 5},
    "GW": {"1": 6, "2": 3, "3": 5, "4": 3, "5": 3, "6": 3},
    "RB1": {"1": 4, "2": 1, "3": 1, "4": 1, "5": 1, "6": 3},
    "RB2": {"1": 2, "2": 1, "3": 2, "4": 2, "5": 2, "6": 5},
}


# Helper functions


def reset_lysi_config():
    with open(os.path.join(src_path, "lysimeter_matches.json"), "w") as f:
        json.dump(default_lysimeter_matches, f)


def save_lysi_config(location, config):
    with open(os.path.join(src_path, "lysimeter_matches.json"), "w") as f:
        changed_matches = lysimeter_matches
        changed_matches[location] = config
        json.dump(changed_matches, f)


def downsample_gapdata_hourly(data, param_code):
    data_temp = data.resample("H", on=data.columns[0]).mean().reset_index()
    exclude_days = (
        data[data[param_code].isin([1, 2, -1, -2])]["Timestamp"]
        .dt.strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )

    data_temp_original = data[
        data[data.columns[0]].dt.strftime("%Y-%m-%d").isin(exclude_days)
    ]
    data_temp = data_temp[
        ~data_temp[data_temp.columns[0]].dt.strftime("%Y-%m-%d").isin(exclude_days)
    ]
    data_downsampled = (
        pd.concat([data_temp, data_temp_original])
        .sort_values(by=[data.columns[0]])
        .reset_index(drop=True)
    )
    return data_downsampled


def location_to_abbr(location):
    if re.match("GW.*", location):
        return "gwg"
    elif re.match("FE*", location):
        return "fen"
    elif re.match("RB*", location):
        return "rbw"
    else:
        return "gap"


def save_day(data, output_file, date_col, lysi_number):
    date_str = output_file.strftime("%Y-%m-%d")
    file_date_str = output_file.strftime("%Y_%m_%d")

    start_filter_date = datetime.datetime.strptime(
        f"{date_str} 00:00:00",
        "%Y-%m-%d %H:%M:%S",
    )
    end_filter_date = datetime.datetime.strptime(
        f"{date_str} 23:59:59",
        "%Y-%m-%d %H:%M:%S",
    )

    dataset_part = data[
        (data[date_col] >= start_filter_date) & (data[date_col] <= end_filter_date)
    ]

    dataset_part.to_csv(
        os.path.join(
            st.session_state["read_path"],
            f"{lysi_number+1}_{file_date_str}.csv",
        ),
        index=False,
    )


def save_days(data, start_index, end_index, date_col, lysi_number, fill_indexes=None):
    if not isinstance(fill_indexes, list):
        fill_dates = data.loc[start_index:end_index, date_col].dt.date.unique().tolist()
    else:
        fill_dates = data.loc[fill_indexes, date_col].dt.date.unique().tolist()

    for date in fill_dates:
        date_str = date.strftime("%Y-%m-%d")
        file_date_str = date.strftime("%Y_%m_%d")

        start_filter_date = datetime.datetime.strptime(
            f"{date_str} 00:00:00",
            "%Y-%m-%d %H:%M:%S",
        )
        end_filter_date = datetime.datetime.strptime(
            f"{date_str} 23:59:59",
            "%Y-%m-%d %H:%M:%S",
        )

        dataset_part = data[
            (data[data.columns[0]] >= start_filter_date)
            & (data[data.columns[0]] <= end_filter_date)
        ]

        dataset_part.to_csv(
            os.path.join(
                st.session_state["read_path"],
                f"{lysi_number+1}_{file_date_str}.csv",
            ),
            index=False,
        )


def undo_last_action(lysi_number):
    datasets = st.session_state["datasets"]
    data = st.session_state["previous_dataset"]
    datasets[lysi_number] = data
    st.session_state["datasets"] = datasets
    st.session_state["previous_dataset"] = None

    remove_start_index = 0
    remove_end_index = data.shape[0] - 1
    save_days(data, remove_start_index, remove_end_index, data.columns[0], lysi_number)
    st.rerun()


# Transform functions


def gap_undo_interpolation(na_date_selector, lysi_number):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]

    undo_index = data.index[data[data.columns[0]] == na_date_selector][0]
    data.loc[
        undo_index, [col_selector, col_selector_na_groups, col_selector_groups]
    ] = np.nan
    data.loc[undo_index, col_selector_code] = -1
    data.loc[undo_index, col_selector_msg] = "edge value"
    output_file = data.loc[undo_index, data.columns[0]]
    datasets[lysi_number] = data

    save_day(data, output_file, data.columns[0], lysi_number)
    st.session_state["datasets"] = datasets
    st.rerun()


def gap_undo_all(fill_date_selector, lysi_number, fill_type):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]

    if fill_type == "fill (lm)":
        remove_code = 2
        remove_msg = f"lysimeter {lysimeter_matches[location_summary][str(lysi_number+1)]} is also NA"
        replace_code = -2
    else:
        remove_code = 1
        remove_msg = "edge case"

        if fill_type == "fill (interpolation multi)":
            replace_code = -2
        else:
            replace_code = -1

    start_date = fill_date_selector[0]
    end_date = fill_date_selector[1]
    start_index = data.index[data[data.columns[0]] >= start_date].tolist()[0]
    end_index = data.index[data[data.columns[0]] <= end_date].tolist()[-1]

    if fill_type == "fill (interpolation multi)":
        replace_condition = (
            data.loc[start_index:end_index, col_selector_code] == remove_code
        ) & (~data.loc[start_index:end_index, col_selector_groups].isnull())
    elif fill_type == "fill (interpolation single)":
        replace_condition = (
            data.loc[start_index:end_index, col_selector_code] == remove_code
        ) & (data.loc[start_index:end_index, col_selector_groups].isnull())
    else:
        replace_condition = (
            data.loc[start_index:end_index, col_selector_code] == remove_code
        )

    data.loc[
        start_index:end_index,
        [
            col_selector,
        ],
    ] = np.where(
        replace_condition,
        np.nan,
        data.loc[start_index:end_index, col_selector],
    )

    data.loc[
        start_index:end_index,
        [
            col_selector_msg,
        ],
    ] = np.where(
        replace_condition,
        remove_msg,
        data.loc[start_index:end_index, col_selector_msg],
    )

    data.loc[
        start_index:end_index,
        [
            col_selector_na_groups,
        ],
    ] = np.where(
        replace_condition,
        data.loc[start_index:end_index, col_selector_groups] + 1e6,
        data.loc[start_index:end_index, col_selector_na_groups],
    )

    data.loc[
        start_index:end_index,
        [
            col_selector_groups,
        ],
    ] = np.where(
        replace_condition,
        np.nan,
        data.loc[start_index:end_index, col_selector_groups],
    )

    data.loc[
        start_index:end_index,
        [
            col_selector_code,
        ],
    ] = np.where(
        replace_condition,
        replace_code,
        data.loc[start_index:end_index, col_selector_code],
    )

    datasets[lysi_number] = data

    save_days(data, start_index, end_index, data.columns[0], lysi_number)
    st.session_state["datasets"] = datasets
    st.rerun()


def gap_undo_multi(na_date_selector, lysi_number):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]

    remove_code = -2
    remove_msg = f"lysimeter {lysimeter_matches[location_summary][str(lysi_number+1)]} is also NA"

    # Check for which date the gap is removed and update that file and remove the flags
    start_date = datetime.datetime.strptime(
        na_date_selector.split("to")[0].strip(),
        "%Y-%m-%d %H:%M:%S",
    )
    end_date = datetime.datetime.strptime(
        na_date_selector.split("to")[-1].strip(),
        "%Y-%m-%d %H:%M:%S",
    )

    start_index = data.index[data[data.columns[0]] == start_date][0]
    end_index = data.index[data[data.columns[0]] == end_date][0]

    data.loc[
        start_index:end_index,
        [
            col_selector,
            col_selector_groups,
        ],
    ] = np.nan
    data.loc[
        start_index:end_index,
        [
            col_selector_code,
        ],
    ] = remove_code
    data.loc[
        start_index:end_index,
        [
            col_selector_msg,
        ],
    ] = remove_msg
    new_na_group_index = randint(1e3, 1e8)
    data.loc[
        start_index:end_index,
        [
            col_selector_na_groups,
        ],
    ] = new_na_group_index

    datasets[lysi_number] = data

    save_days(data, start_index, end_index, data.columns[0], lysi_number)
    st.session_state["datasets"] = datasets
    st.rerun()


def gap_fill_edge(fill_date_selector, lysi_number):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]

    write_index = data.index[data[data.columns[0]] == fill_date_selector][0]
    data.loc[
        write_index,
        [
            col_selector_code,
            col_selector_msg,
            col_selector_groups,
            col_selector_na_groups,
        ],
    ] = np.nan

    data.loc[
        write_index,
        [col_selector],
    ] = fill_value

    output_file = data.loc[write_index, data.columns[0]]
    datasets[lysi_number] = data

    save_day(data, output_file, data.columns[0], lysi_number)
    st.session_state["datasets"] = datasets
    st.rerun()


def gap_fill_no_other(fill_date_selector, lysi_number, fill_value):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]

    start_date = datetime.datetime.strptime(
        fill_date_selector.split("to")[0].strip(),
        "%Y-%m-%d %H:%M:%S",
    )
    end_date = datetime.datetime.strptime(
        fill_date_selector.split("to")[-1].strip(),
        "%Y-%m-%d %H:%M:%S",
    )

    start_index = data.index[data[data.columns[0]] == start_date][0]
    end_index = data.index[data[data.columns[0]] == end_date][0]

    data.loc[
        start_index:end_index,
        [
            col_selector_code,
            col_selector_msg,
            col_selector_na_groups,
            col_selector_groups,
        ],
    ] = np.nan

    data.loc[
        start_index:end_index,
        [col_selector],
    ] = fill_value

    datasets[lysi_number] = data

    save_days(data, start_index, end_index, data.columns[0], lysi_number)

    st.session_state["datasets"] = datasets
    st.rerun()


def gap_fill_interpolate_all(fill_date_selector, lysi_number, col):
    col_selector_code = f"fill_{col.lower()}_code"
    col_selector_msg = f"fill_{col.lower()}_msg"
    col_selector_groups = f"fill_{col.lower()}_groups"
    col_selector_na_groups = f"fill_{col.lower()}_na_groups"
    col_selector = col

    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]
    st.session_state["previous_dataset"] = data.copy()

    start_date = fill_date_selector[0]
    end_date = fill_date_selector[1]
    start_index = data.index[data[data.columns[0]] >= start_date].tolist()[0]
    end_index = data.index[data[data.columns[0]] <= end_date].tolist()[-1]

    data.loc[
        start_index:end_index,
        [
            col_selector_msg,
        ],
    ] = np.where(
        data.loc[start_index:end_index, col_selector_code].isin([-1, -2]),
        np.nan,
        data.loc[start_index:end_index, col_selector_msg],
    )

    data.loc[
        start_index:end_index,
        [
            col_selector_na_groups,
        ],
    ] = np.where(
        data.loc[start_index:end_index, col_selector_code].isin([-1, -2]),
        np.nan,
        data.loc[start_index:end_index, col_selector_na_groups],
    )

    data.loc[
        start_index:end_index,
        [
            col_selector_groups,
        ],
    ] = np.where(
        data.loc[start_index:end_index, col_selector_code].isin([-1, -2]),
        np.nan,
        data.loc[start_index:end_index, col_selector_groups],
    )

    data.loc[
        start_index:end_index,
        [
            col_selector_code,
        ],
    ] = np.where(
        data.loc[start_index:end_index, col_selector_code].isin([-1, -2]),
        np.nan,
        data.loc[start_index:end_index, col_selector_code],
    )

    fill_interpolated = (
        data.loc[start_index:end_index, col_selector]
        .interpolate(method="linear")
        .tolist()
    )

    data.loc[
        start_index:end_index,
        [col_selector],
    ] = fill_interpolated

    datasets[lysi_number] = data

    save_days(data, start_index, end_index, data.columns[0], lysi_number)

    st.session_state["datasets"] = datasets
    st.rerun()


def post_thresh(
    data_thresh,
    thresh_start_index,
    thresh_end_index,
    col_selector_post,
    lysi_number,
):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]
    st.session_state["previous_dataset"] = data.copy()
    data.loc[thresh_start_index:thresh_end_index, col_selector_post] = data_thresh[
        col_selector_post
    ]
    datasets[lysi_number] = data

    save_days(data, thresh_start_index, thresh_end_index, data.columns[0], lysi_number)
    st.session_state["datasets"] = datasets


def post_remove(remove_start_index, remove_end_index, col_selector_post, lysi_number):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]
    st.session_state["previous_dataset"] = data.copy()
    data.loc[remove_start_index:remove_end_index, col_selector_post] = np.nan
    datasets[lysi_number] = data

    save_days(data, remove_start_index, remove_end_index, data.columns[0], lysi_number)
    st.session_state["datasets"] = datasets


def post_fill(
    fill_start_index, fill_end_index, col_selector_post, lysi_number, fill_values_post
):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]
    st.session_state["previous_dataset"] = data.copy()
    data.loc[fill_start_index:fill_end_index, col_selector_post] = fill_values_post
    datasets[lysi_number] = data

    save_days(data, fill_start_index, fill_end_index, data.columns[0], lysi_number)

    st.session_state["datasets"] = datasets


def post_fill_winter(pluvio_df, param_selector_post, lysi_number):
    datasets = st.session_state["datasets"]
    data = datasets[lysi_number]
    st.session_state["previous_dataset"] = data.copy()

    data[data.columns[0]] = [
        datetime.datetime.strptime(i + " 00:00:00", "%Y-%m-%d %H:%M:%S")
        for i in data[data.columns[0]].dt.strftime("%Y-%m-%d").tolist()
    ]

    fill_indexes = data.index[
        data[data.columns[0]].isin(pluvio_df[pluvio_df.columns[0]])
    ]

    if param_selector_post == "Precipitation":
        data.loc[fill_indexes, col_selector_post] = pluvio_df["precipitation_filled"]
    else:
        data.loc[fill_indexes, col_selector_post] = pluvio_df["eto"]

    datasets[lysi_number] = data

    save_days(
        data,
        date_col=data.columns[0],
        start_index=None,
        end_index=None,
        lysi_number=lysi_number,
        fill_indexes=list(fill_indexes),
    )
    st.session_state["datasets"] = datasets


# Main functions


def fetch_ec(data, location):
    start_date = f"{data.iloc[0,0].strftime('%Y-%m-%d')}T00:00:00Z"
    end_date = f"{data.iloc[data.shape[0]-1,0].strftime('%Y-%m-%d')}T00:00:00Z"

    abbr = location_to_abbr(location)

    client = InfluxDBClientOld(
        host="172.27.80.119", port=8086, database="TerenoPreAlpine"
    )
    pluvio_query = f"""SELECT SUM("P_M_EZ_NEZ") FROM "autogen"."raw" WHERE "Location" = '{abbr}' AND "Device" = 'pluvio' AND time >= '{start_date}' AND time <= '{end_date}' GROUP BY time(1d) """
    ec_query = f"""SELECT MEAN("windspeed2m") AS wind_mean,MIN("temperature2m") AS temp_min,MAX("temperature2m") AS temp_max,MEAN("temperature2m") AS temp_mean, MIN("relhum2m") AS hum_min,MAX("relhum2m") AS hum_max,MEAN("relhum2m") AS hum_mean FROM "autogen"."raw" WHERE "Location" = '{abbr}' AND time >= '{start_date}' AND time <= '{end_date}' GROUP BY time(1d) """

    pluvio_result = client.query(pluvio_query)

    ec_result = client.query(ec_query)

    pluvio_time_lst = []
    pluvio_value_lst = []

    for point in pluvio_result.get_points():
        pluvio_value_lst.append(point["sum"])
        pluvio_time_lst.append(point["time"])

    ec_time_lst = []
    ec_temp_min_lst = []
    ec_temp_max_lst = []
    ec_temp_mean_lst = []
    ec_hum_min_lst = []
    ec_hum_max_lst = []
    ec_hum_mean_lst = []
    ec_wind_mean_lst = []

    for point in ec_result.get_points():
        ec_temp_min_lst.append(point["temp_min"])
        ec_temp_max_lst.append(point["temp_max"])
        ec_temp_mean_lst.append(point["temp_mean"])
        ec_hum_min_lst.append(point["hum_min"])
        ec_hum_max_lst.append(point["hum_max"])
        ec_hum_mean_lst.append(point["hum_mean"])
        ec_wind_mean_lst.append(point["wind_mean"])
        ec_time_lst.append(point["time"])

    client.close()

    df_prec = pd.DataFrame(
        {"Timestamp": pluvio_time_lst, "precipitation": pluvio_value_lst}
    )
    df_prec.iloc[:, 0] = pd.to_datetime(
        [
            re.search("(\\d{4}-\\d{2}-\\d{2})T.*", i).group(1) + " 00:00:00"
            for i in df_prec.iloc[:, 0].tolist()
        ],
        format="%Y-%m-%d %H:%M:%S",
    )

    df_ec = pd.DataFrame(
        {
            "Timestamp": ec_time_lst,
            "temp_min": ec_temp_min_lst,
            "temp_max": ec_temp_max_lst,
            "temp_mean": ec_temp_mean_lst,
            "hum_min": ec_hum_min_lst,
            "hum_max": ec_hum_max_lst,
            "hum_mean": ec_hum_mean_lst,
            "wind_mean": ec_wind_mean_lst,
        }
    )
    df_ec.iloc[:, 0] = pd.to_datetime(
        [
            re.search("(\\d{4}-\\d{2}-\\d{2})T.*", i).group(1) + " 00:00:00"
            for i in df_ec.iloc[:, 0].tolist()
        ],
        format="%Y-%m-%d %H:%M:%S",
    )
    df_ec.dropna(inplace=True)

    station = Station(latitude=47.829659, altitude=595)
    station.anemometer_height = 2

    et_list = []

    for index in range(df_ec.shape[0]):
        doy = df_ec.iloc[index, 0].day_of_year

        entry = station.day_entry(
            doy,
            temp_min=df_ec.iloc[index, 1],
            temp_max=df_ec.iloc[index, 2],
        )
        eto = entry.eto()
        et_list.append(eto)

    df_et = pd.DataFrame({"Timestamp": df_ec.iloc[:, 0].tolist(), "eto": et_list})

    date_range = pd.date_range(
        df_prec.iloc[0, 0], df_prec.iloc[df_prec.shape[0] - 1, 0], freq="D"
    )
    date_df = pd.DataFrame({"Timestamp": date_range})

    result_df = date_df.merge(df_et, on="Timestamp", how="left").merge(
        df_prec, on="Timestamp", how="left"
    )

    st.session_state["ec_data"] = result_df


# Correct water release from water tank that is released at weight of 47.5kg and build cumulative sum
# According to AWAS_TO_XLSX.R and xlsx2AWAT.m script
def correct_water_release(water_weight):
    """
    water_weight: [pd.Series] of [numeric] water tank weight

    return      : [pd.Series] of [numeric] cumulative sum of water tank weight
    """
    # According to AWAT_TO_XLSX.R
    ################################################################################################################################
    # Detect water release events (weight > 47.49kg)
    # water_release_indices = list(water_weight.index[water_weight > 47.49])
    # water_release_start_indices = []

    # # Trigger the last value to be also the start of water release by making the conditions true in the following loop for the last value
    # if len(water_release_indices) > 0:
    #     water_release_indices.append(0)

    # # Loop over indices and check where the last index of 47.5 kg is in water tank weight for each release event and add it to a list
    # # --> thats the point where water release starts
    # for release_index in range(len(water_release_indices) - 1):
    #     if (
    #         water_release_indices[release_index] + 1
    #         != water_release_indices[release_index + 1]
    #     ):
    #         water_release_start_indices.append(water_release_indices[release_index])

    # # Calculate the changing rate of water and set the first value to 0 since first value is always NA
    # water_release_delta = water_weight.diff()
    # water_release_delta.iloc[0] = 0

    # # Correct water release by setting the changing rate of the 10 minutes period (that the water release needs) to 0
    # water_release_corrected = water_release_delta.copy()
    # for group_index in water_release_start_indices:
    #     water_release_corrected.iloc[group_index : (group_index + 11)] = 0

    # Idea from xlsx2AWAT.m
    ################################################################################################################################
    # With previous application water release events stay unchanged that stay < 47.49kg but in the next time step already have water release
    # since the time resolution is too low to detect the overextension of the threshold

    water_release_delta = water_weight.diff()
    water_release_delta.iloc[0] = 0
    water_release_corrected = water_release_delta.copy()

    # Idea is to look at moving window of 30 minutes and look whether there are negative changing rates between -4 and -13
    # (according to xlsx2AWAT.m)

    # Look at each possibe 30min moving window of changing rates and build sum to find out which is the moving window with lowest sum
    extra_release_periods = np.sum(
        sliding_window_view(water_release_corrected.to_numpy(), 30), axis=1
    )
    lower_thresh = extra_release_periods <= -4
    upper_thresh = extra_release_periods >= -13
    cond_thresh = np.logical_and(lower_thresh, upper_thresh)
    cond_indices = np.where(cond_thresh)[0]
    # build differences of indices for moving window surpassing the threshold to build groups
    # --> only find smallest sum within group of moving windows, in order to be able to detect that within several water release events
    # groups have to be build
    cond_indices_diff = [1] + list(np.diff(cond_indices))

    # If this condition of water release is not met
    if len(cond_indices_diff) == 1:
        cond_indices_diff = []

    # Create groups
    period_groups = {}
    group_index = 0
    for counter, index in enumerate(cond_indices_diff):
        if index == 1:
            if group_index in period_groups.keys():
                period_groups[group_index].append(counter)
            else:
                period_groups[group_index] = [counter]
        else:
            group_index += 1
            period_groups[group_index] = [counter]

    # See which moving window has the lowest sum of changing rates --> strong water release
    # period of smallest moving window, set changing rate to 0
    for index, group in period_groups.items():
        max_period = cond_indices[group][
            np.argmin(
                np.sum(
                    sliding_window_view(water_release_corrected.to_numpy(), 30)[
                        cond_indices[group], :
                    ],
                    axis=1,
                )
            )
        ]
        if max_period == 0 or (max_period + 30) == water_release_corrected.shape[0]:
            water_release_corrected.loc[(max_period) : (max_period + 29)] = 0
        else:
            water_release_corrected.loc[(max_period) : (max_period + 29)] = np.nan
            water_release_corrected.loc[
                (max_period - 3) : (max_period + 32)
            ].interpolate(inplace=True)
            if any(water_weight.loc[(max_period) : (max_period + 29)].isna()):
                water_release_delta.loc[(max_period) : (max_period + 29)] = 0
        # water_release_corrected.loc[(max_period) : (max_period + 12)] = 0

    # Correct additional outliers showing strong negative changing rate (according to AWAS_TO_XLSX.R)
    # water_release_corrected = np.where(
    #     water_release_corrected <= (-1), 0, water_release_corrected
    # )

    # Create cumulative sum
    water_weight_corrected = water_release_corrected.cumsum()
    return water_weight_corrected


# Get Date object from filename date
def format_file_date(filename):
    """
    filename : [scalar] of [string] filename

    return   : [scalar] of [date] file date
    """
    filename_date = re.search("^\\d{1}_(.*)", filename).group(1)
    filename_date_comp = re.search("(\\d{4})_(\\d{2})_(\\d{2})", filename_date)
    date = datetime.datetime.strptime(
        f"{filename_date_comp.group(1)}-{filename_date_comp.group(2)}-{filename_date_comp.group(3)}",
        "%Y-%m-%d",
    )
    return date.date()


# Read data from specified read path (by user input in sidebar)
def read_data(path, relevant_files, type_selector):
    """
    path           : [scalar] [string] path for folder where files are contained
    relevant_files : [list] [strings] files in read folder that are within date range given by user
    type_selector  : [scalar] [string] data type given by user

    return         : [list] [pd.DataFrames] data for lysimeter 1-6 for given dates
    """
    with st.spinner(text="Loading data..."):
        # Initliaze list for containing the data
        datasets = []
        # Loop over lysimeters
        for lysimeter_number in range(0, 6):
            # List files for this lysimeter
            lysimeter_files = [
                file
                for file in relevant_files
                if re.match(f"^{lysimeter_number+1}_.*", file)
            ]
            # Only when files are existent
            if len(lysimeter_files) > 0:
                lysimeter_data = []

                # Sort existent files after date
                ##########################################################################

                # Sorting files after date
                file_order = []

                # Loop through files and append date for each file from filename
                for file_path in lysimeter_files:
                    if file_path != ".DS_Store":
                        file_date_groups = re.search(
                            "^\\d+_(\\d{4}_\\d{2}_\\d{2}).*", file_path
                        )
                        file_date = datetime.datetime.strptime(
                            file_date_groups.group(1), "%Y_%m_%d"
                        )
                        file_order.append(file_date)
                file_order.sort()

                # Sort filenames by extracted dates in previous step
                lysimeter_filenames = []
                for file_path in file_order:
                    date_str = datetime.datetime.strftime(file_path, "%Y_%m_%d")
                    file_name = [
                        i for i in lysimeter_files if re.match(f".*{date_str}.*", i)
                    ][0]
                    lysimeter_filenames.append(file_name)

                # Read data
                ##########################################################################

                # Loop over sorted files and append to list
                for file in lysimeter_filenames:
                    if not file == ".DS_Store":
                        file_path = os.path.join(path, file)
                        file_data = pd.read_csv(file_path)
                        file_data[file_data.columns[0]] = pd.to_datetime(
                            file_data.iloc[:, 0]
                        )
                        lysimeter_data.append(file_data)

                # Merge all files for this lysimeter number
                merged_data = pd.concat(lysimeter_data)
                merged_data.sort_values(by=[merged_data.columns[0]], inplace=True)

                # Start date range for this lysimeter
                start_date = merged_data.iloc[0, 0]
                # End date range for this lysimeter
                end_date = merged_data.iloc[merged_data.shape[0] - 1, 0]
                # Pandas date range for this lysimeter
                if data_selector == "additional":
                    date_range = pd.date_range(start_date, end_date, freq="10min")
                else:
                    date_range = pd.date_range(start_date, end_date, freq="min")
                # Create empty dataframe with date range as index
                empty_df = pd.DataFrame({"Timestamp": date_range})
                # Merge empty dataframe with data for this lysimeter
                merged_data = empty_df.merge(merged_data, how="left", on="Timestamp")
                merged_data.reset_index(drop=True, inplace=True)

                # If data type is 'balance' and 'gap_check' is the action type
                if re.match(".*balance.*", path) and type_selector == "gap_check":
                    # Correct water release in i schedule
                    water_weight = merged_data[f"L_{lysimeter_number+1}_WAG_D_000"]
                    cumsum_weights = correct_water_release(water_weight)
                    # Add corrected and cumulated sum of water tank as new column
                    merged_data.loc[
                        :, f"L_{lysimeter_number+1}_WAG_D_000_cumsum"
                    ] = cumsum_weights
                    # Create empty flag columns for gap check so that no error is thrown
                    merged_data.loc[
                        :, f"fill_l_{lysimeter_number+1}_wag_d_000_cumsum_code"
                    ] = None
                    merged_data.loc[
                        :, f"fill_l_{lysimeter_number+1}_wag_d_000_cumsum_msg"
                    ] = None
                    merged_data.loc[
                        :, f"fill_l_{lysimeter_number+1}_wag_d_000_cumsum_groups"
                    ] = None
                    merged_data.loc[
                        :, f"fill_l_{lysimeter_number+1}_wag_d_000_cumsum_na_groups"
                    ] = None

                # Append dataset for specifi lysimeter to list for all lysimeter data
                datasets.append(merged_data)
        return datasets


# Reset session states
def reset_data():
    st.session_state["datasets"] = None
    st.session_state["read_path"] = None
    st.session_state["ec_data"] = None
    st.session_state["previous_dataset"] = None


def reset_previous_dataset():
    st.session_state["previous_dataset"] = None


with st.sidebar:
    # Display LysiCheck image
    st.image(image)

    # Selector for action type
    type_selector = st.selectbox(
        "Select type of operation", types, index=0, on_change=reset_data
    )

    # Selector for hexagon and location
    location_selector = st.selectbox(
        "Select lysimeter location", locations, index=0, on_change=reset_data
    )
    location_summary = location_selector  # re.sub("\\d", "", location_selector)

    # Selector for data type
    data_selector = st.selectbox("Select data type", data_type, on_change=reset_data)

    folder_date_lower = datetime.date(2010, 1, 1)
    folder_date_upper = datetime.date.today()
    folder_date_default = (datetime.date(2022, 1, 1), folder_date_upper)

    # Date range selector
    file_selector = st.date_input(
        "Select date range",
        folder_date_default,
        folder_date_lower,
        folder_date_upper,
        on_change=reset_data,
    )

    # Change read path based on action path
    if type_selector == "gap_check":
        read_path = os.path.join(src_path, f"{data_selector}_filled", location_selector)
    else:
        read_path = os.path.join(src_path, f"{data_selector}_post", location_selector)

    # Check for relevant files in read path and whether there are location folders
    try:
        relevant_files = [
            file for file in os.listdir(read_path) if not file == ".DS_Store"
        ]
    except Exception as e:
        print(e)
        st.write(
            f":red[No {type_selector} location folder for {data_selector} files available]"
        )

    # Check whether single date or date range is given (length of 2)
    if len(file_selector) == 2:
        # Check whether there are files for the given date range
        try:
            # When there are files read them and save read path and available files as session states
            file_available = [
                file
                for file in relevant_files
                if format_file_date(file) >= file_selector[0]
                and format_file_date(file) <= file_selector[1]
            ]

            if len(file_available) == 0:
                st.write(":red[No data available for this date range]")

            if st.button("Apply", type="primary", disabled=(len(file_available) == 0)):
                datasets = read_data(read_path, file_available, type_selector)
                st.session_state["datasets"] = datasets
                st.session_state["read_path"] = read_path
        except Exception as e:
            print(e)
            st.write(
                f":red[No {type_selector} {data_selector} files available for any date]"
            )
            st.button("Apply", type="primary", disabled=True, key="btn_error_1")
    else:
        st.write(":red[Select two dates]")

if "datasets" in st.session_state and st.session_state.datasets != None:
    datasets = st.session_state["datasets"]
else:
    datasets = []


# Catch any errors
try:
    # Check if data has been read
    if len(datasets) > 0:
        # Gap Check
        ################################################################################################
        if type_selector == "gap_check":
            st.header(f"Gap Check", divider="red")

            st.write("Select lysimeter")

            lysimeter_selector_col, lysimeter_finish_col = st.columns([0.4, 0.6])

            # Select lysimeter number
            with lysimeter_selector_col:
                lysimeter_selector = st.selectbox(
                    "Select lysimeter",
                    [i + 1 for i in range(6)],
                    key="select_lysimeter",
                    label_visibility="collapsed",
                )

            # When finishing a file (=gap checking completed) the file is separated into different dates and written to new folders
            with lysimeter_finish_col:
                st.info(
                    "Clicking the button passes all loaded lysimeter data to the next workflow step."
                )
                if data_selector == "balance":
                    st.warning(
                        f"The AWAT-Filter will be applied to the data in the next step. In order for this to work, a buffer day before and after the days of interest is needed that will be cut off afterwards: :red[**{datasets[0].iloc[0,0].strftime('%Y-%m-%d')}**] and :red[**{datasets[0].iloc[datasets[0].shape[0]-1,0].strftime('%Y-%m-%d')}**] will be removed."
                    )
                confirm = st.checkbox("I understand")
                if st.button("Finish File", type="primary", disabled=(not confirm)):
                    # Loop through each dataset for each lysimeter
                    for index, dataset in enumerate(datasets):
                        # If additional data write daily files in folder for postprocessing step
                        if data_selector == "additional":
                            unique_dates = [
                                (date.strftime("%Y_%m_%d"), date.strftime("%Y-%m-%d"))
                                for date in pd.to_datetime(dataset[dataset.columns[0]])
                                .dt.date.unique()
                                .tolist()
                            ]
                            for filename, date in unique_dates:
                                start_date = datetime.datetime.strptime(
                                    f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S"
                                )
                                end_date = datetime.datetime.strptime(
                                    f"{date} 23:59:59", "%Y-%m-%d %H:%M:%S"
                                )
                                day_dataset = dataset[
                                    (dataset[dataset.columns[0]] >= start_date)
                                    & (dataset[dataset.columns[0]] <= end_date)
                                ]

                                st.session_state["read_path"].replace(
                                    f"{data_selector}_filled", f"{data_selector}_post"
                                )
                                pathlib.Path(
                                    st.session_state["read_path"].replace(
                                        f"{data_selector}_filled",
                                        f"{data_selector}_post",
                                    )
                                ).mkdir(parents=True, exist_ok=True)
                                day_dataset.to_csv(
                                    os.path.join(
                                        st.session_state["read_path"].replace(
                                            f"{data_selector}_filled",
                                            f"{data_selector}_post",
                                        ),
                                        f"{index+1}_{filename}.csv",
                                    ),
                                    index=False,
                                )
                        # If balance data AWAT filter needs to be executed next
                        # put merged data in folder for AWAT
                        if data_selector == "balance":
                            dataset_truncated_cols = [
                                col
                                for col in dataset.columns
                                if not re.match(".*fill.*", col)
                            ]
                            export_rawdata = dataset[dataset_truncated_cols]
                            start_time = datetime.datetime.strptime(
                                "1899-12-30 00:00:00", "%Y-%m-%d %H:%M:%S"
                            )
                            end_time = pd.to_datetime(
                                export_rawdata.iloc[:, 0]
                            ).to_list()
                            time_diff = [
                                ((i - start_time).total_seconds() / (60 * 60 * 24))
                                for i in end_time
                            ]
                            export_rawdata.iloc[:, 0] = time_diff

                            export_rawdata.iloc[:, 0] = export_rawdata.iloc[:, 0].map(
                                lambda x: "{0:.8f}".format(x)
                            )

                            export_rawdata.iloc[:, 2] = export_rawdata.iloc[:, 2].map(
                                lambda x: "{0:.8f}".format(x)
                            )
                            export_rawdata.iloc[:, 3] = export_rawdata.iloc[:, 3].map(
                                lambda x: "{0:.8f}".format(x)
                            )
                            pathlib.Path(os.path.join(src_path, "awat")).mkdir(
                                parents=True, exist_ok=True
                            )
                            export_rawdata.drop(
                                columns=[f"L_{index+1}_WAG_D_000"]
                            ).to_csv(
                                os.path.join(
                                    src_path,
                                    "awat",
                                    f"{location_selector}_rawdata{index+1}_{export_rawdata.iloc[0, 0]}.dat",
                                ),
                                header=False,
                                sep="\t",
                                index=False,
                            )
                    reset_data()
                    st.rerun()

            # Display lysimeter plots for gap check
            ################################################################################################

            gap_lysi_number = lysimeter_selector - 1
            data = datasets[gap_lysi_number]

            st.subheader(f"Lysimeter {lysimeter_selector}", divider="gray")
            show_1 = show_2 = show_3 = show_4 = show_4_interpolation = show_5 = True

            # Catch if data for a specific lysimeter is missing
            try:
                col_selector_col, _ = st.columns([0.4, 0.6])

                # Select which parameter should be visualized
                with col_selector_col:
                    col_selector = st.selectbox(
                        "Select column",
                        [
                            col
                            for col in data.columns[1:]
                            if not re.match("^fill_.*", col)
                        ],
                        key=lysimeter_selector,
                    )

                # Clean flagged data so that each sequential group of missing data can be displayed as rectangle and single missing gaps as lines
                # as well as single filled values

                col_selector_code = f"fill_{col_selector.lower()}_code"
                col_selector_msg = f"fill_{col_selector.lower()}_msg"
                col_selector_groups = f"fill_{col_selector.lower()}_groups"
                col_selector_na_groups = f"fill_{col_selector.lower()}_na_groups"

                col_selector_filled = data[data[col_selector_code].isin([1, 2, -1, -2])]
                col_selector_filled.reset_index(drop=True, inplace=True)

                multi_gaps = data[
                    (data[col_selector_groups].notnull())
                    | (data[col_selector_na_groups].notnull())
                ]

                multi_gaps["date"] = multi_gaps["Timestamp"].dt.date
                multi_gaps_grouped_filled = multi_gaps.groupby(
                    [col_selector_groups, "date"]
                )
                multi_gaps_grouped_na = multi_gaps.groupby(
                    [col_selector_na_groups, "date"]
                )

                # Downsample data except for days where gaps are

                data = downsample_gapdata_hourly(data, param_code=col_selector_code)

                render_expandables_gap = st.checkbox(
                    "Render expandable section for all lysimeter plots"
                )
                st.info(
                    "Enabling rendering will result in loading of this section every time user input is updated. This can consume a lot of extra time."
                )
                if render_expandables_gap:
                    # Expandable section that shows the plot of the selected parameter for all lysimeters
                    with st.expander("All lysimeter plots"):
                        st.info(
                            "The partner lysimeter is shown according to the 'lysimeter_matches.json' file. If you want to change the partner lysimeter, change it at the expandable section 'Partner Lysimeter Configuration' in the postprocessing. But be aware that the automatic gap filling was also based on the default lysimeter matches!"
                        )
                        subtitles = []
                        for i in range(6):
                            if (i + 1) == lysimeter_selector:
                                subtitles.append(f"Lysimeter {i+1} (selected)")
                            elif (i + 1) == lysimeter_matches[location_summary][
                                str(lysimeter_selector)
                            ]:
                                subtitles.append(f"Lysimeter {i+1} (partner)")
                            else:
                                subtitles.append(f"Lysimeter {i+1}")

                        fig_all = make_subplots(
                            rows=3,
                            cols=2,
                            subplot_titles=subtitles,
                        )

                        for index, dataset in enumerate(datasets):
                            index_selector = re.sub(
                                "_\\d{1}_", f"_{index+1}_", col_selector
                            )

                            index_selector_code = re.sub(
                                "_\\d{1}_", f"_{index+1}_", col_selector_code
                            )

                            data_downsampled = downsample_gapdata_hourly(
                                dataset, param_code=index_selector_code
                            )

                            rows = [1, 1, 2, 2, 3, 3]
                            cols = [1, 2, 1, 2, 1, 2]
                            fig_all.add_trace(
                                go.Scattergl(
                                    x=data_downsampled[data_downsampled.columns[0]],
                                    y=data_downsampled[index_selector],
                                    mode="lines+markers",
                                    showlegend=False,
                                    marker=dict(size=2, color="#4574ba"),
                                ),
                                row=rows[index],
                                col=cols[index],
                            )
                        fig_all.update_layout(
                            height=800, width=800, title_text=f"{col_selector}"
                        )
                        st.plotly_chart(fig_all, use_container_width=True)

                # Adapt ylims since the lines showing missing single gaps or filled gaps have ylim of -Inf to Inf
                if np.isnan(data[col_selector].min()):
                    ylim = [-10, 10]
                else:
                    margin = (
                        abs((data[col_selector].max() - data[col_selector].min()))
                        * 0.15
                    )
                    ylim = [
                        data[col_selector].min() - margin,
                        data[col_selector].max() + margin,
                    ]

                # Adapt ylim when one boundary is 0 at least
                if 0 in ylim:
                    change_index = ylim.index(0)
                    if change_index == 0:
                        ylim[change_index] = -1 * margin
                    else:
                        ylim[change_index] = margin

                # Create main plot showing the selected parameter
                fig = px.line(
                    data,
                    x=data.columns[0],
                    y=col_selector,
                    color_discrete_sequence=["#4574ba"],
                    title=f"{col_selector} of Lysimeter {lysimeter_selector}",
                    markers=True,
                    render_mode="webgl",
                )
                fig.update_traces(marker={"size": 2})

                # Show multi gaps filled
                for name, group in multi_gaps_grouped_filled:
                    if group.loc[group.index[0], col_selector_code] == 2:
                        fig.add_vrect(
                            x0=group.iloc[0, 0],
                            x1=group.iloc[(group.shape[0] - 1), 0],
                            fillcolor="#3abd68",
                            opacity=0.25,
                            line_width=0,
                        )

                        for x in [
                            group.iloc[0, 0],
                            group.iloc[(group.shape[0] - 1), 0],
                        ]:
                            fig.add_trace(
                                go.Scatter(
                                    x=[
                                        x,
                                        x,
                                    ],
                                    y=ylim,
                                    mode="lines",
                                    line=dict(color="#3abd68"),
                                    legendgroup="2",
                                    name="filled (lm)",
                                    showlegend=show_4,
                                    opacity=0.5,
                                )
                            )

                            if show_4:
                                show_4 = False

                    else:
                        fig.add_vrect(
                            x0=group.iloc[0, 0],
                            x1=group.iloc[(group.shape[0] - 1), 0],
                            fillcolor="#a344ad",
                            opacity=0.25,
                            line_width=0,
                        )

                        for x in [
                            group.iloc[0, 0],
                            group.iloc[(group.shape[0] - 1), 0],
                        ]:
                            fig.add_trace(
                                go.Scatter(
                                    x=[
                                        x,
                                        x,
                                    ],
                                    y=ylim,
                                    mode="lines",
                                    line=dict(color="#a344ad"),
                                    legendgroup="-1.5",
                                    name="filled (interpolation multi)",
                                    showlegend=show_4_interpolation,
                                    opacity=0.5,
                                )
                            )

                            if show_4_interpolation:
                                show_4_interpolation = False

                # Show multi gaps with remaining NAS
                for name, group in multi_gaps_grouped_na:
                    fig.add_vrect(
                        x0=group.iloc[0, 0],
                        x1=group.iloc[(group.shape[0] - 1), 0],
                        fillcolor="#db6363",
                        opacity=0.25,
                        line_width=0,
                    )
                    for x in [
                        group.iloc[0, 0],
                        group.iloc[(group.shape[0] - 1), 0],
                    ]:
                        fig.add_trace(
                            go.Scatter(
                                x=[
                                    x,
                                    x,
                                ],
                                y=ylim,
                                mode="lines",
                                legendgroup="-2",
                                line=dict(color="#db6363"),
                                name="NA (no other value)",
                                showlegend=show_5,
                                opacity=0.5,
                            )
                        )

                        if show_5:
                            show_5 = False

                # Show single gaps or filled values
                for index in range(col_selector_filled.shape[0]):
                    if col_selector_filled.loc[index, col_selector_code] == -1:
                        fig.add_trace(
                            go.Scatter(
                                x=[
                                    col_selector_filled.iloc[index, 0],
                                    col_selector_filled.iloc[index, 0],
                                ],
                                y=ylim,
                                mode="lines",
                                legendgroup="-1",
                                line=dict(color="#dba053"),
                                name="NA (edge value)",
                                showlegend=show_1,
                                opacity=0.5,
                            )
                        )

                        if show_1:
                            show_1 = False
                    elif (
                        col_selector_filled.loc[index, col_selector_code] == 1
                        and col_selector_filled.loc[index, col_selector_groups] == None
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=[
                                    col_selector_filled.iloc[index, 0],
                                    col_selector_filled.iloc[index, 0],
                                ],
                                y=ylim,
                                mode="lines",
                                line=dict(color="#a344ad"),
                                name="filled (interpolation single)",
                                showlegend=show_3,
                                legendgroup="1",
                                opacity=0.5,
                            )
                        )

                        if show_3:
                            show_3 = False

                fig.update_layout(yaxis=dict(range=[ylim[0], ylim[1]]))

                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)

                # Fill remaining NAs for gap check
                ################################################################################################

                with col1:
                    st.subheader("Fill remaining NAs", divider="red")

                    # Type of single gaps
                    na_type_selector = st.selectbox(
                        "Select NA type",
                        [
                            "NA (edge value)",
                            "NA (no other value)",
                        ],
                        key="select_na",
                    )

                    # Method options for filling the gap based on the selected type of gap
                    if na_type_selector == "NA (edge value)":
                        na_type_condition = "edge value"
                        fill_date_selector_options = col_selector_filled[
                            (col_selector_filled[col_selector_msg] == na_type_condition)
                        ][data.columns[0]]

                    else:
                        fill_date_selector_options = sorted(
                            [
                                f"{group.iloc[0,0]} to {group.iloc[(group.shape[0]-1),0]}"
                                for name, group in multi_gaps_grouped_na
                            ],
                            key=lambda x: datetime.datetime.strptime(
                                x.split("to")[0].strip(), "%Y-%m-%d %H:%M:%S"
                            ),
                        )

                    # Selector for gap date
                    fill_date_selector = st.selectbox(
                        "Select gap",
                        fill_date_selector_options,
                        key="select_date_na",
                    )

                    fill_method_selector_options = {
                        "NA (edge value)": [
                            "single value",
                            "linear interpolation",
                        ],
                        "NA (no other value)": ["single value", "linear interpolation"],
                    }

                    # Selector for fill method
                    fill_method_selector = st.selectbox(
                        "Select fill method",
                        fill_method_selector_options[na_type_selector],
                        key="select_method_na",
                    )

                    # Different fill methods for gap filling for gap check
                    ################################################################################################
                    if (
                        fill_method_selector == "single value"
                        and fill_date_selector != None
                    ):
                        # Single value fill
                        fill_value = st.number_input("Enter value", step=1.0)

                    elif (
                        fill_method_selector == "linear interpolation"
                        and fill_date_selector != None
                    ):
                        # Linear interpolation fill

                        # Data for selected lysimeter
                        current_lysimeter_df = data

                        # If gap range interpolate data from beginning to end with linear interpolation of previous and value afterwards
                        if na_type_selector == "NA (no other value)":
                            start_date = datetime.datetime.strptime(
                                fill_date_selector.split("to")[0].strip(),
                                "%Y-%m-%d %H:%M:%S",
                            )
                            end_date = datetime.datetime.strptime(
                                fill_date_selector.split("to")[-1].strip(),
                                "%Y-%m-%d %H:%M:%S",
                            )

                            start_index = current_lysimeter_df.index[
                                current_lysimeter_df[current_lysimeter_df.columns[0]]
                                == start_date
                            ][0]
                            end_index = current_lysimeter_df.index[
                                current_lysimeter_df[current_lysimeter_df.columns[0]]
                                == end_date
                            ][0]

                            gap_start_time = datetime.datetime.strftime(
                                current_lysimeter_df.loc[start_index, "Timestamp"],
                                "%H:%M:%S",
                            )
                            gap_end_time = datetime.datetime.strftime(
                                current_lysimeter_df.loc[end_index, "Timestamp"],
                                "%H:%M:%S",
                            )

                            if (
                                start_index == 0
                                or end_index == current_lysimeter_df.shape[0]
                            ):
                                fill_value = np.nan
                                st.error("There are no values before or after the gap.")
                            elif (
                                gap_start_time == "00:00:00"
                                or gap_end_time == "23:59:59"
                            ):
                                start_index = datasets[gap_lysi_number].index[
                                    datasets[gap_lysi_number][
                                        datasets[gap_lysi_number].columns[0]
                                    ]
                                    == start_date
                                ][0]
                                end_index = datasets[gap_lysi_number].index[
                                    datasets[gap_lysi_number][
                                        datasets[gap_lysi_number].columns[0]
                                    ]
                                    == end_date
                                ][0]

                                if np.isnan(
                                    datasets[gap_lysi_number].loc[
                                        (start_index - 1), col_selector
                                    ]
                                ) or np.isnan(
                                    datasets[gap_lysi_number].loc[
                                        (end_index + 1), col_selector
                                    ]
                                ):
                                    fill_value = np.nan
                                    st.error(
                                        "There are no values before or after the gap."
                                    )
                                else:
                                    fill_df = datasets[gap_lysi_number].loc[
                                        (start_index - 1) : (end_index + 1),
                                        [
                                            datasets[gap_lysi_number].columns[0],
                                            col_selector,
                                        ],
                                    ]

                                    fill_value = (
                                        fill_df[[col_selector]]
                                        .interpolate(axis=0)
                                        .iloc[1 : (fill_df.shape[0] - 1), 0]
                                        .tolist()
                                    )
                            else:
                                if np.isnan(
                                    current_lysimeter_df.loc[
                                        (start_index - 1), col_selector
                                    ]
                                ) or np.isnan(
                                    current_lysimeter_df.loc[
                                        (end_index + 1), col_selector
                                    ]
                                ):
                                    fill_value = np.nan
                                    st.error(
                                        "There are no values before or after the gap."
                                    )
                                else:
                                    fill_df = current_lysimeter_df.loc[
                                        (start_index - 1) : (end_index + 1),
                                        [current_lysimeter_df.columns[0], col_selector],
                                    ]

                                    fill_value = (
                                        fill_df[[col_selector]]
                                        .interpolate(axis=0)
                                        .iloc[1 : (fill_df.shape[0] - 1), 0]
                                        .tolist()
                                    )
                        # If single gap interpolate data from previous and value afterwards in case they are not NA
                        else:
                            current_value_index = current_lysimeter_df.index[
                                current_lysimeter_df[current_lysimeter_df.columns[0]]
                                == fill_date_selector
                            ][0]

                            # If the gap is at the edge the value will be NA
                            if (
                                current_value_index == 0
                                or current_value_index == current_lysimeter_df.shape[0]
                            ):
                                fill_value = np.nan
                            else:
                                fill_df = current_lysimeter_df.loc[
                                    [
                                        current_value_index - 1,
                                        current_value_index,
                                        current_value_index + 1,
                                    ],
                                    [current_lysimeter_df.columns[0], col_selector],
                                ]
                                # If previous value or value afterwards is NA the fill value will also be NA
                                if np.isnan(fill_df.iloc[:, 1].tolist()[0]) or np.isnan(
                                    fill_df.iloc[:, 1].tolist()[2]
                                ):
                                    fill_value = np.nan
                                else:
                                    fill_value = (
                                        fill_df[[col_selector]]
                                        .interpolate(axis=0)
                                        .iloc[1, 0]
                                    )

                        st.text("NA(s) will be filled with:")
                        fill_value

                    # Apply filling
                    if st.button(
                        "Fill",
                        key="btn_apply_na",
                        type="primary",
                        disabled=fill_date_selector == None,
                    ):
                        # Fill the values in their respective gaps and remove the flags
                        if na_type_selector == "NA (no other value)":
                            gap_fill_no_other(
                                fill_date_selector, gap_lysi_number, fill_value
                            )

                        else:
                            gap_fill_edge(fill_date_selector, gap_lysi_number)

                    # Interpolate all NAs in given date range
                    st.subheader("Interpolate all NAs", divider="red")

                    # Streamlit row for date and time input from
                    from_col_interpolate, to_col_interpolate = st.columns(2)

                    with from_col_interpolate:
                        from_date_start = data[data.columns[0]].tolist()[0]

                        # From date
                        from_date_interpolate = st.date_input(
                            "From",
                            from_date_start,
                            from_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="from_date_interpolate",
                        )

                        # From time
                        from_time_interpolate = st.time_input(
                            "To",
                            datetime.datetime.strptime("00:00:00", "%H:%M:%S").time(),
                            key="from_time_interpolate",
                            label_visibility="hidden",
                        )

                        from_input = datetime.datetime.combine(
                            from_date_interpolate, from_time_interpolate
                        )

                    with to_col_interpolate:
                        to_date_start = data[data[data.columns[0]] >= from_input][
                            data.columns[0]
                        ].tolist()[0]

                        # To date
                        to_date_interpolate = st.date_input(
                            "To",
                            to_date_start,
                            to_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="to_date_interpolate",
                        )

                        # From time
                        to_time_interpolate = st.time_input(
                            "To",
                            to_date_start.time(),
                            key="to_time_interpolate",
                            label_visibility="hidden",
                        )

                        to_input = datetime.datetime.combine(
                            to_date_interpolate, to_time_interpolate
                        )

                    if from_input > to_input:
                        st.error("Select a valid date range.")
                        st.stop()

                    # Apply interpolation
                    if st.button(
                        "Interpolate", key="btn_apply_interpolate", type="primary"
                    ):
                        gap_fill_interpolate_all(
                            [from_input, to_input], gap_lysi_number, col_selector
                        )

                with col2:
                    # Undo filled NAs for gap check
                    ################################################################################################

                    st.subheader("Undo filled NAs", divider="red")

                    # Select which fill type should be undone
                    fill_type_selector = st.selectbox(
                        "Select fill type",
                        [
                            "fill (lm)",
                            "fill (interpolation single)",
                            "fill (interpolation multi)",
                        ],
                        key="select_fill",
                    )

                    # Available date ranges for multi filled gaps
                    multi_gaps_filled_labels = sorted(
                        [
                            f"{group.iloc[0,0]} to {group.iloc[(group.shape[0]-1),0]}"
                            for name, group in multi_gaps_grouped_filled
                            if group.loc[group.index[0], col_selector_code] == 2
                        ],
                        key=lambda x: datetime.datetime.strptime(
                            x.split("to")[0].strip(), "%Y-%m-%d %H:%M:%S"
                        ),
                    )

                    multi_gaps_filled_labels_interpolation = sorted(
                        [
                            f"{group.iloc[0,0]} to {group.iloc[(group.shape[0]-1),0]}"
                            for name, group in multi_gaps_grouped_filled
                            if group.loc[group.index[0], col_selector_code] == 1
                        ],
                        key=lambda x: datetime.datetime.strptime(
                            x.split("to")[0].strip(), "%Y-%m-%d %H:%M:%S"
                        ),
                    )

                    # Available dates for single filled gaps
                    single_gaps_labels = col_selector_filled[
                        (
                            col_selector_filled[col_selector_msg]
                            == "linear interpolation"
                        )
                        & (col_selector_filled[col_selector_groups].isnull())
                    ]

                    fill_type_dict = {
                        "fill (lm)": multi_gaps_filled_labels,
                        "fill (interpolation single)": single_gaps_labels,
                        "fill (interpolation multi)": multi_gaps_filled_labels_interpolation,
                    }

                    # Selector for date or date range
                    na_date_selector = st.selectbox(
                        "Select gap",
                        fill_type_dict[fill_type_selector],
                        key="select_date_fill",
                    )

                    # Apply removing fills
                    if st.button(
                        "Undo",
                        key="btn_apply_fill",
                        type="primary",
                        disabled=na_date_selector == None,
                    ):
                        # For fills of multi gaps
                        if fill_type_selector == "fill (lm)":
                            gap_undo_multi(na_date_selector, gap_lysi_number)
                        elif fill_type_selector == "fill (interpolation multi)":
                            gap_undo_multi(na_date_selector, gap_lysi_number)
                        # For fills of single gaps
                        else:
                            gap_undo_interpolation(na_date_selector, gap_lysi_number)

                    # Undo all NAs in given date range
                    st.subheader("Undo all filled NAs", divider="red")

                    undo_type = st.selectbox(
                        "filled type",
                        [
                            "fill (lm)",
                            "fill (interpolation single)",
                            "fill (interpolation multi)",
                        ],
                        key="select_undo_type",
                    )

                    # Streamlit row for date and time input from
                    from_col_undo, to_col_undo = st.columns(2)

                    with from_col_undo:
                        from_date_start = data[data.columns[0]].tolist()[0]

                        # From date
                        from_date_undo = st.date_input(
                            "From",
                            from_date_start,
                            from_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="from_date_undo",
                        )

                        # From time
                        from_time_undo = st.time_input(
                            "To",
                            datetime.datetime.strptime("00:00:00", "%H:%M:%S").time(),
                            key="from_time_undo",
                            label_visibility="hidden",
                        )

                        from_input = datetime.datetime.combine(
                            from_date_undo, from_time_undo
                        )

                    with to_col_undo:
                        to_date_start = data[data[data.columns[0]] >= from_input][
                            data.columns[0]
                        ].tolist()[0]

                        # To date
                        to_date_undo = st.date_input(
                            "To",
                            to_date_start,
                            to_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="to_date_undo",
                        )

                        # From time
                        to_time_undo = st.time_input(
                            "To",
                            to_date_start.time(),
                            key="to_time_undo",
                            label_visibility="hidden",
                        )

                        to_input = datetime.datetime.combine(to_date_undo, to_time_undo)

                    if from_input > to_input:
                        st.error("Select a valid date range.")
                        st.stop()

                    # Apply interpolation
                    if st.button("Undo", key="btn_apply_undo", type="primary"):
                        gap_undo_all(
                            [from_input, to_input], gap_lysi_number, fill_type=undo_type
                        )

            except Exception as e:
                print(e)
                st.text(e)
                st.text(f"Data for Lysimeter {lysimeter_selector} is missing")

        else:
            # Postprocessing
            ################################################################################################
            st.header(f"Postprocessing", divider="red")

            with st.expander("Partner Lysimeter Configuration"):
                st.info(
                    "Be aware that the automatic gap filling was based on the default lysimeter matches initially shown here."
                )
                current_matches = lysimeter_matches[location_selector]

                partner_col_1, partner_col_2, partner_col_3 = st.columns(3)

                with partner_col_1:
                    st.write(":red[Lysimeter 1]")
                    options_1 = [i + 1 for i in range(6) if not i == 0]
                    partner_1 = st.selectbox(
                        "Select partner lysimeter",
                        options_1,
                        index=options_1.index(current_matches["1"]),
                        key="partner_1",
                    )
                    st.write(":red[Lysimeter 4]")
                    options_4 = [i + 1 for i in range(6) if not i == 3]
                    partner_4 = st.selectbox(
                        "Select partner lysimeter",
                        options_4,
                        index=options_4.index(current_matches["4"]),
                        key="partner_4",
                    )

                with partner_col_2:
                    st.write(":red[Lysimeter 2]")
                    options_2 = [i + 1 for i in range(6) if not i == 1]
                    partner_2 = st.selectbox(
                        "Select partner lysimeter",
                        options_2,
                        index=options_2.index(current_matches["2"]),
                        key="partner_2",
                    )
                    st.write(":red[Lysimeter 5]")
                    options_5 = [i + 1 for i in range(6) if not i == 4]
                    partner_5 = st.selectbox(
                        "Select partner lysimeter",
                        options_5,
                        index=options_5.index(current_matches["5"]),
                        key="partner_5",
                    )

                with partner_col_3:
                    st.write(":red[Lysimeter 3]")
                    options_3 = [i + 1 for i in range(6) if not i == 2]
                    partner_3 = st.selectbox(
                        "Select partner lysimeter",
                        options_3,
                        index=options_3.index(current_matches["3"]),
                        key="partner_3",
                    )
                    st.write(":red[Lysimeter 6]")
                    options_6 = [i + 1 for i in range(6) if not i == 5]
                    partner_6 = st.selectbox(
                        "Select partner lysimeter",
                        options_6,
                        index=options_6.index(current_matches["6"]),
                        key="partner_6",
                    )

                save_col, reset_col, _ = st.columns([0.15, 0.15, 0.7])

                with save_col:
                    st.button(
                        "Save Config",
                        type="primary",
                        on_click=save_lysi_config,
                        kwargs={
                            "location": location_selector,
                            "config": {
                                "1": partner_1,
                                "2": partner_2,
                                "3": partner_3,
                                "4": partner_4,
                                "5": partner_5,
                                "6": partner_6,
                            },
                        },
                    )

                with reset_col:
                    if st.button("Reset", on_click=reset_lysi_config):
                        st.rerun()

            st.write("Select lysimeter")

            lysimeter_selector_col_post, lysimeter_finish_col_post = st.columns(
                [0.4, 0.6]
            )

            # Selector for lysimeter number
            with lysimeter_selector_col_post:
                lysimeter_selector_post = st.selectbox(
                    "Select lysimeter",
                    [i + 1 for i in range(6)],
                    key="select_lysimeter_post",
                    label_visibility="collapsed",
                    on_change=reset_previous_dataset,
                )

            #
            with lysimeter_finish_col_post:
                # When finishing a file (=postprocessing completed) the file is written as one big file to new folder
                st.info(
                    "Clicking the button passes all loaded lysimeter data to the next workflow step."
                )
                confirm_post = st.checkbox("I understand")
                if st.button(
                    "Finish File",
                    type="primary",
                    disabled=(not confirm_post),
                    key="finish_post",
                ):
                    all_data = []
                    # Loop over data from different lysimeter numbers
                    for index, dataset in enumerate(datasets):
                        unique_dates = [
                            date.strftime("%Y_%m_%d")
                            for date in pd.to_datetime(dataset[dataset.columns[0]])
                            .dt.date.unique()
                            .tolist()
                        ]
                        start_date = unique_dates[0]
                        end_date = unique_dates[-1]

                        pathlib.Path(
                            st.session_state["read_path"].replace(
                                f"{data_selector}_post",
                                f"{data_selector}_finished",
                            )
                        ).mkdir(parents=True, exist_ok=True)
                        os.path.join(
                            st.session_state["read_path"].replace(
                                f"{data_selector}_post",
                                f"{data_selector}_finished",
                            ),
                            f"{index+1}_{start_date}_{end_date}.csv",
                        )
                        # List existing files in the destination folder
                        existing_files = os.listdir(
                            st.session_state["read_path"].replace(
                                f"{data_selector}_post",
                                f"{data_selector}_finished",
                            )
                        )
                        # Check for each date of the lysimeter data if there is another finished file that includes this date
                        # If yes an error will be thrown
                        already_existing = False
                        for file in existing_files:
                            if not file == ".DS_Store":
                                file_search = re.search(
                                    "(\\d{4}_\\d{2}_\\d{2})_(\\d{4}_\\d{2}_\\d{2}).*",
                                    file,
                                )
                                start_date_existing = datetime.datetime.strptime(
                                    file_search.group(1), "%Y_%m_%d"
                                )
                                end_date_existing = datetime.datetime.strptime(
                                    file_search.group(2), "%Y_%m_%d"
                                )

                                start_date_saving = datetime.datetime.strptime(
                                    start_date, "%Y_%m_%d"
                                )
                                end_date_saving = datetime.datetime.strptime(
                                    end_date, "%Y_%m_%d"
                                )

                                if (
                                    (start_date_saving >= start_date_existing)
                                    and (start_date_saving <= end_date_existing)
                                ) or (
                                    (end_date_saving >= start_date_existing)
                                    and (end_date_saving <= end_date_existing)
                                ):
                                    already_existing = True

                        # If there is no conflict with existing dates the file is appended to a list
                        if not already_existing:
                            # Remove the flag columns
                            dataset_truncated_cols = [
                                col
                                for col in dataset.columns
                                if not re.match(".*fill.*", col)
                            ]
                            lysimeter_data = dataset[dataset_truncated_cols]
                            lysimeter_data.set_index(
                                lysimeter_data.columns[0], inplace=True
                            )
                            all_data.append(lysimeter_data)
                    if already_existing:
                        st.error(
                            "The loaded date range for this lysimeter overlaps with already saved data!"
                        )
                    else:
                        # Data is merged together and written
                        export_data = pd.concat(all_data, axis=1)
                        export_data.reset_index(inplace=True)
                        export_data.to_csv(
                            os.path.join(
                                st.session_state["read_path"].replace(
                                    f"{data_selector}_post",
                                    f"{data_selector}_finished",
                                ),
                                f"{start_date}_{end_date}.csv",
                            ),
                            index=False,
                        )

                        with InfluxDBClient(
                            url=os.environ.get("URL"),
                            token=os.environ.get("TOKEN"),
                            org=os.environ.get("ORG"),
                            debug=False,
                        ) as client:
                            with client.write_api(
                                write_options=SYNCHRONOUS
                            ) as write_api:
                                try:
                                    for dataset in datasets:
                                        cols_truncated = [
                                            col
                                            for col in dataset.columns
                                            if not re.match(".*fill.*", col)
                                        ]
                                        dataset_local = dataset[cols_truncated]
                                        dataset_local["location"] = location_selector
                                        dataset_local.set_index(
                                            dataset_local.columns[0], inplace=True
                                        )
                                        dataset_local["component"] = data_selector
                                        write_api.write(
                                            os.environ.get("BUCKET"),
                                            "kit",
                                            # Since influxdb adds +2 hours internal (is always UTC)
                                            record=dataset_local.tz_localize(
                                                "UTC"
                                            ),  # .tz_convert("UTC"),
                                            data_frame_measurement_name="processed",
                                            data_frame_tag_columns=[
                                                "location",
                                                "component",
                                            ],
                                        )
                                        st.session_state["influxdb_error"] = False
                                except Exception as e:
                                    print("influxdb error:")
                                    print(e)
                                    st.session_state["influxdb_error"] = True

                        reset_data()
                        st.rerun()

            # Display lysimeter plots for postprocessing
            ################################################################################################

            post_lysi_number = lysimeter_selector_post - 1
            data = datasets[post_lysi_number]
            st.subheader(f"Lysimeter {lysimeter_selector_post}", divider="gray")

            try:
                col_selector_col_post, _ = st.columns([0.4, 0.6])

                with col_selector_col_post:
                    col_selector_post = st.selectbox(
                        "Select column",
                        [
                            col
                            for col in data.columns[1:]
                            if not re.match("^fill_.*", col)
                        ],
                        key=lysimeter_selector_post,
                    )

                options_col1, options_col2 = st.columns(2)

                with options_col1:
                    downsample_data = st.checkbox("Downsample data", value=True)
                    st.info(
                        "When downsampling data, the data for the plots will be averaged to hours. This will reduce the loading time but will result in lower time resolution. Also data from the partner lysimeter will be averaged hourly for the plots."
                    )

                with options_col2:
                    static_plot = st.checkbox("Static plot", value=False)
                    st.info(
                        "When activated, all plots will be static. This can be useful when the plots are not loading properly due to the high data amount."
                    )

                if static_plot:
                    config = {"staticPlot": True}
                else:
                    config = {"staticPlot": False}

                if downsample_data and data.shape[0] > 0:
                    data_plot = (
                        data.resample("H", on=data.columns[0]).mean().reset_index()
                    )
                else:
                    data_plot = data

                render_expandables_post = st.checkbox(
                    "Render expandable section for all lysimeter plots"
                )
                st.info(
                    "Enabling rendering will result in loading of this section every time user input is updated. This can consume a lot of extra time."
                )

                if render_expandables_post:
                    # Expandable section that shows the plot of the selected parameter for all lysimeters
                    with st.expander("All lysimeter plots"):
                        st.info(
                            "The partner lysimeter is shown according to the 'lysimeter_matches.json' file. If you want to change the partner lysimeter, change it at the expandable section 'Partner Lysimeter Configuration'. But be aware that the automatic gap filling was also based on the default lysimeter matches!"
                        )
                        subtitles = []
                        for i in range(6):
                            if (i + 1) == lysimeter_selector_post:
                                subtitles.append(f"Lysimeter {i+1} (selected)")
                            elif (i + 1) == lysimeter_matches[location_summary][
                                str(lysimeter_selector_post)
                            ]:
                                subtitles.append(f"Lysimeter {i+1} (partner)")
                            else:
                                subtitles.append(f"Lysimeter {i+1}")
                        fig_all = make_subplots(
                            rows=3,
                            cols=2,
                            subplot_titles=subtitles,
                        )

                        for index, dataset in enumerate(datasets):
                            if downsample_data and dataset.shape[0] > 0:
                                dataset = (
                                    dataset.resample("H", on=dataset.columns[0])
                                    .mean()
                                    .reset_index()
                                )

                            rows = [1, 1, 2, 2, 3, 3]
                            cols = [1, 2, 1, 2, 1, 2]
                            fig_all.add_trace(
                                go.Scattergl(
                                    x=dataset[dataset.columns[0]],
                                    y=dataset[
                                        re.sub(
                                            "_\\d{1}_",
                                            f"_{index+1}_",
                                            col_selector_post,
                                        )
                                    ],
                                    mode="lines+markers",
                                    showlegend=False,
                                    marker=dict(size=2, color="#4574ba"),
                                ),
                                row=rows[index],
                                col=cols[index],
                            )
                        fig_all.update_layout(
                            height=800, width=800, title_text=f"{col_selector_post}"
                        )
                        st.plotly_chart(
                            fig_all, use_container_width=True, **{"config": config}
                        )

                # Spot for lysimeter plot of data in postprocessing
                fig_post_spot = st.empty()

                threshold_container = st.container()

                # Apply threshold in postprocessing
                ################################################################################################
                with threshold_container:
                    st.subheader("Apply Threshold", divider="red")

                    check_col1, check_col2, _ = st.columns([0.3, 0.3, 0.4])

                    # Checkbox options
                    with check_col1:
                        show_treshold_preview = st.checkbox("Show threshold preview")
                    with check_col2:
                        show_treshold_mirror_preview = st.checkbox(
                            "Mirror threshold at y-axis"
                        )

                    thresh_base_col1, thresh_base_col2 = st.columns(2)

                    with thresh_base_col1:
                        base_type = st.selectbox(
                            "Select base type", ["single value", "mean"]
                        )
                    with thresh_base_col2:
                        if base_type == "single value":
                            thresh_base = st.number_input("Select basis", step=1.0)
                        else:
                            thresh_base = data[col_selector_post].mean()

                    st.write("Select threshold relative to base")

                    thresh_col, apply_thresh_col = st.columns(2)

                    # Select threshold where values are cut off --> NA
                    with thresh_col:
                        cut_threshold = st.number_input(
                            "Select threshold", label_visibility="collapsed", step=1.0
                        )

                    data_thresh = data.copy()

                    # If threshold should be mirrored at y-axis, cutt off threshold of 10 is also -10 relative to base...
                    if show_treshold_mirror_preview:
                        lower = thresh_base - abs(cut_threshold)
                        upper = thresh_base + abs(cut_threshold)
                        data_thresh_plot = data[
                            data[col_selector_post].between(lower, upper)
                        ]
                        # Set rows where condition was met to NA
                        data_thresh.loc[:, col_selector_post] = np.where(
                            data[col_selector_post].between(lower, upper),
                            data_thresh[col_selector_post],
                            np.nan,
                        )
                        st.write(
                            f":red[Values between {round(lower,2)} and {round(upper,2)}]"
                        )
                    else:
                        # Check whether the threshold is negative or positive and filter rows accordingly
                        if cut_threshold < 0:
                            data_thresh_plot = data[
                                data[col_selector_post] >= (thresh_base + cut_threshold)
                            ]
                            data_thresh.loc[:, col_selector_post] = np.where(
                                data[col_selector_post]
                                >= (thresh_base + cut_threshold),
                                data_thresh[col_selector_post],
                                np.nan,
                            )
                            st.write(
                                f":red[Values above {round(thresh_base + cut_threshold,2)}]"
                            )
                        else:
                            data_thresh_plot = data[
                                data[col_selector_post] <= (thresh_base + cut_threshold)
                            ]
                            data_thresh.loc[:, col_selector_post] = np.where(
                                data[col_selector_post]
                                <= (thresh_base + cut_threshold),
                                data_thresh[col_selector_post],
                                np.nan,
                            )
                            st.write(
                                f":red[Values below {round(thresh_base + cut_threshold,2)}]"
                            )

                    # Find row indexes where the data has been set to NA due to the cut off threshold
                    thresh_start_index = data.index[
                        data[data.columns[0]] == data_thresh.iloc[0, 0]
                    ][0]
                    thresh_end_index = data.index[
                        data[data.columns[0]]
                        == data_thresh.iloc[(data_thresh.shape[0] - 1), 0]
                    ][0]

                    # Apply threshold on the original data and write it
                    # Similarly to earlier saving of data, date where data has been changed is filtered and these days are written out
                    with apply_thresh_col:
                        if st.button("Apply", key="btn_thresh_apply", type="primary"):
                            post_thresh(
                                data_thresh,
                                thresh_start_index,
                                thresh_end_index,
                                col_selector_post,
                                post_lysi_number,
                            )

                    # If threshold preview is checked show how the plots look before changes are applied
                    if show_treshold_preview:
                        if downsample_data and data_thresh_plot.shape[0] > 0:
                            data_thresh_plot_show = (
                                data_thresh_plot.resample(
                                    "H", on=data_thresh_plot.columns[0]
                                )
                                .mean()
                                .reset_index()
                            )
                            data_thresh_plot_show_na = (
                                data_thresh_plot.resample(
                                    "H", on=data_thresh_plot.columns[0]
                                )
                                .mean()
                                .reset_index()
                                .dropna(subset=[col_selector_post])
                            )
                        elif not downsample_data and data_thresh_plot.shape[0] > 0:
                            if data_selector == "balance":
                                data_thresh_plot_show = (
                                    data_thresh_plot.resample(
                                        "min", on=data_thresh_plot.columns[0]
                                    )
                                    .mean()
                                    .reset_index()
                                )
                            else:
                                data_thresh_plot_show = (
                                    data_thresh_plot.resample(
                                        "10min", on=data_thresh_plot.columns[0]
                                    )
                                    .mean()
                                    .reset_index()
                                )
                            data_thresh_plot_show_na = data_thresh_plot.dropna(
                                subset=[col_selector_post]
                            )
                        else:
                            data_thresh_plot_show = data_thresh_plot
                            data_thresh_plot_show_na = data_thresh_plot

                        fig_thresh = go.Figure(
                            layout=go.Layout(title="Threshold Preview")
                        )

                        fig_thresh.add_trace(
                            go.Scattergl(
                                x=data_thresh_plot_show_na[
                                    data_thresh_plot_show_na.columns[0]
                                ],
                                y=data_thresh_plot_show_na[col_selector_post],
                                showlegend=True,
                                mode="lines+markers",
                                marker=dict(size=2, color="red"),
                                name="filtered connected",
                            )
                        )
                        fig_thresh.add_trace(
                            go.Scattergl(
                                x=data_thresh_plot_show[
                                    data_thresh_plot_show.columns[0]
                                ],
                                y=data_thresh_plot_show[col_selector_post],
                                showlegend=True,
                                mode="lines+markers",
                                marker=dict(size=2, color="#4574ba"),
                                name="filtered",
                            )
                        )

                        st.plotly_chart(
                            fig_thresh, use_container_width=True, **{"config": config}
                        )

                # Remove values in postprocessing
                ################################################################################################

                remove_container = st.container()

                with remove_container:
                    st.subheader("Remove Values", divider="red")
                    st.write("Select date range")
                    remove_col1, remove_col2 = st.columns(2)

                    # Selector for date range in which to remove the values
                    with remove_col1:
                        from_date_start = data[data.columns[0]].tolist()[0]

                        # From date
                        remove_date_from = st.date_input(
                            "From",
                            from_date_start,
                            from_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="remove_range_from",
                        )

                        # From time
                        remove_time_from = st.time_input(
                            "To",
                            datetime.datetime.strptime("00:00:00", "%H:%M:%S").time(),
                            key="remove_time_from",
                            label_visibility="hidden",
                        )

                        remove_range_from = datetime.datetime.combine(
                            remove_date_from, remove_time_from
                        )

                    with remove_col2:
                        to_date_start = data[
                            data[data.columns[0]] >= remove_range_from
                        ][data.columns[0]].tolist()[0]

                        # From date
                        remove_date_to = st.date_input(
                            "To",
                            to_date_start,
                            to_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="remove_range_to",
                        )

                        # From time
                        remove_time_to = st.time_input(
                            "To",
                            to_date_start.time(),
                            key="remove_time_to",
                            label_visibility="hidden",
                        )

                        remove_range_to = datetime.datetime.combine(
                            remove_date_to, remove_time_to
                        )

                    if remove_range_from > remove_range_to:
                        st.error("The selected date range is not valid.")
                        st.stop()

                    # Filter the data accordingly to the given date ranges
                    data_filt = data.index[
                        (data[data.columns[0]] >= remove_range_from)
                        & (data[data.columns[0]] <= remove_range_to)
                    ]

                    remove_start_index = data_filt[0]
                    remove_end_index = data_filt[(data_filt.shape[0] - 1)]

                    # Preview how many data points will be removed
                    st.write("Number of Data points that will be removed:")
                    data.index[
                        (data[data.columns[0]] >= remove_range_from)
                        & (data[data.columns[0]] <= remove_range_to)
                        & (data[col_selector_post].notna())
                    ].shape[0]

                    # Apply removement
                    # For each day in the data, the data is separately written
                    if st.button("Remove", type="primary"):
                        post_remove(
                            remove_start_index,
                            remove_end_index,
                            col_selector_post,
                            post_lysi_number,
                        )

                # Add values in postprocessing
                ################################################################################################

                add_container = st.container()

                with add_container:
                    st.subheader("Fill Values", divider="red")

                    st.write("Select fill type")

                    fill_type_col, fill_preview_col = st.columns(2)

                    # Selector for fill method
                    with fill_type_col:
                        fill_type_post = st.selectbox(
                            "Select fill type",
                            [
                                "lm with partner lysimeter",
                                "trend from partner lysimeter",
                                "own trend shifted",
                                "single value",
                            ],
                            label_visibility="collapsed",
                        )

                    with fill_preview_col:
                        fill_show_preview = st.checkbox(
                            "Show fill preview", key="fill_preview"
                        )

                    st.write("Select date range")
                    fill_col1, fill_col2 = st.columns(2)

                    # Selector for date range
                    with fill_col1:
                        from_date_start = data[data.columns[0]].tolist()[0]

                        # From date
                        fill_date_from = st.date_input(
                            "From",
                            from_date_start,
                            from_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="fill_date_from",
                        )

                        # From time
                        fill_time_from = st.time_input(
                            "To",
                            datetime.datetime.strptime("00:00:00", "%H:%M:%S").time(),
                            key="fill_time_from",
                            label_visibility="hidden",
                        )

                        fill_range_from = datetime.datetime.combine(
                            fill_date_from, fill_time_from
                        )

                    with fill_col2:
                        to_date_start = data[data[data.columns[0]] >= fill_range_from][
                            data.columns[0]
                        ].tolist()[0]

                        # From date
                        fill_date_to = st.date_input(
                            "To",
                            to_date_start,
                            to_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="fill_date_to",
                        )

                        # From time
                        fill_time_to = st.time_input(
                            "To",
                            to_date_start.time(),
                            key="fill_time_to",
                            label_visibility="hidden",
                        )

                        fill_range_to = datetime.datetime.combine(
                            fill_date_to, fill_time_to
                        )

                    if fill_range_from > fill_range_to:
                        st.error("The selected date range is not valid.")
                        st.stop()

                    if fill_type_post == "single value":
                        fill_value_multi = st.number_input("Enter value", step=1.0)

                    # Get data from partner lysimeter and select respective columns
                    matching_lysimeter = lysimeter_matches[location_summary][
                        str(lysimeter_selector_post)
                    ]
                    partner_lysimeter_df = datasets[matching_lysimeter - 1]

                    partner_col = re.sub(
                        "_\\d{1}_",
                        f"_{lysimeter_matches[location_summary][str(lysimeter_selector_post)]}_",
                        col_selector_post,
                    )

                    # Different fill methods for add values for postprocessing
                    ################################################################################################
                    if fill_type_post == "single value":
                        # Single value fill
                        fill_values_post = fill_value_multi

                    elif fill_type_post == "own trend shifted":
                        # Trend fill from own lysimeter

                        # Filter data for given date range
                        current_lysimeter_df_filtered = data[
                            (data[data.columns[0]] >= fill_range_from)
                            & (data[data.columns[0]] <= fill_range_to)
                        ]

                        if fill_range_from == data.iloc[0, 0]:
                            st.error("For trend shift, edge values result in NAs!")
                            fill_values_post = [
                                np.nan
                            ] * current_lysimeter_df_filtered.shape[0]
                        else:
                            start_value_index = data.index[
                                data[data.columns[0]] >= fill_range_from
                            ][0]

                            end_value_index = data.index[
                                data[data.columns[0]] <= fill_range_to
                            ].tolist()[-1]

                            previous_value_index = start_value_index - 1

                            # Check if there are any NAs in the given date range
                            if any(
                                current_lysimeter_df_filtered[col_selector_post].isna()
                            ):
                                st.error(
                                    "There are NAs in the given date range. These values will become NAs in the filled values as well!"
                                )

                            if np.isnan(
                                data.iloc[previous_value_index][col_selector_post]
                            ):
                                st.error(
                                    "The previous value is NA. This will result in NAs in the filled lysimeter!"
                                )

                                fill_values_post = [
                                    np.nan
                                ] * current_lysimeter_df_filtered.shape[0]
                            else:
                                current_delta = (
                                    current_lysimeter_df_filtered[col_selector_post]
                                    .diff()
                                    .tolist()
                                )
                                previous_value = data.iloc[previous_value_index][
                                    col_selector_post
                                ]
                                current_delta[0] = previous_value

                                fill_values_post = np.cumsum(current_delta)

                    elif fill_type_post == "lm with partner lysimeter":
                        # regression model value fill

                        # Filter data for given date range
                        partner_lysimeter_df_filtered = partner_lysimeter_df[
                            (
                                partner_lysimeter_df[partner_lysimeter_df.columns[0]]
                                >= fill_range_from
                            )
                            & (
                                partner_lysimeter_df[partner_lysimeter_df.columns[0]]
                                <= fill_range_to
                            )
                        ]
                        # Check if partner lysimeter or current lysimeter only has NAs
                        # Only apply regression model if this is not the case
                        if not all(
                            partner_lysimeter_df_filtered[partner_col].isna()
                        ) and not all(data[col_selector_post].isna()):
                            # Exclude all dates for the regression model that are in partner_lysimeter_df_filtered where the current lysimeter has NAS/strange values
                            # This is done to avoid that the regression model is influenced by these values

                            data_exlcuded = data[
                                (data[data.columns[0]] < fill_range_from)
                                | (data[data.columns[0]] > fill_range_to)
                            ]
                            partner_excluded = partner_lysimeter_df[
                                (
                                    partner_lysimeter_df[
                                        partner_lysimeter_df.columns[0]
                                    ]
                                    < fill_range_from
                                )
                                | (
                                    partner_lysimeter_df[
                                        partner_lysimeter_df.columns[0]
                                    ]
                                    > fill_range_to
                                )
                            ]

                            if (
                                data_exlcuded.shape[0] > 1
                                and partner_excluded.shape[0] > 1
                            ):
                                model = sm.OLS(
                                    data_exlcuded[col_selector_post].values.reshape(
                                        -1, 1
                                    ),
                                    sm.add_constant(
                                        partner_excluded[partner_col].values.reshape(
                                            -1, 1
                                        )
                                    ),
                                    missing="drop",
                                )
                                lm = model.fit()

                                # Calculate values based on regression model
                                fill_values_post = (
                                    lm.params[0]
                                    + lm.params[1]
                                    * partner_lysimeter_df_filtered[partner_col]
                                )
                            else:
                                st.error(
                                    "There are not enough remaining data points for a regression model."
                                )
                                fill_values_post = [
                                    np.nan
                                ] * partner_lysimeter_df_filtered.shape[0]

                        else:
                            fill_values_post = [
                                np.nan
                            ] * partner_lysimeter_df_filtered.shape[0]

                        # If there are NAs in the partner lysimeter during this time the fill values also become NA
                        if any([np.isnan(value) for value in fill_values_post]):
                            st.error(
                                "There are NAs in the filled values! This could be due to the fact that there are also NAs in the other lysimeter."
                            )

                    else:
                        # trend fill from partner lysimeter

                        # In edge cases the first value will be NA since differences of vectors lead to the first
                        # value (or potentially the last one) being NAs

                        # in either case, differences from other lysimeter are applied to current lysimeter --> same dynamic
                        if fill_range_from == data.iloc[0, 0]:
                            st.error(
                                "For trend applications, edge values withouth previous values result in NAs!"
                            )

                            start_date_index_all = 0
                            end_date_index_current = data.index[
                                data[data.columns[0]] == fill_range_to
                            ][0]

                            end_date_index_partner = partner_lysimeter_df.index[
                                partner_lysimeter_df[partner_lysimeter_df.columns[0]]
                                == fill_range_to
                            ][0]

                            current_lysimeter_df_filtered = (
                                data.copy()
                                .loc[
                                    start_date_index_all:end_date_index_current,
                                    :,
                                ]
                                .reset_index(drop=True)
                            )

                            partner_lysimeter_df_filtered = partner_lysimeter_df.loc[
                                start_date_index_all:end_date_index_partner, :
                            ].reset_index(drop=True)

                            current_col_original_values = current_lysimeter_df_filtered[
                                col_selector_post
                            ][: (current_lysimeter_df_filtered.shape[0] - 1)].to_numpy()

                            partner_col_replace_values = (
                                partner_lysimeter_df_filtered[partner_col]
                                .diff()[1:]
                                .to_numpy()
                            )

                            if len(current_col_original_values) == 0:
                                fill_values_post = np.nan
                            else:
                                # If there are any na values inside the current or partner lysimter
                                if any(np.isnan(current_col_original_values)) or any(
                                    np.isnan(partner_col_replace_values)
                                ):
                                    # If the first value is missing a trend replacement cannot be conducted and fill value is set to NA
                                    if np.isnan(
                                        current_col_original_values[0]
                                    ) or np.isnan(partner_col_replace_values[0]):
                                        fill_values_post = np.nan

                                        st.error(
                                            "There are NAs in the partner lysimeter or current lysimeter at the beginning of this date range. This will result in NAs in the current lysimeter!"
                                        )

                                    # Otherwise trend is taken from other lysimeter and applied
                                    # In cases where a NA is in the middle of the data the fill values will become NA from that point on
                                    else:
                                        init_value = (
                                            current_col_original_values[0]
                                            + partner_col_replace_values[0]
                                        )

                                        current_col_original_values[0] = init_value
                                        current_col_original_values[
                                            1:
                                        ] = partner_col_replace_values[1:]

                                        fill_values_post = np.cumsum(
                                            current_col_original_values
                                        )

                                        if any(np.isnan(partner_col_replace_values)):
                                            st.error(
                                                "There are NAs in the partner lysimeter of this date range. This will result in NAs in the current lysimeter!"
                                            )

                                else:
                                    init_value = (
                                        current_col_original_values[0]
                                        + partner_col_replace_values[0]
                                    )

                                    current_col_original_values[0] = init_value
                                    current_col_original_values[
                                        1:
                                    ] = partner_col_replace_values[1:]

                                    fill_values_post_temp = np.cumsum(
                                        current_col_original_values
                                    )

                                    fill_values_post = [
                                        np.nan
                                    ] + fill_values_post_temp.tolist()

                        # Same procedure just that the first value is existent
                        else:
                            previous_date_index_current = (
                                data.index[data[data.columns[0]] == fill_range_from][0]
                                - 1
                            )
                            end_date_index_current = data.index[
                                data[data.columns[0]] == fill_range_to
                            ][0]

                            previous_date_index_partner = (
                                partner_lysimeter_df.index[
                                    partner_lysimeter_df[
                                        partner_lysimeter_df.columns[0]
                                    ]
                                    == fill_range_from
                                ][0]
                                - 1
                            )

                            end_date_index_partner = partner_lysimeter_df.index[
                                partner_lysimeter_df[partner_lysimeter_df.columns[0]]
                                == fill_range_to
                            ][0]

                            current_lysimeter_df_filtered = (
                                data.copy()
                                .loc[
                                    previous_date_index_current:end_date_index_current,
                                    :,
                                ]
                                .reset_index(drop=True)
                            )

                            partner_lysimeter_df_filtered = partner_lysimeter_df.loc[
                                previous_date_index_partner:end_date_index_partner, :
                            ].reset_index(drop=True)

                            current_col_original_values = current_lysimeter_df_filtered[
                                col_selector_post
                            ][: (current_lysimeter_df_filtered.shape[0] - 1)].to_numpy()

                            partner_col_replace_values = (
                                partner_lysimeter_df_filtered[partner_col]
                                .diff()[1:]
                                .to_numpy()
                            )

                            if any(np.isnan(current_col_original_values)) or any(
                                np.isnan(partner_col_replace_values)
                            ):
                                if np.isnan(current_col_original_values[0]) or np.isnan(
                                    partner_col_replace_values[0]
                                ):
                                    fill_values_post = np.nan
                                    st.error(
                                        "There are NAs in the partner lysimeter or current lysimeter at the beginning of this date range. This will result in NAs in the current lysimeter!"
                                    )

                                else:
                                    init_value = (
                                        current_col_original_values[0]
                                        + partner_col_replace_values[0]
                                    )

                                    current_col_original_values[0] = init_value
                                    current_col_original_values[
                                        1:
                                    ] = partner_col_replace_values[1:]

                                    fill_values_post = np.cumsum(
                                        current_col_original_values
                                    )

                                    if any(np.isnan(partner_col_replace_values)):
                                        st.error(
                                            "There are NAs in the partner lysimeter of this date range. This will result in NAs in the current lysimeter!"
                                        )
                            else:
                                init_value = (
                                    current_col_original_values[0]
                                    + partner_col_replace_values[0]
                                )

                                current_col_original_values[0] = init_value
                                current_col_original_values[
                                    1:
                                ] = partner_col_replace_values[1:]

                                fill_values_post = np.cumsum(
                                    current_col_original_values
                                )

                    # Plot preview for trend filling
                    if fill_show_preview:
                        fig_fill = go.Figure(layout=go.Layout(title="Fill Preview"))

                        fill_df = data.copy()[
                            (data[data.columns[0]] >= fill_range_from)
                            & (data[data.columns[0]] <= fill_range_to)
                        ]
                        fill_df.loc[:, col_selector_post] = fill_values_post

                        if downsample_data and fill_df.shape[0] > 0:
                            fill_df_plot = (
                                fill_df.resample("H", on=fill_df.columns[0])
                                .mean()
                                .reset_index()
                            )
                            partner_lysimeter_df_plot = (
                                partner_lysimeter_df.resample(
                                    "H", on=partner_lysimeter_df.columns[0]
                                )
                                .mean()
                                .reset_index()
                            )
                        else:
                            fill_df_plot = fill_df
                            partner_lysimeter_df_plot = partner_lysimeter_df

                        data_plot_exlcuded = data_plot[
                            (data_plot[data_plot.columns[0]] < fill_range_from)
                            | (data_plot[data_plot.columns[0]] > fill_range_to)
                        ]
                        filled_min = fill_df_plot[col_selector_post].min()
                        filled_max = fill_df_plot[col_selector_post].max()

                        ymin_temp = min(
                            data_plot_exlcuded[col_selector_post].min(), filled_min
                        )
                        ymin = ymin_temp - abs(0.001 * ymin_temp)
                        ymax_temp = max(
                            data_plot_exlcuded[col_selector_post].max(), filled_max
                        )
                        ymax = ymax_temp + abs(0.001 * ymax_temp)

                        fig_fill.update_layout(yaxis=dict(range=[ymin, ymax]))

                        fig_fill.add_trace(
                            go.Scattergl(
                                x=data_plot[data_plot.columns[0]],
                                y=data_plot[col_selector_post],
                                mode="lines+markers",
                                marker=dict(size=2, color="#4574ba"),
                                name="original",
                            )
                        )

                        fig_fill.add_trace(
                            go.Scattergl(
                                x=fill_df_plot[fill_df_plot.columns[0]],
                                y=fill_df_plot[col_selector_post],
                                mode="lines+markers",
                                marker=dict(size=2, color="red"),
                                name="replaced",
                            )
                        )

                        st.plotly_chart(
                            fig_fill, use_container_width=True, **{"config": config}
                        )

                        # In case the regression model has been used for filling, the regression is shown in a plot
                        if fill_type_post == "lm with partner lysimeter":
                            data_plot_exlcuded = data_plot[
                                (data_plot[data_plot.columns[0]] < fill_range_from)
                                | (data_plot[data_plot.columns[0]] > fill_range_to)
                            ]
                            partner_lysimeter_df_plot_excluded = (
                                partner_lysimeter_df_plot[
                                    (
                                        partner_lysimeter_df_plot[
                                            partner_lysimeter_df_plot.columns[0]
                                        ]
                                        < fill_range_from
                                    )
                                    | (
                                        partner_lysimeter_df_plot[
                                            partner_lysimeter_df_plot.columns[0]
                                        ]
                                        > fill_range_to
                                    )
                                ]
                            )

                            fig_fill_lm = px.scatter(
                                x=partner_lysimeter_df_plot_excluded[partner_col],
                                y=data_plot_exlcuded[col_selector_post],
                                trendline="ols",
                                color_discrete_sequence=["red"],
                                title=f"Used Linear Regression Between Lysimeter {lysimeter_matches[location_summary][str(lysimeter_selector_post)]} and Lysimeter {lysimeter_selector_post} {col_selector_post}",
                                labels={
                                    "x": f"Lysimeter {lysimeter_matches[location_summary][str(lysimeter_selector_post)]} {col_selector_post}",
                                    "y": f"Lysimeter {lysimeter_selector_post} {col_selector_post}",
                                },
                                render_mode="webgl",
                            )
                            st.plotly_chart(
                                fig_fill_lm,
                                use_container_width=True,
                                **{"config": config},
                            )

                    fill_start_index = data.index[
                        data[data.columns[0]] == fill_range_from
                    ][0]
                    fill_end_index = data.index[data[data.columns[0]] == fill_range_to][
                        0
                    ]

                    # Apply fill to the data and save each day individually in file
                    if st.button("Fill", type="primary", key="fill_post"):
                        post_fill(
                            fill_start_index,
                            fill_end_index,
                            col_selector_post,
                            post_lysi_number,
                            fill_values_post,
                        )

                # Show still remaining NAs and give option to fill them with interpolation or single value
                post_na_container = st.container()

                with post_na_container:
                    st.subheader("Interpolate Remaining NAs", divider="red")

                    # Make plot showing vlines for each missing value

                    post_na_fig = go.Figure(
                        layout=go.Layout(
                            title=f"Remaining NAs for {col_selector_post}", height=300
                        )
                    )

                    missing_dates = data_plot[data_plot[col_selector_post].isna()][
                        data_plot.columns[0]
                    ]
                    available_dates = data_plot[data_plot[col_selector_post].notna()][
                        data_plot.columns[0]
                    ]

                    post_na_fig.add_trace(
                        go.Scattergl(
                            x=available_dates,
                            y=[1] * available_dates.shape[0],
                            mode="markers+lines",
                            marker=dict(size=2, color="#4574ba"),
                            name="not NA",
                            showlegend=True,
                        )
                    )

                    post_na_fig.add_trace(
                        go.Scattergl(
                            x=missing_dates,
                            y=[1] * missing_dates.shape[0],
                            mode="markers",
                            marker=dict(size=2, color="red"),
                            name="NA",
                            showlegend=True,
                        )
                    )

                    post_na_fig.update_xaxes(showgrid=False, zeroline=False)
                    post_na_fig.update_yaxes(
                        showgrid=False,
                        zeroline=False,
                        visible=False,
                        showticklabels=False,
                    )

                    st.plotly_chart(
                        post_na_fig, use_container_width=True, **{"config": config}
                    )

                    st.write("Select date range")
                    post_na_col1, post_na_col2 = st.columns(2)

                    # Selector for date range
                    with post_na_col1:
                        from_date_start = data[data.columns[0]].tolist()[0]

                        # From date
                        post_na_date_from = st.date_input(
                            "From",
                            from_date_start,
                            from_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="post_na_date_from",
                        )

                        # From time
                        post_na_time_from = st.time_input(
                            "To",
                            datetime.datetime.strptime("00:00:00", "%H:%M:%S").time(),
                            key="post_na_time_from",
                            label_visibility="hidden",
                        )

                        post_na_range_from = datetime.datetime.combine(
                            post_na_date_from, post_na_time_from
                        )
                    with post_na_col2:
                        to_date_start = data[
                            data[data.columns[0]] >= post_na_range_from
                        ][data.columns[0]].tolist()[0]

                        post_na_date_to = st.date_input(
                            "To",
                            to_date_start,
                            to_date_start,
                            data[data.columns[0]].tolist()[-1],
                            key="post_na_date_to",
                        )

                        # From time
                        post_na_time_to = st.time_input(
                            "To",
                            to_date_start.time(),
                            key="post_na_time_to",
                            label_visibility="hidden",
                        )

                        post_na_range_to = datetime.datetime.combine(
                            post_na_date_to, post_na_time_to
                        )

                    if post_na_range_from > post_na_range_to:
                        st.error("The selected date range is not valid.")
                        st.stop()

                    # Filter data for given date range
                    data_filt = data.index[
                        (data[data.columns[0]] >= post_na_range_from)
                        & (data[data.columns[0]] <= post_na_range_to)
                    ]

                    post_na_start_index = data_filt[0]
                    post_na_end_index = data_filt[(data_filt.shape[0] - 1)]

                    na_preview, na_fill_preview = st.columns(2)

                    with na_preview:
                        # Preview how many data points will be removed
                        st.write("Total number of NAs in loaded data:")
                        data.index[data[col_selector_post].isna()].shape[0]

                    with na_fill_preview:
                        # Preview how many data points will be removed
                        st.write("Number of Data points that will be filled:")
                        data.index[
                            (data[data.columns[0]] >= post_na_range_from)
                            & (data[data.columns[0]] <= post_na_range_to)
                            & (data[col_selector_post].isna())
                        ].shape[0]

                    # Apply removement
                    # For each day in the data, the data is separately written
                    if st.button(
                        "Interpolate",
                        type="primary",
                        disabled=(
                            data.index[data[col_selector_post].isna()].shape[0] == 0
                        ),
                    ):
                        if (
                            not post_na_range_from == data.iloc[0, 0]
                            and not post_na_range_to
                            == data.iloc[(data.shape[0] - 1), 0]
                        ):
                            # Previous value before post_na_range_from
                            previous_value = data[col_selector_post][
                                data.index[data[data.columns[0]] == post_na_range_from][
                                    0
                                ]
                                - 1
                            ]

                            # Next value after post_na_range_to
                            next_value = data[col_selector_post][
                                data.index[data[data.columns[0]] == post_na_range_to][0]
                                + 1
                            ]

                            if not np.isnan(previous_value) and not np.isnan(
                                next_value
                            ):
                                gap_fill_interpolate_all(
                                    [post_na_range_from, post_na_range_to],
                                    post_lysi_number,
                                    col_selector_post,
                                )

                            else:
                                st.error(
                                    "There are NAs in the data before or after the selected date range. Please fill those first."
                                )

                        else:
                            st.error("Cannot interpolate at the edges of the data.")

                # In case the data type is 'balance', there is winter values filling available using albedo values from the EC station close by
                # TODO: clearify ec data structure
                try:
                    if data_selector == "balance":
                        winter_container = st.container()

                        with winter_container:
                            st.subheader("P and ET Winter Filling", divider="red")

                            # Fetch EC and pluvio data from InfluxDB
                            if st.button("Fetch Data", type="primary"):
                                fetch_ec(data, location_selector)

                            # If data is available, make winter filling available
                            if "ec_data" in st.session_state and isinstance(
                                st.session_state.ec_data, pd.DataFrame
                            ):
                                ec_data = st.session_state.ec_data
                                data_day = (
                                    data.resample("D", on=data.columns[0])
                                    .mean()
                                    .reset_index()
                                )

                                winter_type_col, parameter_col = st.columns(2)

                                # Select winter filling method
                                with winter_type_col:
                                    winter_type_selector = st.selectbox(
                                        "Select winter indication", ["albedo", "manual"]
                                    )

                                # Select parameter to fill
                                with parameter_col:
                                    param_selector_post = st.selectbox(
                                        "Select parameter",
                                        ["Precipitation", "Evapotranspiration"],
                                    )

                                # Warning for evapotranspiration filling due to lack of data
                                if param_selector_post == "Evapotranspiration":
                                    st.info(
                                        "At the moment the evapotranspiration is estimated using the penmon method based on the 'penmon' python package. Due to the lack of data it is only based on the min and max temperature as well as on the day of the year as well as all metrices that can be derived from that."
                                    )

                                # If albedo is used for winter filling
                                if winter_type_selector == "albedo":
                                    # TODO: implement albedo method
                                    st.info(
                                        "The albedo method is not implemented yet. Please use the manual method."
                                    )

                                    try:
                                        (
                                            albedo_col,
                                            thaw_col_length,
                                            thaw_col_type,
                                        ) = st.columns([0.5, 0.25, 0.25])

                                        # Select albedo threshold
                                        with albedo_col:
                                            albedo_threshold = st.number_input(
                                                "Select albedo threshold",
                                                min_value=0,
                                                max_value=100,
                                                value=50,
                                                step=1,
                                            )
                                            close_gaps = st.checkbox(
                                                "Ignore gaps in the middle"
                                            )
                                            show_preview = st.checkbox(
                                                "Show fill preview"
                                            )

                                        # Select thaw period
                                        with thaw_col_length:
                                            thaw_period_length = st.number_input(
                                                "Select thaw period afterwards",
                                                min_value=0,
                                                step=1,
                                            )

                                        # Select thaw period frequency unit
                                        with thaw_col_type:
                                            thaw_period_selector_post = st.selectbox(
                                                "Select thaw period afterwards",
                                                ["days", "weeks", "months"],
                                                label_visibility="hidden",
                                            )

                                        # Calculate timedelta for thaw period
                                        if thaw_period_selector_post == "days":
                                            timedelta = datetime.timedelta(
                                                days=thaw_period_length
                                            )
                                        elif thaw_period_selector_post == "weeks":
                                            timedelta = datetime.timedelta(
                                                days=(thaw_period_length * 7)
                                            )
                                        else:
                                            timedelta = relativedelta(
                                                months=+thaw_period_length
                                            )

                                        # if downsample_data and ec_data.shape[0] > 0:
                                        #     ec_data_plot = (
                                        #         ec_data.resample(
                                        #             "H", on=ec_data.columns[0]
                                        #         )
                                        #         .mean()
                                        #         .reset_index()
                                        #     )
                                        # else:

                                        ec_data_plot = ec_data

                                        # Plot albedo data
                                        fig_albedo = go.Figure(
                                            layout=go.Layout(
                                                height=350, title="Albedo by EC Station"
                                            )
                                        )

                                        fig_albedo.add_trace(
                                            go.Scattergl(
                                                x=ec_data_plot[ec_data_plot.columns[0]],
                                                y=ec_data_plot["Albedo"],
                                                mode="markers+lines",
                                                line=dict(color="red"),
                                                name="Albedo",
                                                showlegend=True,
                                            )
                                        )

                                        fig_albedo.add_hline(
                                            y=albedo_threshold, line_color="gray"
                                        )

                                        st.plotly_chart(
                                            fig_albedo,
                                            use_container_width=True,
                                            **{"config": config},
                                        )

                                        # When gaps in between should be also filled
                                        if close_gaps:
                                            # Find all dates with albedo values above threshold
                                            albedo_fills = ec_data[
                                                ec_data["Albedo"] >= albedo_threshold
                                            ].sort_values(by=[ec_data.columns[0]])
                                            pluvio_start_date = albedo_fills.iloc[0, 0]
                                            pluvio_end_date_temp = albedo_fills.iloc[
                                                (albedo_fills.shape[0] - 1), 0
                                            ]

                                            # Add thaw period to end date
                                            pluvio_end_date = (
                                                pluvio_end_date_temp + timedelta
                                            )

                                            # Filter data for given date range
                                            pluvio_df = ec_data[
                                                (
                                                    ec_data[ec_data.columns[0]]
                                                    >= pluvio_start_date
                                                )
                                                & (
                                                    ec_data[ec_data.columns[0]]
                                                    <= pluvio_end_date
                                                )
                                            ]

                                            # If precipitation is selected, apply regression model and plot relationship
                                            if (
                                                param_selector_post == "Precipitation"
                                                and not pluvio_df.shape[0]
                                                >= (
                                                    data_day[col_selector_post].shape[0]
                                                    - 1
                                                )
                                            ):
                                                model = sm.OLS(
                                                    data_day[
                                                        ~data_day[
                                                            data_day.columms[0]
                                                        ].isin(
                                                            pluvio_df[
                                                                pluvio_df.columns[0]
                                                            ].tolist()
                                                        )
                                                    ][col_selector_post].values.reshape(
                                                        -1, 1
                                                    ),
                                                    sm.add_constant(
                                                        ec_data[
                                                            ~ec_data[
                                                                ec_data.columns[0]
                                                            ].isin(
                                                                pluvio_df[
                                                                    pluvio_df.columns[0]
                                                                ].tolist()
                                                            )
                                                        ][
                                                            "precipitation"
                                                        ].values.reshape(
                                                            -1, 1
                                                        )
                                                    ),
                                                    missing="drop",
                                                )
                                                lm = model.fit()

                                                pluvio_fill_values = (
                                                    lm.params[0]
                                                    + lm.params[1]
                                                    * pluvio_df["precipitation"]
                                                )

                                                pluvio_df[
                                                    "precipitation_filled"
                                                ] = pluvio_fill_values
                                            else:
                                                pluvio_df["precipitation_filled"] = [
                                                    np.nan
                                                ] * pluvio_df.shape[0]
                                        # When no gaps should be filled
                                        else:
                                            # Select date range for filling
                                            albedo_fills = ec_data[
                                                ec_data["Albedo"] >= albedo_threshold
                                            ].sort_values(by=[ec_data.columns[0]])

                                            # When precipitation is selected as param create linear model
                                            if (
                                                param_selector_post == "Precipitation"
                                                and not albedo_fills.shape[0]
                                                >= (
                                                    data_day[col_selector_post].shape[0]
                                                    - 1
                                                )
                                            ):
                                                model = sm.OLS(
                                                    data_day[
                                                        ~data_day[
                                                            data_day.columns[0]
                                                        ].isin(
                                                            albedo_fills[
                                                                albedo_fills.columns[0]
                                                            ].tolist()
                                                        )
                                                    ][col_selector_post].values.reshape(
                                                        -1, 1
                                                    ),
                                                    sm.add_constant(
                                                        ec_data[
                                                            ~ec_data[
                                                                ec_data.columns[0]
                                                            ].isin(
                                                                albedo_fills[
                                                                    albedo_fills.columns[
                                                                        0
                                                                    ]
                                                                ].tolist()
                                                            )
                                                        ][
                                                            "precipitation"
                                                        ].values.reshape(
                                                            -1, 1
                                                        )
                                                    ),
                                                    missing="drop",
                                                )
                                                lm = model.fit()

                                            # Calculate time difference between each albedo value
                                            timestep = (
                                                albedo_fills[albedo_fills.columns[0]]
                                                .diff()
                                                .dt.seconds[1]
                                            )

                                            # Calculate timedelta between each albedo value
                                            time_diff = [0] + (
                                                albedo_fills[albedo_fills.columns[0]]
                                                / timestep
                                            ).tolist()

                                            # Add timedelta and timedelta groups to dataframe
                                            albedo_fills.loc[:, "timedelta"] = time_diff
                                            albedo_fills.loc[
                                                :, "timedelta_groups"
                                            ] = np.where(
                                                albedo_fills["timedelta"] != 1.0,
                                                "start",
                                                None,
                                            )
                                            group_start_indices = albedo_fills.index[
                                                albedo_fills["timedelta_groups"]
                                                == "start"
                                            ]

                                            groups = []
                                            # Split dataframe into groups based on timedelta
                                            for index, group_index in enumerate(
                                                group_start_indices
                                            ):
                                                if not (index + 1) == len(
                                                    group_start_indices
                                                ):
                                                    groups.append(
                                                        albedo_fills.loc[
                                                            group_index : (
                                                                group_start_indices[
                                                                    index + 1
                                                                ]
                                                                - 1
                                                            ),
                                                            :,
                                                        ]
                                                    )
                                                else:
                                                    groups.append(
                                                        albedo_fills.loc[
                                                            group_index:, :
                                                        ]
                                                    )

                                            pluvio_lst = []
                                            pluvio_counter = 1

                                            # For each group apply regression model
                                            for group in groups:
                                                pluvio_start_date = group.iloc[0, 0]
                                                pluvio_end_date_temp = group.iloc[
                                                    (group.shape[0] - 1), 0
                                                ]

                                                pluvio_end_date = (
                                                    pluvio_end_date_temp + timedelta
                                                )

                                                pluvio_df = ec_data[
                                                    (
                                                        ec_data[ec_data.columns[0]]
                                                        >= pluvio_start_date
                                                    )
                                                    & (
                                                        ec_data[ec_data.columns[0]]
                                                        <= pluvio_end_date
                                                    )
                                                ]
                                                if (
                                                    param_selector_post
                                                    == "Precipitation"
                                                    and not albedo_fills.shape[0]
                                                    >= (
                                                        data_day[
                                                            col_selector_post
                                                        ].shape[0]
                                                        - 1
                                                    )
                                                ):
                                                    pluvio_fill_values = (
                                                        lm.params[0]
                                                        + lm.params[1]
                                                        * pluvio_df["precipitation"]
                                                    )

                                                    pluvio_df[
                                                        "precipitation_filled"
                                                    ] = pluvio_fill_values
                                                    pluvio_df[
                                                        "precipitation_groups"
                                                    ] = pluvio_counter
                                                    pluvio_counter += 1

                                                    pluvio_lst.append(pluvio_df)
                                                elif (
                                                    param_selector_post
                                                    == "Evapotranspiration"
                                                ):
                                                    pluvio_lst.append(pluvio_df)
                                                else:
                                                    pluvio_df[
                                                        "precipitation_filled"
                                                    ] = [np.nan] * pluvio_df.shape[0]
                                                    pluvio_lst.append(pluvio_df)

                                            pluvio_df_temp = pd.concat(pluvio_lst)
                                            pluvio_df = pluvio_df_temp.drop_duplicates(
                                                subset=[pluvio_df_temp.columns[0]]
                                            )

                                    except Exception:
                                        pass

                                else:
                                    st.write("Select date range")

                                    winter_from_col, winter_to_col = st.columns(2)

                                    with winter_from_col:
                                        from_date_start = data[
                                            data.columns[0]
                                        ].tolist()[0]

                                        # From date
                                        winter_date_from = st.date_input(
                                            "From",
                                            from_date_start,
                                            from_date_start,
                                            data[data.columns[0]].tolist()[-1],
                                            key="winter_date_from",
                                        )

                                        # From time
                                        winter_time_from = st.time_input(
                                            "To",
                                            datetime.datetime.strptime(
                                                "00:00:00", "%H:%M:%S"
                                            ).time(),
                                            key="winter_time_from",
                                            label_visibility="hidden",
                                        )

                                        from_winter = datetime.datetime.combine(
                                            winter_date_from, winter_time_from
                                        )

                                    with winter_to_col:
                                        to_date_start = data[
                                            data[data.columns[0]] >= from_winter
                                        ][data.columns[0]].tolist()[0]

                                        winter_date_to = st.date_input(
                                            "To",
                                            to_date_start,
                                            to_date_start,
                                            data[data.columns[0]].tolist()[-1],
                                            key="winter_date_to",
                                        )

                                        # From time
                                        winter_time_to = st.time_input(
                                            "To",
                                            to_date_start.time(),
                                            key="winter_time_to",
                                            label_visibility="hidden",
                                        )

                                        to_winter = datetime.datetime.combine(
                                            winter_date_to, winter_time_to
                                        )

                                    if from_winter > to_winter:
                                        st.error(
                                            "The selected date range is not valid."
                                        )
                                        st.stop()

                                    show_preview = st.checkbox("Show fill preview")
                                    close_gaps = False

                                    pluvio_df = ec_data[
                                        (ec_data[ec_data.columns[0]] >= from_winter)
                                        & (ec_data[ec_data.columns[0]] <= to_winter)
                                    ]

                                    # If precipitation is selected, apply regression model
                                    if (
                                        param_selector_post == "Precipitation"
                                        and not pluvio_df.shape[0]
                                        >= (data_day[col_selector_post].shape[0] - 1)
                                    ):
                                        model = sm.OLS(
                                            data_day[
                                                ~data_day[data_day.columns[0]].isin(
                                                    pluvio_df[
                                                        pluvio_df.columns[0]
                                                    ].tolist()
                                                )
                                            ][col_selector_post].values.reshape(-1, 1),
                                            sm.add_constant(
                                                ec_data[
                                                    ~ec_data[ec_data.columns[0]].isin(
                                                        pluvio_df[
                                                            pluvio_df.columns[0]
                                                        ].tolist()
                                                    )
                                                ]["precipitation"].values.reshape(-1, 1)
                                            ),
                                            missing="drop",
                                        )
                                        lm = model.fit()

                                        pluvio_fill_values = (
                                            lm.params[0]
                                            + lm.params[1] * pluvio_df["precipitation"]
                                        )

                                        pluvio_df[
                                            "precipitation_filled"
                                        ] = pluvio_fill_values
                                    else:
                                        pluvio_df["precipitation_filled"] = [
                                            np.nan
                                        ] * pluvio_df.shape[0]

                                if (
                                    show_preview
                                    and not winter_type_selector == "albedo"
                                ):  # TODO: change when method implemented
                                    # if downsample_data and pluvio_df.shape[0] > 0:
                                    #     pluvio_df_plot = (
                                    #         pluvio_df.resample(
                                    #             "H", on=pluvio_df.columns[0]
                                    #         )
                                    #         .mean()
                                    #         .reset_index()
                                    #     )
                                    # else:
                                    pluvio_df_plot = pluvio_df

                                    fig_pluvio = go.Figure(
                                        layout=go.Layout(
                                            height=450, title="Fill Preview"
                                        )
                                    )
                                    if close_gaps and winter_type_selector == "albedo":
                                        if param_selector_post == "Precipitation":
                                            fig_pluvio.add_scatter(
                                                x=pluvio_df_plot[
                                                    pluvio_df_plot.columns[0]
                                                ],
                                                y=pluvio_df_plot[
                                                    "precipitation_filled"
                                                ],
                                                name="replaced",
                                                mode="markers+lines",
                                                marker=dict(size=2, color="red"),
                                            )
                                        else:
                                            fig_pluvio.add_scatter(
                                                x=pluvio_df_plot[
                                                    pluvio_df_plot.columns[0]
                                                ],
                                                y=pluvio_df_plot["eto"],
                                                name="replaced",
                                                mode="markers+lines",
                                                marker=dict(size=2, color="green"),
                                            )

                                    elif (
                                        not close_gaps
                                        and winter_type_selector == "albedo"
                                    ):
                                        show_legend = True
                                        scatter_groups = pluvio_df.groupby(
                                            [f"{param_selector_post}_groups"]
                                        )
                                        for name, group in scatter_groups:
                                            group_plot = group

                                            # if downsample_data and group.shape[0] > 0:
                                            #     group_plot = (
                                            #         group.resample(
                                            #             "H", on=group.columns[0]
                                            #         )
                                            #         .mean()
                                            #         .reset_index()
                                            #     )

                                            if param_selector_post == "Precipitation":
                                                fig_pluvio.add_scatter(
                                                    x=group_plot[group_plot.columns[0]],
                                                    y=group_plot[
                                                        f"precipitation_filled"
                                                    ],
                                                    name="replaced",
                                                    mode="markers+lines",
                                                    marker=dict(size=2, color="red"),
                                                    showlegend=show_legend,
                                                )
                                                if show_legend:
                                                    show_legend = False
                                            else:
                                                fig_pluvio.add_scatter(
                                                    x=group_plot[group_plot.columns[0]],
                                                    y=group_plot["eto"],
                                                    name="replaced",
                                                    mode="markers+lines",
                                                    marker=dict(size=2, color="green"),
                                                    showlegend=show_legend,
                                                )
                                                if show_legend:
                                                    show_legend = False
                                    else:
                                        if param_selector_post == "Precipitation":
                                            fig_pluvio.add_scatter(
                                                x=pluvio_df_plot[
                                                    pluvio_df_plot.columns[0]
                                                ],
                                                y=pluvio_df_plot[
                                                    "precipitation_filled"
                                                ],
                                                name="replaced",
                                                mode="markers+lines",
                                                marker=dict(size=2, color="red"),
                                            )
                                        else:
                                            fig_pluvio.add_scatter(
                                                x=pluvio_df_plot[
                                                    pluvio_df_plot.columns[0]
                                                ],
                                                y=pluvio_df_plot["eto"],
                                                name="replaced",
                                                mode="markers+lines",
                                                marker=dict(size=2, color="green"),
                                            )

                                    fig_pluvio.add_trace(
                                        go.Scattergl(
                                            x=data_day[data_day.columns[0]],
                                            y=data_day[col_selector_post],
                                            name="original",
                                            mode="markers+lines",
                                            marker=dict(size=2, color="#4574ba"),
                                        )
                                    )
                                    st.plotly_chart(
                                        fig_pluvio,
                                        use_container_width=True,
                                        **{"config": config},
                                    )

                                    lm_x = ec_data[
                                        ~ec_data[ec_data.columns[0]].isin(
                                            pluvio_df[pluvio_df.columns[0]].tolist()
                                        )
                                    ]

                                    lm_y = data_day[
                                        ~data_day[data_day.columns[0]].isin(
                                            pluvio_df[pluvio_df.columns[0]].tolist()
                                        )
                                    ]

                                    if (
                                        param_selector_post == "Precipitation"
                                        and lm_x.shape[0] > 1
                                    ):
                                        # if downsample_data and lm_x.shape[0] > 0:
                                        #     lm_y = (
                                        #         lm_y.resample("H", on=lm_y.columns[0])
                                        #         .mean()
                                        #         .reset_index()
                                        #     )
                                        #     lm_x = (
                                        #         lm_x.resample("H", on=lm_x.columns[0])
                                        #         .mean()
                                        #         .reset_index()
                                        #     )

                                        fig_lm = px.scatter(
                                            x=lm_x["precipitation"],
                                            y=lm_y[col_selector_post],
                                            trendline="ols",
                                            color_discrete_sequence=["red"],
                                            title=f"Used Linear Regression Between Pluvio and Lysimeter {lysimeter_selector_post} Data",
                                            labels={
                                                "x": "Pluvio Precipitation",
                                                "y": f"Lysimeter {lysimeter_selector_post} Precipitation",
                                            },
                                            render_mode="webgl",
                                        )
                                        st.plotly_chart(
                                            fig_lm,
                                            use_container_width=True,
                                            **{"config": config},
                                        )
                                    elif param_selector_post == "Precipitation":
                                        st.error(
                                            "There is no data for the regression model! This may be due to the fact that at least two days that are not within the winter period are needed for the regression model."
                                        )

                                if st.button(
                                    "Fill",
                                    type="primary",
                                    key="albedo_post",
                                    disabled=(winter_type_selector == "albedo"),
                                ):  # TODO: change when method implemented
                                    post_fill_winter(
                                        pluvio_df, param_selector_post, post_lysi_number
                                    )

                    ## Undo last action
                    ################################################################################################

                    undo_container = st.container()

                    with undo_container:
                        st.subheader("Undo last action", divider="red")

                        st.info(
                            "When changes have been made to this location, lysimeter, date range, type of operation and data type, the last action can be undone."
                        )

                    if st.button(
                        "Undo",
                        key="btn_undo",
                        type="primary",
                        disabled="previous_dataset" not in st.session_state
                        or not isinstance(
                            st.session_state["previous_dataset"], pd.DataFrame
                        ),
                    ):
                        undo_last_action(post_lysi_number)

                    with fig_post_spot:
                        if downsample_data and data.shape[0] > 0:
                            data_plot = (
                                data.resample("H", on=data.columns[0])
                                .mean()
                                .reset_index()
                            )
                        else:
                            data_plot = data

                        # Lysimeter plot that shows the actual state of the data
                        fig_post = go.Figure(
                            layout=go.Layout(
                                title=f"{col_selector_post} of Lysimeter {lysimeter_selector_post}"
                            )
                        )
                        fig_post.add_scatter(
                            x=data_plot[data_plot.columns[0]],
                            y=data_plot[col_selector_post],
                            mode="lines+markers",
                            showlegend=False,
                            marker=dict(size=2, color="#4574ba"),
                        )

                        st.plotly_chart(
                            fig_post, use_container_width=True, **{"config": config}
                        )
                except Exception:
                    st.error(
                        "Some error occured during the winter filling! May be caused by problems with fetching the data."
                    )

            except Exception as e:
                print(e)
                st.text(e)
                st.text(f"Data for Lysimeter {lysimeter_selector_post} is missing")

    else:
        st.text("No data loaded!")


except Exception:
    st.text("No data loaded")
    if "influxdb_error" in st.session_state and st.session_state.influxdb_error:
        st.error(
            "The data saved after postprocessing could not be written to the influxdb!"
        )
    st.session_state["influxdb_error"] = None
