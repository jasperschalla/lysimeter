#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :postprocess.py
@Time     :2023/10/25 10:40:09
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""

import warnings

# Ignore warnings for concatenate empty pandas dataframes
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd

# Ignore warnings regarding creation of new columns, warning related to copying of datafram --> intentionally
pd.options.mode.chained_assignment = None

import openpyxl
import os
import re
import pathlib
import numpy as np
import statsmodels.api as sm
import json
import datetime
import matplotlib.pyplot as plt
import sys
from scipy.stats import iqr

# Given arguments by bash script
filepath = sys.argv[1]
# Filename without file extension
filename = filepath.split("/")[-1]
location = sys.argv[2]


# # Fitler outliers by inter quartile range for all components except water tank weights
# def filter_outliers(df, param_values, param):
#     inter_quartile_range = iqr(param_values, axis=0)
#     q1 = np.percentile(param_values, 25, axis=0)
#     q3 = np.percentile(param_values, 75, axis=0)

#     lower_thresh = q1 - 1.5 * inter_quartile_range
#     upper_thresh = q3 + 1.5 * inter_quartile_range

#     df.loc[:, param] = np.where(
#         (df[param] > upper_thresh) | (df[param] < lower_thresh),
#         np.nan,
#         df[param],
#     )

#     return df


# Open configuration which gives each lysimeter a partner lysimeter that can be used to derived gap filling data
with open("./lysimeter_matches.json", "r") as f:
    lysimeter_matches = json.loads(f.read())

# Counters for later start and end values of sequential time periods with missing values
group_counter = 1
na_group_counter = 1

filename_groups = re.search(
    "\\d*_(\\d{8})T.*",
    filename,
)
# End date from the file since the date in filename marks the end of the day e.g., 20231008 signals values for the whole 07.10
filename_date_end = datetime.datetime.strptime(filename_groups.group(1), "%Y%m%d")
# Actual date
filename_date = filename_date_end - datetime.timedelta(days=1)
# Format date for writing file
filename_date_formatted = filename_date.strftime("%Y_%m_%d")

# Get location withouth hexagon number
location_summary = location  # re.sub("\\d", "", location)

# Read the excel for the respective file given by the bash script
df_sheets = pd.read_excel(
    os.path.join(f"./excel/{location}", filename), sheet_name=None
)
# Read H and I schedules from the file
df_h = df_sheets["H"]
df_i = df_sheets["I"]

df_h_lst = []
df_i_lst = []

# Sometimes file name is not right about date in file --> check if it is the same, when not update it with date from file
if not df_sheets["G"].iloc[0, 0] == filename_date:
    filename_date_formatted = df_sheets["G"].iloc[0, 0].strftime("%Y_%m_%d")

# Loop over 6 lysimeters for this hexagon to separate data for each lysimeter in different list item
for index in range(6):
    lysimeter_number = index + 1

    # Get relevant column names for the respective lysimeter number of the i schedule
    lysimeter_cols_i = [df_i.columns[0]] + [
        col for col in df_i.columns[1:] if re.match(f"^L_{lysimeter_number}.*", col)
    ]
    df_i_temp = df_i[lysimeter_cols_i]
    # Convert datetime column to datetime
    df_i_temp.iloc[:, 0] = df_i_temp.iloc[:, 0].astype("datetime64[ns]")
    df_i_lst.append(df_i_temp)

    # Get relevant column names for the respective lysimeter number of the i schedule
    lysimeter_cols_h = [df_h.columns[0]] + [
        col for col in df_h.columns[1:] if re.match(f"^L_{lysimeter_number}.*", col)
    ]
    df_h_temp = df_h[lysimeter_cols_h]
    # Convert datetime column to datetime
    df_h_temp.iloc[:, 0] = df_h_temp.iloc[:, 0].astype("datetime64[ns]")
    df_h_lst.append(df_h_temp)

# Loop over the list where each lysimeter has own item
for lysimeter_index in range(len(df_h_lst)):
    lysimeter_number = lysimeter_index + 1

    # Correct Chamber measurements --> short periods of strong incline in weight due to weight of measuring chamber
    ############################################################################################################

    # Check if file for previous day exists that is needed for the edge values at the beginning
    # Chamber measurement at the beginning withouth incline in weight cannot be easily detected wihtouth previous file
    previous_date = filename_date_end - datetime.timedelta(days=2)
    previous_date_formatted = previous_date.strftime("%Y_%m_%d")
    filepath_folder = filepath.replace("excel", "balance_filled").split("/")
    filepath_folder.pop()
    previous_file = [
        f"{file.split('.')[0]}.csv"
        for file in os.listdir("/".join(filepath_folder))
        if re.match(f"{lysimeter_number}_{previous_date_formatted}.*", file)
    ]

    df_weights = df_i_lst[lysimeter_index]

    # If previous file exists, read it and append it to data from this date
    if len(previous_file) > 0:
        previous_filename = previous_file[0]
        df_previous = pd.read_csv(
            os.path.join(
                f"./balance_filled/{location}",
                previous_filename,
            ),
        )[df_weights.columns]
        df_previous.iloc[:, 0] = df_previous.iloc[:, 0].astype("datetime64[ns]")
        df_weights = pd.concat([df_previous, df_weights])
        df_weights = df_weights.reset_index(drop=True)

    # When no previous date exists, do some extra checks whether there is a strong decline or not finished inclince
    # (also hard to detect withouth previous file) in the first 15 time periods

    # Second column after date column is weight of lysimeter
    lysimeter_weight = df_weights[df_weights.columns[2]]
    # Check first 15 minutes for high/low changing rates indicating chamber measurement at the beginning of data for this date
    value_shift = lysimeter_weight[:20].diff()[1:]
    value_shift_start = value_shift[value_shift > 5]
    value_shift_end = value_shift[value_shift < -5]

    # Only reacts when there is a strong decrease in the 15 min window and replace values by value after 15 timesteps period
    if len(value_shift_end) > 0:
        if len(value_shift_start) > 0:
            index_shift_start = value_shift_start.index[0] - 1
            index_shift_end = value_shift_end.index[-1]
            lysimeter_weight[index_shift_start:index_shift_end] = lysimeter_weight[
                index_shift_end + 1
            ]
        else:
            index_shift_end = value_shift_end.index[-1]
            lysimeter_weight[:index_shift_end] = lysimeter_weight[index_shift_end + 1]

    # Filter corrected data by current in case previous date is still prepended
    date_start_index = df_weights.index[
        df_weights[df_weights.columns[0]] >= filename_date
    ]

    # Loop over data and check for chamber measurement (changing rate >|5|) and fill these cases by previous value
    for weight_index in range(lysimeter_weight.shape[0] - 1):
        diff = (
            lysimeter_weight.iloc[weight_index + 1]
            - lysimeter_weight.iloc[weight_index]
        )
        # Detect chamber measurement by aprupt weight changes >5kg for one time unit
        if diff > 5:
            lysimeter_weight.iloc[weight_index + 1] = lysimeter_weight.iloc[
                weight_index
            ]
        elif diff < -5:
            lysimeter_weight.iloc[weight_index + 1] = lysimeter_weight.iloc[
                weight_index
            ] = np.nan
    df_i_lst[lysimeter_index].loc[:, df_weights.columns[2]] = lysimeter_weight.iloc[
        date_start_index
    ].reset_index(drop=True)

for lysimeter_index in range(len(df_h_lst)):
    lysimeter_number = lysimeter_index + 1
    # Gap filling --> gaps = 1 timestep filled by linear interpolation, multi gaps filled by regression model with partner lysimeter
    ############################################################################################################

    # Loop over schedules with respective output folders and list items
    for schedule in ["H", "I"]:
        if schedule == "H":
            folder = "additional"
            df_all = df_h_lst
            df = df_h_lst[lysimeter_index]
        else:
            folder = "balance"
            df_all = df_i_lst
            df = df_i_lst[lysimeter_index]

        date_col = df.columns[0]

        # Loop over measured parameters
        for param in df.columns[1:]:
            # Initliaze columns for flagging missing data with extra information (how was it filled and mark groups)
            df.loc[:, f"fill_{param.lower()}_code"] = None
            df.loc[:, f"fill_{param.lower()}_msg"] = None
            df.loc[:, f"fill_{param.lower()}_groups"] = None
            df.loc[:, f"fill_{param.lower()}_na_groups"] = None

            # Plausibility check for water tank weights and other params
            partner_param = re.sub(
                "_\\d{1}_",
                f"_{lysimeter_matches[location_summary][str(lysimeter_number)]}_",
                param,
            )

            partner_lysimeter = df_all[
                (lysimeter_matches[location_summary][str(lysimeter_number)] - 1)
            ]

            if folder == "balance" and re.match(".*WAG_D_000.*", param):
                df.loc[:, param] = np.where(
                    (df[param] >= 55) | (df[param] < 0), np.nan, df[param]
                )
            elif folder == "balance" and re.match(".*WAG_L_000.*", param):
                df.loc[:, param] = np.where(
                    (df[param] >= 3600) | (df[param] <= 2400), np.nan, df[param]
                )
            # elif (
            #     df[param].dropna().shape[0] > 0
            #     and partner_lysimeter[partner_param].dropna().shape[0] > 0
            # ):
            #     param_values = df[param].dropna().to_numpy()
            #     df = filter_outliers(df, param_values, param)

            #     partner_param_values = (
            #         partner_lysimeter[partner_param].dropna().to_numpy()
            #     )
            #     partner_df_filtered = filter_outliers(
            #         partner_lysimeter, partner_param_values, partner_param
            #     )
            #     df_all[
            #         (lysimeter_matches[location_summary][str(lysimeter_number)] - 1)
            #     ] = partner_df_filtered

            #     if re.match(".*WAG_L_000.*", param):
            #         df.loc[:, param] = np.where(
            #             (df[param] >= 4000) | (df[param] <= 2000), np.nan, df[param]
            #         )
            #         df_all[
            #             (lysimeter_matches[location_summary][str(lysimeter_number)] - 1)
            #         ].loc[:, partner_param] = np.where(
            #             (partner_df_filtered[partner_param] >= 4000)
            #             | (partner_df_filtered[partner_param] <= 2000),
            #             np.nan,
            #             partner_df_filtered[partner_param],
            #         )

            # Multi gaps
            ############################################################################################################

            # Check how big a single time step is
            timestep = df[date_col].diff().dt.seconds[1]
            start_value = [0.0] if np.isnan(df[param].loc[0]) else [np.nan]

            # Data where value and next value is NA
            multi_gaps_timestamp = df[
                df[param].isna() & df[param].shift(periods=-1).isna()
            ][[date_col, param]]

            # Group timesteps where sequential gaps exist into groups with start date but open end (yet)
            if multi_gaps_timestamp.shape[0] > 0:
                multi_gaps_timestamp.loc[:, "Timedelta"] = (
                    start_value
                    + (
                        multi_gaps_timestamp[date_col].diff()[1:].dt.seconds / timestep
                    ).tolist()
                )
                multi_dates = multi_gaps_timestamp[
                    multi_gaps_timestamp["Timedelta"] != 1.0
                ][date_col].tolist()
            else:
                multi_dates = []

            # Loop over groups and check for each group where is stops to be directly sequential
            for date in multi_dates:
                prec_df = df[df[date_col] >= date]
                prec_df_interpol = df[
                    df[date_col] >= (date - datetime.timedelta(minutes=1))
                ]
                stop_index = 0
                for line, value in enumerate(prec_df[param].tolist()):
                    if not np.isnan(value):
                        stop_index = line
                        break

                stop_index = prec_df.shape[0] if stop_index == 0 else stop_index

                # Extracted group with multi-timestep gaps
                prec_df = prec_df.iloc[0:stop_index, :]

                not_edge = (
                    prec_df.loc[prec_df.index[0], "Timestamp"] != df.loc[0, "Timestamp"]
                ) and (
                    (
                        prec_df.loc[prec_df.index[-1], "Timestamp"]
                        != df.loc[(df.shape[0] - 1), "Timestamp"]
                    )
                )

                if (
                    folder == "balance"
                    and re.match(".*WAG_D_000.*", param)
                    and not_edge
                ):
                    prec_df_interpol = prec_df_interpol.iloc[0 : (stop_index + 2), :]
                    not_na = (
                        not np.isnan(
                            prec_df_interpol.loc[prec_df_interpol.index[0], param]
                        )
                    ) and (
                        not np.isnan(
                            prec_df_interpol.loc[prec_df_interpol.index[-1], param]
                        )
                    )
                else:
                    not_na = False

                if (
                    folder == "balance"
                    and re.match(".*WAG_D_000.*", param)
                    and not_edge
                    and not_na
                ):
                    result_values = (
                        prec_df_interpol.loc[:, param]
                        .interpolate(axis=0)
                        .to_numpy()[1:-1]
                    )
                else:
                    # Get data from patner lysimeter and repsective columns
                    partner_values = partner_lysimeter[
                        partner_lysimeter[date_col].isin(prec_df[date_col].tolist())
                    ][partner_param].to_numpy()

                    # Data for current date
                    current_lysimeter_values = df[param].values.reshape(-1, 1)
                    partner_lysimeter_values = partner_lysimeter[
                        partner_param
                    ].values.reshape(-1, 1)

                    ols_df = pd.DataFrame(
                        {
                            "partner": partner_lysimeter_values.flatten(),
                            "current": current_lysimeter_values.flatten(),
                        }
                    ).dropna()

                    # When all data for this or partner lysimeter is NA, output is also NA
                    if (
                        all(np.isnan(current_lysimeter_values))
                        or all(np.isnan(partner_lysimeter_values))
                        or ols_df.shape[0] == 0
                    ):
                        result_values = partner_values

                    else:
                        # Create regression model so that partner lysimeter values can predict values from current date data
                        model = sm.OLS(
                            ols_df["current"].values.reshape(-1, 1),
                            sm.add_constant(ols_df["partner"].values.reshape(-1, 1)),
                            missing="drop",
                        )
                        lm = model.fit()

                        # Sometimes there is no intercept when it is exactly 0 --> add 0.0 as intercept
                        if len(lm.params) == 1:
                            lm.params = np.concatenate([np.array([0.0]), lm.params])
                        # Calculate fill values based on the regression model
                        result_values = lm.params[0] + lm.params[1] * partner_values

                # Flag groups and single instances of missing data due to missing data in other lysimeter for later visualization
                na_indices_group = pd.Series(
                    np.argwhere(np.isnan(result_values.flatten())).flatten()
                )
                na_inidces_diff = na_indices_group - na_indices_group.shift(periods=1)

                if len(na_inidces_diff) != 0:
                    na_inidces_diff.loc[0] = 0
                group_start_indices = list(
                    na_inidces_diff.index[na_inidces_diff != 1.0]
                )

                prec_df.loc[:, param] = result_values

                # Loop over identified groups, check where the groups end and flag these time steps
                for index in group_start_indices:
                    temp_df = prec_df[na_indices_group[index] :]

                    stop_index = 0
                    for line, value in enumerate(temp_df[param].tolist()):
                        if not np.isnan(value):
                            stop_index = line
                            break
                    stop_index = prec_df.shape[0] if stop_index == 0 else stop_index

                    if stop_index > 1:
                        temp_df_dates = temp_df.iloc[:stop_index][date_col].tolist()
                        original_indices = df.index[df[date_col].isin(temp_df_dates)]
                        df.loc[
                            original_indices, f"fill_{param.lower()}_na_groups"
                        ] = na_group_counter
                        na_group_counter += 1

                # Get row indices where multi flagging has been applied
                row_indices = list(
                    df.index[df[date_col].isin(prec_df[date_col].tolist())]
                )

                # Get rows where gap filling has been applied successfully
                clean_indicators = [
                    (row_indices[index], value)
                    for index, value in enumerate(result_values)
                    if not np.isnan(value)
                ]

                row_clean_indices = [i[0] for i in clean_indicators]
                result_clean_values = [i[1] for i in clean_indicators]

                # Get rows where gap filling has been applied unsuccessfully
                na_indicators = [
                    (row_indices[index], value)
                    for index, value in enumerate(result_values)
                    if np.isnan(value)
                ]
                row_na_indices = [i[0] for i in na_indicators]

                if (
                    folder == "balance"
                    and re.match(".*WAG_D_000.*", param)
                    and not_edge
                    and not_na
                ):
                    df.loc[row_clean_indices, param] = result_clean_values
                    df.loc[row_clean_indices, f"fill_{param.lower()}_code"] = 1
                    df.loc[
                        row_clean_indices, f"fill_{param.lower()}_msg"
                    ] = "linear interpolation"
                    df.loc[
                        row_clean_indices, f"fill_{param.lower()}_groups"
                    ] = group_counter
                    group_counter += 1

                    df.loc[row_na_indices, f"fill_{param.lower()}_code"] = -1
                    df.loc[row_na_indices, f"fill_{param.lower()}_msg"] = "edge value"
                else:
                    # Flag values
                    df.loc[row_clean_indices, param] = result_clean_values
                    df.loc[row_clean_indices, f"fill_{param.lower()}_code"] = 2
                    df.loc[
                        row_clean_indices, f"fill_{param.lower()}_msg"
                    ] = f"lm with lysimeter {lysimeter_matches[location_summary][str(lysimeter_number)]}"
                    df.loc[
                        row_clean_indices, f"fill_{param.lower()}_groups"
                    ] = group_counter
                    group_counter += 1

                    df.loc[row_na_indices, f"fill_{param.lower()}_code"] = -2
                    df.loc[
                        row_na_indices, f"fill_{param.lower()}_msg"
                    ] = f"lysimeter {lysimeter_matches[location_summary][str(lysimeter_number)]} is also NA"

            # Single gaps
            ############################################################################################################

            # Select rows where value is NA and next value is not
            single_gaps_timestamp = df[
                (
                    df[param].isna()
                    & df[param].shift().notna()
                    & (~df[date_col].isin(multi_gaps_timestamp[date_col].tolist()))
                )
            ][[date_col, param]]

            # Loop over these rows and interpolate by getting previous and next value as well
            single_gap_dict = {}
            for timestamp in single_gaps_timestamp[date_col].tolist():
                index = df.index[df[date_col] == timestamp].tolist()[0]

                if not index == 0 and not index == df.shape[0]:
                    single_gap_dict[index] = (
                        df[[param]]
                        .iloc[[index - 1, index, index + 1], :]
                        .interpolate(axis=0)
                    ).iloc[1, 0]

            # Flag values where interpolation has been applied
            for key in list(single_gap_dict.keys()):
                df.loc[key, param] = single_gap_dict[key]
                df.loc[key, f"fill_{param.lower()}_code"] = 1
                df.loc[key, f"fill_{param.lower()}_msg"] = "linear interpolation"
                df.loc[key, f"fill_{param.lower()}_na_groups"] = None

            # In cases where a single value at either edge (start or finish) is missing linear interpolation cannot be applied
            if np.isnan(df[param].loc[0]) or np.isnan(df[param].loc[df.shape[0] - 1]):
                # Get current lysimeter values
                current_lysimeter_values = df[param].values.reshape(-1, 1)
                partner_lysimeter_values = partner_lysimeter[
                    partner_param
                ].values.reshape(-1, 1)

                ols_df = pd.DataFrame(
                    {
                        "partner": partner_lysimeter_values.flatten(),
                        "current": current_lysimeter_values.flatten(),
                    }
                ).dropna()

                # If first edge value is missing
                if np.isnan(df[param].loc[0]):
                    if np.isnan(partner_lysimeter[partner_param].loc[0]) or all(
                        np.isnan(current_lysimeter_values)
                    ):
                        if df.loc[0, f"fill_{param.lower()}_code"] == None:
                            df.loc[0, f"fill_{param.lower()}_code"] = -1
                            df.loc[0, f"fill_{param.lower()}_msg"] = "edge value"
                    else:
                        # Linear regression model is created
                        model = sm.OLS(
                            ols_df["current"].values.reshape(-1, 1),
                            sm.add_constant(ols_df["partner"].values.reshape(-1, 1)),
                            missing="drop",
                        )
                        lm = model.fit()
                        # Sometimes there is no intercept when it is exactly 0 --> add 0.0 as intercept
                        if len(lm.params) == 1:
                            lm.params = np.concatenate([np.array([0.0]), lm.params])
                        # Filling values are calculated by regression model
                        result_value = (
                            lm.params[0]
                            + lm.params[1] * partner_lysimeter[partner_param].loc[0]
                        )
                        # Values are filled in and flagged
                        df.loc[0, param] = result_value
                        df.loc[0, f"fill_{param.lower()}_code"] = 2
                        df.loc[
                            0, f"fill_{param.lower()}_msg"
                        ] = f"lm with lysimeter {lysimeter_matches[location_summary][str(lysimeter_number)]}"
                        df.loc[0, f"fill_{param.lower()}_groups"] = group_counter
                        group_counter += 1

                # If first edge value is missing
                if np.isnan(df[param].loc[df.shape[0] - 1]):
                    # If partner lysimeter is missing the value for this timestep the filled value is also NA and is flagged
                    if np.isnan(
                        partner_lysimeter[partner_param].loc[
                            partner_lysimeter.shape[0] - 1
                        ]
                    ) or all(np.isnan(current_lysimeter_values)):
                        if (
                            df.loc[df.shape[0] - 1, f"fill_{param.lower()}_code"]
                            == None
                        ):
                            df.loc[(df.shape[0] - 1), f"fill_{param.lower()}_code"] = -1
                            df.loc[
                                (df.shape[0] - 1), f"fill_{param.lower()}_msg"
                            ] = "edge value"
                    else:
                        # Linear regression model is created
                        model = sm.OLS(
                            ols_df["current"].values.reshape(-1, 1),
                            sm.add_constant(ols_df["partner"].values.reshape(-1, 1)),
                            missing="drop",
                        )
                        lm = model.fit()

                        # Sometimes there is no intercept when it is exactly 0 --> add 0.0 as intercept
                        if len(lm.params) == 1:
                            lm.params = np.concatenate([np.array([0.0]), lm.params])

                        # Filling values are calculated by regression model
                        result_value = (
                            lm.params[0]
                            + lm.params[1]
                            * partner_lysimeter[partner_param].loc[
                                partner_lysimeter.shape[0] - 1
                            ]
                        )
                        # Values are filled in and flagged
                        df.loc[(df.shape[0] - 1), param] = result_value
                        df.loc[(df.shape[0] - 1), f"fill_{param.lower()}_code"] = 2
                        df.loc[
                            (df.shape[0] - 1), f"fill_{param.lower()}_msg"
                        ] = f"lm with lysimeter {lysimeter_matches[location_summary][str(lysimeter_number)]}"
                        df.loc[
                            (df.shape[0] - 1), f"fill_{param.lower()}_groups"
                        ] = group_counter
                        group_counter += 1

        # Write results to file
        ############################################################################################################

        pathlib.Path(os.path.join(f"./{folder}_filled", location)).mkdir(
            parents=True, exist_ok=True
        )
        df.to_csv(
            os.path.join(
                os.path.join(f"./{folder}_filled", location),
                f"{lysimeter_number}_{filename_date_formatted}.csv",
            ),
            index=False,
        )
