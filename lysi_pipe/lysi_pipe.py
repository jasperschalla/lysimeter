#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :lysi_pipe.py
@Time     :2023/10/25 10:39:47
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""


import streamlit as st
from PIL import Image
import os
import re
from datetime import datetime, date, timedelta
import pandas as pd
import pathlib
from calplot import calplot
import matplotlib.pyplot as plt
from matplotlib.colors import (
    ListedColormap,
)

# plt.style.use("dark_background")

bg_color = "#0f1116"
plt.rcParams.update(
    {
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "black",
        "axes.facecolor": "white",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "savefig.facecolor": bg_color,
        "savefig.edgecolor": bg_color,
    }
)

cm = ListedColormap(["#f0796c", "#71c26e"])

log_file_error = False
location_list = ["FE1", "FE2", "FE3", "GW", "RB1", "RB2"]


def finished_dates(filename):
    """
    filename : [scalar] [string] of filename

    return   : [tuple] of [string] filename, [date] start date and [date] end date
    """
    date_groups = re.search(
        ".*(\\d{4}_\\d{2}_\\d{2})_(\\d{4}_\\d{2}_\\d{2}).csv", filename
    )
    date_start = datetime.strptime(date_groups.group(1), "%Y_%m_%d")
    date_end = datetime.strptime(date_groups.group(2), "%Y_%m_%d")

    return (filename, date_start, date_end)


# Configure page layout, title and icon
st.set_page_config(layout="wide", page_title="LysiPipe", page_icon="./favicon_pipe.ico")
# Load image for LysiPipe
image = Image.open("./lysi_pipe_text.png")
src_path = "./"
file_actions = {}
with st.columns(3)[1]:
    st.image(image)

try:
    # Read the log file and separate each entry by specific conditions
    with open(os.path.join(src_path, "lysi_data.log"), "r") as f:
        lines = f.readlines()
        for line_index, line in enumerate(lines):
            # If the line no separator or entry for watchdog
            if not re.match(".*watcher.*|.*--{5,}", line):
                line_relevant = line.split(":")[-1]

                # Get date of file
                filenames_current_temp = datetime.strptime(
                    re.search(".*_(\\d+)T.*", line_relevant).group(1), "%Y%m%d"
                ) - timedelta(days=1)
                filenames_current = filenames_current_temp.strftime("%Y-%m-%d")

                # Get location of file
                location = re.search(".*location=(.*).*", line_relevant).group(1)

                # Append to dict entry for this date
                file_actions[location][filenames_current].append(line_relevant)

            # If the line is a separator but not the last line
            elif re.match(".*--{5,}", line) and not len(lines) == (line_index + 1):
                # Only if the next line is no watchdog entry
                if not re.match(".*watcher.*", lines[line_index + 1]):
                    filenames_current_temp = datetime.strptime(
                        re.search(".*_(\\d+)T.*", lines[line_index + 1]).group(1),
                        "%Y%m%d",
                    ) - timedelta(days=1)
                    filenames_current = filenames_current_temp.strftime("%Y-%m-%d")

                    # Get location of file
                    location = re.search(
                        ".*location=(.*).*", lines[line_index + 1]
                    ).group(1)

                    # If date is not a key in the dict yet create it
                    if not location in file_actions:
                        file_actions[location] = {}
                    if not filenames_current in file_actions[location]:
                        file_actions[location][filenames_current] = []
            prev_line = line
except Exception as e:
    print(e)
    st.error(
        "Log file cannot be read! Information for tracking processes is missing, have you deleted or moved the 'lysi_data.log' file? Are there some empty lines at the end or weird characters?"
    )
    log_file_error = True

col_date, col_location, _ = st.columns([0.3, 0.3, 0.4])

# Select dates of interest (range)
with col_date:
    date_lower = date(2010, 1, 1)
    date_upper = date.today()
    date_default = (date_lower, date_upper)
    filter_date = st.date_input(
        "Select Date Range", date_default, date_lower, date_upper
    )

# Select location
with col_location:
    location_picked = st.selectbox(
        "Select Location",
        list(file_actions.keys()),
    )


if not log_file_error:
    if len(list(file_actions.keys())) == 0:
        df_status_merged = pd.DataFrame(
            columns=["date", "process", "FE1", "FE2", "FE3", "RB1", "RB2", "GW"],
        )
    else:
        # Check whether single date or date range is given
        if len(filter_date) < 2:
            st.error("Insert date range!")
        else:
            # Filter created dataframe by date range
            filter_datetime_lower = datetime.strptime(
                filter_date[0].strftime("%Y-%m-%d"), "%Y-%m-%d"
            )
            filter_datetime_upper = datetime.strptime(
                filter_date[1].strftime("%Y-%m-%d"), "%Y-%m-%d"
            )
            date_range = pd.date_range(
                filter_datetime_lower, filter_datetime_upper, freq="D"
            )
            date_str_range = date_range.strftime("%Y-%m-%d").tolist()

            file_actions = file_actions[location_picked]
            file_actions = dict(
                filter(lambda x: x[0] in date_str_range, file_actions.items())
            )

            unique_years = list(
                set([i.split("-")[0] for i in list(file_actions.keys())])
            )

            df_status = []

            for key, value in file_actions.items():
                type = [
                    "dumped",
                    "I csv",
                    "G csv",
                    "H csv",
                    "I influxdb",
                    "G influxdb",
                    "H influxdb",
                    "xlsx",
                ]

                row = pd.DataFrame(
                    {
                        "date": key,
                        "process": type,
                        "FE1": 8 * [1],
                        "FE2": 8 * [1],
                        "FE3": 8 * [1],
                        "RB1": 8 * [1],
                        "RB2": 8 * [1],
                        "GW": 8 * [1],
                    }
                )
                # Flag them as successful when the specific date and process can be found in logs for the specific location
                dumped_flag = [
                    i for i in value if re.match(f".* and .* was created.*", i)
                ]
                i_csv_flag = [i for i in value if re.match(f".*/I.*was created.*", i)]
                g_csv_flag = [i for i in value if re.match(f".*/G.*was created.*", i)]
                h_csv_flag = [i for i in value if re.match(f".*/H.*was created.*", i)]
                i_influx_flag = [
                    i for i in value if re.match(f".*I schedule.*successfully.*", i)
                ]
                g_influx_flag = [
                    i for i in value if re.match(f".*G schedule.*successfully.*", i)
                ]
                h_influx_flag = [
                    i for i in value if re.match(f".*H schedule.*successfully.*", i)
                ]
                excel_flag = [
                    i for i in value if re.match(f".*.xlsx' was created.*", i)
                ]

                row_flags = []
                row_flags.append(2 if len(dumped_flag) > 0 else 1)
                row_flags.append(2 if len(i_csv_flag) > 0 else 1)
                row_flags.append(2 if len(g_csv_flag) > 0 else 1)
                row_flags.append(2 if len(h_csv_flag) > 0 else 1)
                row_flags.append(2 if len(i_influx_flag) > 0 else 1)
                row_flags.append(2 if len(g_influx_flag) > 0 else 1)
                row_flags.append(2 if len(h_influx_flag) > 0 else 1)
                row_flags.append(2 if len(excel_flag) > 0 else 1)
                row.loc[:, location_picked] = row_flags

                df_status.append(row)

            # Concatenate all dataframes together to create dataframe to display
            df_status_merged = pd.concat(df_status)
            df_status_merged.reset_index(drop=True, inplace=True)
            df_status_merged.iloc[:, 0] = pd.to_datetime(
                df_status_merged.iloc[:, 0], format="%Y-%m-%d", errors="coerce"
            ).dt.date

            filtered_date_range = pd.date_range(
                df_status_merged.iloc[0, 0],
                df_status_merged.iloc[(df_status_merged.shape[0] - 1), 0],
                freq="D",
            )

            st.header("1. Data Restructuring", divider="red")

            st.info(
                "The log file 'lysi_data.log' is the source for this status information. Deletion of its content or the file itself will result in untrackable changes."
            )

            st.write(
                "Days where errors occured during automatic data structuring are marked in :red[red] and where no errorcs occured in :green[green]."
            )

            first_col, second_col = st.columns(2)

            with first_col:
                for process in [
                    "dumped",
                    "I csv",
                    "G csv",
                    "H csv",
                ]:
                    st.subheader(process, divider="gray")

                    date_df = pd.DataFrame({"date": filtered_date_range}).reset_index(
                        drop=True
                    )
                    value_df = pd.DataFrame(
                        {
                            "value": df_status_merged[
                                df_status_merged["process"] == process
                            ][location_picked].tolist(),
                            "date": pd.to_datetime(
                                df_status_merged[
                                    df_status_merged["process"] == process
                                ]["date"]
                            ),
                        }
                    )
                    merged_df = date_df.merge(value_df, on="date", how="left").fillna(1)

                    process_data = pd.Series(
                        merged_df["value"].to_numpy(),
                        index=filtered_date_range,
                    )

                    figure = calplot(
                        process_data,
                        edgecolor="#5c5b5b",
                        linecolor="white",
                        fillcolor="#e3e1e1",
                        cmap=cm,
                        vmin=1,
                        vmax=2,
                        yearlabel_kws={"color": "white"},
                    )
                    if len(figure[0].axes) > len(unique_years):
                        figure[0].delaxes(ax=figure[0].axes[-1])
                    st.pyplot(figure[0])

            with second_col:
                for process in [
                    "xlsx",
                    "I influxdb",
                    "G influxdb",
                    "H influxdb",
                ]:
                    st.subheader(process, divider="gray")

                    date_df = pd.DataFrame({"date": filtered_date_range}).reset_index(
                        drop=True
                    )
                    value_df = pd.DataFrame(
                        {
                            "value": df_status_merged[
                                df_status_merged["process"] == process
                            ][location_picked].tolist(),
                            "date": pd.to_datetime(
                                df_status_merged[
                                    df_status_merged["process"] == process
                                ]["date"]
                            ),
                        }
                    )
                    merged_df = date_df.merge(value_df, on="date", how="left").fillna(1)
                    process_data = pd.Series(
                        merged_df["value"].to_numpy(),
                        index=filtered_date_range,
                    )

                    figure = calplot(
                        process_data,
                        edgecolor="#5c5b5b",
                        linecolor="white",
                        fillcolor="#e3e1e1",
                        cmap=cm,
                        vmin=1,
                        vmax=2,
                        yearlabel_kws={"color": "white"},
                    )
                    if len(figure[0].axes) > len(unique_years):
                        figure[0].delaxes(ax=figure[0].axes[-1])
                    st.pyplot(figure[0])

            st.header("2. Gap Filling and Postprocessing", divider="red")

            st.info(
                "The output files of the different processes contained in the folder 'processing' are the source for this status information. Deletion of files or folders will result in untrackable changes."
            )

            st.write(
                "Days where gap filling and postprocessing has not been applied are marked in :red[red] and where it has been applied in :green[green]."
            )

            # Initliaze object to store whether files for the filtered dates lie in the folders which represent specific processes itself
            df_process = []

            # Loop over each date in filtered dataset
            df_status_merged.drop_duplicates(subset=["date"], inplace=True)
            df_status_merged.reset_index(drop=True, inplace=True)

            start_date_file = df_status_merged.loc[0, "date"]
            end_date_file = df_status_merged.loc[
                (df_status_merged.shape[0] - 1), "date"
            ]
            file_date_range = pd.date_range(start_date_file, end_date_file, freq="D")

            for file_date in file_date_range:
                file_date = file_date.date()
                # Initialize each process with unsuccesful status
                row = pd.DataFrame(
                    {
                        "date": 6
                        * [
                            file_date,
                        ],
                        "type": [
                            "balance",
                            "balance",
                            "balance",
                            "additional",
                            "additional",
                            "additional",
                        ],
                        "process": [
                            "gap_automatic",
                            "gap_check + AWAT",
                            "postprocess",
                            "gap_automatic",
                            "gap_check",
                            "postprocess",
                        ],
                        "FE1": 6 * [1],
                        "FE2": 6 * [1],
                        "FE3": 6 * [1],
                        "RB1": 6 * [1],
                        "RB2": 6 * [1],
                        "GW": 6 * [1],
                    }
                )
                date_formatted = datetime.strftime(file_date, "%Y_%m_%d")
                date_original = file_date

                # Check process for each lysimeter hexagon
                pathlib.Path(
                    os.path.join(src_path, "balance_filled", location_picked)
                ).mkdir(parents=True, exist_ok=True)
                pathlib.Path(
                    os.path.join(src_path, "additional_filled", location_picked)
                ).mkdir(parents=True, exist_ok=True)
                pathlib.Path(
                    os.path.join(src_path, "balance_post", location_picked)
                ).mkdir(parents=True, exist_ok=True)
                pathlib.Path(
                    os.path.join(src_path, "balance_finished", location_picked)
                ).mkdir(parents=True, exist_ok=True)

                pathlib.Path(
                    os.path.join(src_path, "additional_post", location_picked)
                ).mkdir(parents=True, exist_ok=True)
                pathlib.Path(
                    os.path.join(src_path, "additional_finished", location_picked)
                ).mkdir(parents=True, exist_ok=True)

                # Check whether automatic gap checking for balance data has happend
                ############################################################################

                blance_gap_flag = [
                    file
                    for file in os.listdir(
                        os.path.join(src_path, "balance_filled", location_picked)
                    )
                    if re.match(f".*{date_formatted}", file)
                ]

                balance_gap_index = row.index[
                    (row["date"] == file_date)
                    & (row["type"] == "balance")
                    & (row["process"] == "gap_automatic")
                ][0]
                row.loc[balance_gap_index, location_picked] = (
                    2 if len(blance_gap_flag) > 0 else 1
                )

                # Check whether manual gap checking for balance data has happend
                ############################################################################

                balance_gapcheck_flag = [
                    file
                    for file in os.listdir(
                        os.path.join(src_path, "balance_post", location_picked)
                    )
                    if re.match(f".*{date_formatted}", file)
                ]
                balance_gapcheck_index = row.index[
                    (row["date"] == file_date)
                    & (row["type"] == "balance")
                    & (
                        (row["process"] == "gap_check")
                        | (row["process"] == "gap_check + AWAT")
                    )
                ][0]
                row.loc[balance_gapcheck_index, location_picked] = (
                    2 if len(balance_gapcheck_flag) > 0 else 1
                )

                # Check whether postprocessing for balance data has happend
                ############################################################################
                balance_postprocess_dates = []
                balance_postprocess_date_range = [
                    finished_dates(file)
                    for file in os.listdir(
                        os.path.join(src_path, "balance_finished", location_picked)
                    )
                    if not file == ".DS_Store"
                ]
                for file_date_temp in file_date_range:
                    file = datetime.strftime(file_date_temp, "%Y-%m-%d")
                    file_date_temp = datetime.strptime(file, "%Y-%m-%d")
                    for item in balance_postprocess_date_range:
                        if file_date_temp >= item[1] and file_date_temp <= item[2]:
                            balance_postprocess_dates.append(file)

                balance_postprocess_index = row.index[
                    (row["date"] == file_date)
                    & (row["type"] == "balance")
                    & (row["process"] == "postprocess")
                ][0]
                row.loc[balance_postprocess_index, location_picked] = (
                    2
                    if len(
                        [
                            i
                            for i in balance_postprocess_dates
                            if re.match(f".*{date_original}.*", i)
                        ]
                    )
                    > 0
                    else 1
                )

                # Check whether automatic gap checking for additional data has happend
                ############################################################################

                additional_gap_flag = [
                    file
                    for file in os.listdir(
                        os.path.join(src_path, "additional_filled", location_picked)
                    )
                    if re.match(f".*{date_formatted}", file)
                ]
                additional_gap_index = row.index[
                    (row["date"] == file_date)
                    & (row["type"] == "additional")
                    & (row["process"] == "gap_automatic")
                ][0]
                row.loc[additional_gap_index, location_picked] = (
                    2 if len(additional_gap_flag) > 0 else 1
                )

                # Check whether manual gap checking for additional data has happend
                ############################################################################

                additional_gapcheck_flag = [
                    file
                    for file in os.listdir(
                        os.path.join(src_path, "additional_post", location_picked)
                    )
                    if re.match(f".*{date_formatted}", file)
                ]
                additional_gapcheck_index = row.index[
                    (row["date"] == file_date)
                    & (row["type"] == "additional")
                    & (
                        (row["process"] == "gap_check")
                        | (row["process"] == "gap_check + AWAT")
                    )
                ][0]
                row.loc[additional_gapcheck_index, location_picked] = (
                    2 if len(additional_gapcheck_flag) > 0 else 1
                )

                # Check whether postprocessing for additional data has happend
                ############################################################################

                additional_postprocess_dates = []
                additional_postprocess_date_range = [
                    finished_dates(file)
                    for file in os.listdir(
                        os.path.join(src_path, "additional_finished", location_picked)
                    )
                    if not file == ".DS_Store"
                ]
                for file_date_temp in file_date_range:
                    file = datetime.strftime(file_date_temp, "%Y-%m-%d")
                    file_date_temp = datetime.strptime(file, "%Y-%m-%d")
                    for item in additional_postprocess_date_range:
                        if file_date_temp >= item[1] and file_date_temp <= item[2]:
                            additional_postprocess_dates.append(file)

                additional_postprocess_index = row.index[
                    (row["date"] == file_date)
                    & (row["type"] == "additional")
                    & (row["process"] == "postprocess")
                ][0]
                row.loc[additional_postprocess_index, location_picked] = (
                    2
                    if len(
                        [
                            i
                            for i in additional_postprocess_dates
                            if re.match(f".*{date_original}.*", i)
                        ]
                    )
                    > 0
                    else 1
                )
                df_process.append(row)

            # If there are no dates create empty dataframe
            if len(df_process) == 0:
                df_process_merged = pd.DataFrame(
                    columns=[
                        "date",
                        "type",
                        "process",
                        "FE1",
                        "FE2",
                        "FE3",
                        "RB1",
                        "RB2",
                        "GW",
                    ]
                )
            else:
                # Concatenate all status information to dataframe
                df_process_merged = pd.concat(df_process)
                df_process_merged.iloc[:, 0] = pd.to_datetime(
                    df_process_merged.iloc[:, 0]
                )
                df_process_merged.sort_values(by=["date", "type"], inplace=True)
                df_process_merged.iloc[:, 0] = [
                    datetime.strftime(i, "%Y-%m-%d")
                    for i in df_process_merged["date"].tolist()
                ]
                df_process_merged.reset_index(drop=True, inplace=True)

            additional_col, balance_col = st.columns(2)

            with additional_col:
                st.subheader(":red[Additional]")
                for process in [
                    "gap_automatic",
                    "gap_check",
                    "postprocess",
                ]:
                    st.subheader(process, divider="gray")

                    date_df = pd.DataFrame({"date": filtered_date_range}).reset_index(
                        drop=True
                    )
                    value_df = pd.DataFrame(
                        {
                            "value": df_process_merged[
                                (df_process_merged["process"] == process)
                                & (df_process_merged["type"] == "additional")
                            ][location_picked].tolist(),
                            "date": pd.to_datetime(
                                df_process_merged[
                                    (df_process_merged["process"] == process)
                                    & (df_process_merged["type"] == "additional")
                                ]["date"]
                            ),
                        }
                    )
                    merged_df = date_df.merge(value_df, on="date", how="left").fillna(1)

                    process_data = pd.Series(
                        merged_df["value"].to_numpy(),
                        index=filtered_date_range,
                    )

                    figure = calplot(
                        process_data,
                        edgecolor="#5c5b5b",
                        linecolor="white",
                        fillcolor="#e3e1e1",
                        cmap=cm,
                        vmin=1,
                        vmax=2,
                        yearlabel_kws={"color": "white"},
                    )
                    if len(figure[0].axes) > len(unique_years):
                        figure[0].delaxes(ax=figure[0].axes[-1])

                    st.pyplot(figure[0])

            with balance_col:
                st.subheader(":red[Balance]")
                for process in [
                    "gap_automatic",
                    "gap_check + AWAT",
                    "postprocess",
                ]:
                    st.subheader(process, divider="gray")

                    date_df = pd.DataFrame({"date": filtered_date_range}).reset_index(
                        drop=True
                    )
                    value_df = pd.DataFrame(
                        {
                            "value": df_process_merged[
                                (df_process_merged["process"] == process)
                                & (df_process_merged["type"] == "balance")
                            ][location_picked].tolist(),
                            "date": pd.to_datetime(
                                df_process_merged[
                                    (df_process_merged["process"] == process)
                                    & (df_process_merged["type"] == "balance")
                                ]["date"]
                            ),
                        }
                    )
                    merged_df = date_df.merge(value_df, on="date", how="left").fillna(1)

                    process_data = pd.Series(
                        merged_df["value"].to_numpy(),
                        index=filtered_date_range,
                    )

                    figure = calplot(
                        process_data,
                        edgecolor="#5c5b5b",
                        linecolor="white",
                        fillcolor="#e3e1e1",
                        cmap=cm,
                        vmin=1,
                        vmax=2,
                        yearlabel_kws={"color": "white"},
                    )
                    if len(figure[0].axes) > len(unique_years):
                        figure[0].delaxes(ax=figure[0].axes[-1])
                    st.pyplot(figure[0])

            st.header("3. Finished File Summary", divider="red")

            additional_finished_files = []
            additional_postprocess_date_range_sorted = sorted(
                additional_postprocess_date_range, key=lambda x: x[1]
            )
            for item in additional_postprocess_date_range_sorted:
                if (
                    item[1] <= filter_datetime_upper
                    and filter_datetime_lower <= item[1]
                ) or (
                    item[2] <= filter_datetime_upper
                    and filter_datetime_lower <= item[2]
                ):
                    additional_finished_files.append(item[0])

            balance_finished_files = []
            balance_postprocess_date_range_sorted = sorted(
                balance_postprocess_date_range, key=lambda x: x[1]
            )
            for item in balance_postprocess_date_range_sorted:
                if (
                    item[1] <= filter_datetime_upper
                    and filter_datetime_lower <= item[1]
                ) or (
                    item[2] <= filter_datetime_upper
                    and filter_datetime_lower <= item[2]
                ):
                    balance_finished_files.append(item[0])

            additional_col_finished, balance_col_finished = st.columns(2)

            with additional_col_finished:
                st.subheader(":red[Additional]")

                for file in additional_finished_files:
                    st.write(f":file_folder:  {file}")

            with balance_col_finished:
                st.subheader(":red[Balance]")

                for file in balance_finished_files:
                    st.write(f":file_folder:  {file}")
