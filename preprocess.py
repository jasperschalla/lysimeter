#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :preprocess.py
@Time     :2023/10/25 10:40:20
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""

import warnings

# Ignore warnings for concatenate empty pandas dataframes
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import pathlib
import sys
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
import os
import openpyxl
import numpy as np
import re
import datetime

# Logger config

logFormatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)-8s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(
    filename="./lysi_data.log", mode="a", encoding="utf-8"
)
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(logFormatter)

logger.addHandler(fileHandler)

# Given arguments in bash script
filepath = sys.argv[1]
# Filename without file extension and path
filename = filepath.split("/")[-1].split(".")[0]
filename_date = filename.split("T")[0].split("_")[-1]
filename_date_comp = re.search("(\\d{4})(\\d{2})(\\d{2})", filename_date)
date = f"{filename_date_comp.group(1)}-{filename_date_comp.group(2)}-{filename_date_comp.group(3)}"

location = sys.argv[2]

# Header names for I schedule

i_lst = []

# Header names for G schedule

g_lst = []

# Header names and list for lines for H schedule

h_lst = []

# Check if folders exist and when not create folders

pathlib.Path("./raw/G").mkdir(parents=True, exist_ok=True)
pathlib.Path("./raw/H").mkdir(parents=True, exist_ok=True)
pathlib.Path("./raw/I").mkdir(parents=True, exist_ok=True)

# Get header names by reading the dumped file with header names
with open(filepath, "r") as f:
    all_lines = f.readlines()

    split_lines = []

    for line_number, line in enumerate(all_lines):
        # 'data start' signals header names in next line
        if re.match(".*data start.*", all_lines[line_number], flags=re.IGNORECASE):
            split_lines.append(all_lines[line_number + 1])

    # Order is always the same: G,H and then I schedule --> extract the header names for the respective schedule
    g_cols = [
        i.strip()
        for i in re.sub('"|\n|\\([a-zA-Z% ]*\\)', "", split_lines[0]).split(",")
    ]
    h_cols = [
        i.strip()
        for i in re.sub('"|\n|\\([a-zA-Z% ]*\\)', "", split_lines[1]).split(",")
    ]
    i_cols = [
        i.strip()
        for i in re.sub('"|\n|\\([a-zA-Z% ]*\\)', "", split_lines[2]).split(",")
    ]

# Get data from the dumped file withouth header
with open(filepath.replace("dumped", "dumped_head"), "r") as f:
    all_lines = f.readlines()

    # Decide which line belongs to which schedule by separating each line by delimiter and check with which header list length (G,H or I) the line coincides
    for line in all_lines:
        if len(line.split(",")) == len(i_cols):
            i_lst.append(line)
        elif len(line.split(",")) == len(g_cols):
            g_lst.append(line)
        elif len(line.split(",")) == len(h_cols):
            h_lst.append(line)


# Merge read infos and write csv files for all different schedules
try:
    with open(f"./raw/I/{filename}.csv", "w") as i_file:
        i_file.write(",".join(i_cols))
        i_file.write("\n")
        for i_line in i_lst:
            if not i_line == "" and not re.match(
                ".*timestamp.*", i_line, flags=re.IGNORECASE
            ):
                i_file.write(i_line)

    logger.info(f"./raw/I/{filename}.csv was created for location={location}")
except Exception:
    logger.error(f"./raw/I/{filename}.csv could not be created for location={location}")

try:
    with open(f"./raw/G/{filename}.csv", "w") as g_file:
        g_file.write(",".join(g_cols))
        g_file.write("\n")
        for g_line in g_lst:
            if not g_line == "" and not re.match(
                ".*timestamp.*", g_line, flags=re.IGNORECASE
            ):
                g_file.write(g_line)

    logger.info(f"./raw/G/{filename}.csv was created for location={location}")
except Exception:
    logger.error(f"./raw/G/{filename}.csv could not be created for location={location}")

try:
    with open(f"./raw/H/{filename}.csv", "w") as h_file:
        h_file.write(",".join(h_cols))
        h_file.write("\n")
        for h_line in h_lst:
            if not h_line == "" and not re.match(
                ".*timestamp.*", h_line, flags=re.IGNORECASE
            ):
                h_file.write(h_line)

    logger.info(f"./raw/H/{filename}.csv was created for location={location}")
except Exception:
    logger.error(f"./raw/H/{filename}.csv could not be created for location={location}")

# Establish connection to influxdb
with InfluxDBClient(
    url=os.environ.get("URL"),
    token=os.environ.get("TOKEN"),
    org=os.environ.get("ORG"),
    debug=False,
) as client:
    with client.write_api(write_options=SYNCHRONOUS) as write_api:
        try:
            # Read created schedule files (H,I,G), round Timestamp for all three files and flag H and I schedule

            i_csv = pd.read_csv(f"./raw/I/{filename}.csv", parse_dates=["Timestamp"])
            i_csv["Timestamp"] = i_csv["Timestamp"].map(lambda x: x.round("min"))

            # Create full date range for 1 and 10min so that missing dates are in the dataset but filled with NA
            start_date_min_i = datetime.datetime.strptime(
                f"{i_csv.loc[0,'Timestamp'].strftime('%Y-%m-%d')} 00:00:00",
                "%Y-%m-%d %H:%M:%S",
            )

            end_date_min_i = datetime.datetime.strptime(
                f"{i_csv.loc[i_csv.shape[0]-1,'Timestamp'].strftime('%Y-%m-%d')} 23:59:00",
                "%Y-%m-%d %H:%M:%S",
            )

            date_range_min_i = pd.date_range(
                start_date_min_i, end_date_min_i, freq="min"
            )
            date_min_df_i = pd.DataFrame({"Timestamp": date_range_min_i})

            i_csv = date_min_df_i.merge(i_csv, on="Timestamp", how="left")

            i_csv.set_index("Timestamp", inplace=True)
            # Still, sometimes there are duplicate timestamps
            i_csv = i_csv[~i_csv.index.duplicated(keep="first")]

            ############################################################################################
            # Flag i values
            # nr = not responsive, oos = out of service, nsc = no service conditions, ip = implausible
            i_flags = []
            for col in i_cols[1:]:
                nr_df = i_csv[
                    (i_csv[col] == 999.8)
                    | (i_csv[col] == 99.8)
                    | (i_csv[col] == -991.0)
                ].copy()
                nr_df["parameter"] = col
                nr_df["reason"] = "nr"

                oos_df = i_csv[i_csv[col] == 990.0].copy()
                oos_df["parameter"] = col
                oos_df["reason"] = "oos"

                nsc_df = i_csv[(i_csv[col] == -99.8) | (i_csv[col] == -999.8)].copy()
                nsc_df["parameter"] = col
                nsc_df["reason"] = "nsc"

                if i_csv[col].dtype == "object":
                    ip_df = i_csv[
                        (i_csv[col] == "Overrange")
                        | (i_csv[col] == "overrange")
                        | (i_csv[col] == "Underrange")
                        | (i_csv[col] == "underrange")
                        | (i_csv[col] == 999.0)
                    ].copy()
                    # ip_df = i_csv[
                    #     i_csv[col].str.contains(
                    #         "Overrange|overrange|Underrange|underrange", na=False
                    #     )
                    # ].copy()
                    ip_df["parameter"] = col
                    ip_df["reason"] = "ip"
                else:
                    ip_df = pd.DataFrame(columns=["parameter", "reason"])

                flag_df = pd.concat(
                    [
                        nr_df[["parameter", "reason"]],
                        oos_df[["parameter", "reason"]],
                        nsc_df[["parameter", "reason"]],
                        ip_df[["parameter", "reason"]],
                    ]
                )
                i_flags.append(flag_df)

            i_flag_sheet = pd.concat(i_flags).copy()

            i_csv.replace(990.0, np.nan, inplace=True)
            i_csv.replace(999.8, np.nan, inplace=True)
            i_csv.replace(-999.8, np.nan, inplace=True)
            i_csv.replace(999.0, np.nan, inplace=True)
            i_csv.replace(-99.8, np.nan, inplace=True)

            # Sometimes 'Overrange' value can be found in columns making the column type string
            # Change this 'Overrange' to NA and convert the column to numeric
            i_string_cols = list(i_csv.select_dtypes(include=["object"]).columns)
            if len(i_string_cols) > 0:
                for col in i_string_cols:
                    i_csv[col] = pd.to_numeric(i_csv[col], errors="coerce")
            i_csv = i_csv.apply(pd.to_numeric, downcast="float")
            i_csv.loc[:, "location"] = location

            ############################################################################################

            h_csv = pd.read_csv(f"./raw/H/{filename}.csv", parse_dates=["Timestamp"])
            h_csv["Timestamp"] = h_csv["Timestamp"].map(lambda x: x.round("min"))

            start_date_min_h = datetime.datetime.strptime(
                f"{h_csv.loc[0,'Timestamp'].strftime('%Y-%m-%d')} 00:00:00",
                "%Y-%m-%d %H:%M:%S",
            )

            end_date_min_h = datetime.datetime.strptime(
                f"{h_csv.loc[h_csv.shape[0]-1,'Timestamp'].strftime('%Y-%m-%d')} 23:59:00",
                "%Y-%m-%d %H:%M:%S",
            )

            date_range_10min_h = pd.date_range(
                start_date_min_h, end_date_min_h, freq="10min"
            )
            date_10min_df_h = pd.DataFrame({"Timestamp": date_range_10min_h})

            h_csv = date_10min_df_h.merge(h_csv, on="Timestamp", how="left")
            h_csv.set_index("Timestamp", inplace=True)
            # Still, sometimes there are duplicate timestamps
            h_csv = h_csv[~h_csv.index.duplicated(keep="first")]

            ############################################################################################
            # Flag h values
            # nr = not responsive, oos = out of service, nsc = no service conditions, ip = implausible
            h_flags = []
            for col in h_cols[1:]:
                nr_df = h_csv[
                    (h_csv[col] == 999.8)
                    | (h_csv[col] == 99.8)
                    | (h_csv[col] == -991.0)
                ].copy()
                nr_df["parameter"] = col
                nr_df["reason"] = "nr"

                oos_df = h_csv[h_csv[col] == 990.0].copy()
                oos_df["parameter"] = col
                oos_df["reason"] = "oos"

                nsc_df = h_csv[(h_csv[col] == -99.8) | (h_csv[col] == -999.8)].copy()
                nsc_df["parameter"] = col
                nsc_df["reason"] = "nsc"

                if h_csv[col].dtype == "object":
                    ip_df = h_csv[
                        (h_csv[col] == "Overrange")
                        | (h_csv[col] == "overrange")
                        | (h_csv[col] == "Underrange")
                        | (h_csv[col] == "underrange")
                        | (h_csv[col] == 999.0)
                    ].copy()
                    # ip_df = h_csv[
                    #     h_csv[col].str.contains(
                    #         "Overrange|overrange|Underrange|underrange", na=False
                    #     )
                    # ].copy()
                    ip_df["parameter"] = col
                    ip_df["reason"] = "ip"
                else:
                    ip_df = pd.DataFrame(columns=["parameter", "reason"])

                flag_df = pd.concat(
                    [
                        nr_df[["parameter", "reason"]],
                        oos_df[["parameter", "reason"]],
                        nsc_df[["parameter", "reason"]],
                        ip_df[["parameter", "reason"]],
                    ]
                )
                h_flags.append(flag_df)

            h_flag_sheet = pd.concat(h_flags).copy()

            h_csv.replace(990.0, np.nan, inplace=True)
            h_csv.replace(999.8, np.nan, inplace=True)
            h_csv.replace(-999.8, np.nan, inplace=True)
            h_csv.replace(999.0, np.nan, inplace=True)
            h_csv.replace(-99.8, np.nan, inplace=True)

            # Sometimes 'Overrange' value can be found in columns making the column type string
            # Change this 'Overrange' to NA and convert the column to numeric
            h_string_cols = list(h_csv.select_dtypes(include=["object"]).columns)
            if len(h_string_cols) > 0:
                for col in h_string_cols:
                    h_csv[col] = pd.to_numeric(h_csv[col], errors="coerce")
            h_csv = h_csv.apply(pd.to_numeric, downcast="float")
            h_csv.loc[:, "location"] = location

            ############################################################################################

            g_csv = pd.read_csv(f"./raw/G/{filename}.csv", parse_dates=["Timestamp"])

            start_date_min_g = datetime.datetime.strptime(
                f"{g_csv.loc[0,'Timestamp'].strftime('%Y-%m-%d')} 00:00:00",
                "%Y-%m-%d %H:%M:%S",
            )

            end_date_min_g = datetime.datetime.strptime(
                f"{g_csv.loc[g_csv.shape[0]-1,'Timestamp'].strftime('%Y-%m-%d')} 23:59:00",
                "%Y-%m-%d %H:%M:%S",
            )

            date_range_min_g = pd.date_range(
                start_date_min_g, end_date_min_g, freq="min"
            )
            date_min_df_g = pd.DataFrame({"Timestamp": date_range_min_g})

            g_csv["Timestamp"] = g_csv["Timestamp"].map(lambda x: x.round("min"))
            g_csv = date_min_df_g.merge(g_csv, on="Timestamp", how="left")

            g_csv.set_index("Timestamp", inplace=True)
            # Still, sometimes there are duplicate timestamps
            g_csv = g_csv[~g_csv.index.duplicated(keep="first")]

            # Sometimes 'Overrange' value can be found in columns making the column type string
            # Change this 'Overrange' to NA and convert the column to numeric
            g_string_cols = list(g_csv.select_dtypes(include=["object"]).columns)
            if len(g_string_cols) > 0:
                for col in g_string_cols:
                    g_csv[col] = pd.to_numeric(g_csv[col], errors="coerce")
            g_csv = g_csv.apply(pd.to_numeric, downcast="float")
            g_csv.loc[:, "location"] = location

            # Merge the flagged values for H and I schedule in a dataframe
            # Sometimes there are values with too short distance between each other (only a few seconds) => remove duplicate flags
            # merged_flags = (
            #     pd.concat([h_flag_sheet, i_flag_sheet])
            #     .reset_index(names="Timestamp")
            #     .drop_duplicates()
            #     .set_index("Timestamp")
            # )
            # pivoted_flags = pd.pivot(
            #     merged_flags, values="reason", columns=["parameter"]
            # )

            pivoted_flags_h = pd.pivot(
                h_flag_sheet.reset_index(names="Timestamp").set_index("Timestamp"),
                values="reason",
                columns=["parameter"],
            )

            pivoted_flags_h["schedule"] = "H"
            pivoted_flags_h["location"] = location

            pivoted_flags_i = pd.pivot(
                i_flag_sheet.reset_index(names="Timestamp").set_index("Timestamp"),
                values="reason",
                columns=["parameter"],
            )

            pivoted_flags_i["schedule"] = "I"
            pivoted_flags_i["location"] = location

            available_dates_g = (
                g_csv.reset_index()["Timestamp"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            available_dates_h = (
                h_csv.reset_index()["Timestamp"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            available_dates_i = (
                i_csv.reset_index()["Timestamp"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )

            # Write all schedules to influxdb when there is more than one day of data available (Error in the data logger)
            pathlib.Path(f"./excel/{location}").mkdir(parents=True, exist_ok=True)
            if (
                len(available_dates_g) > 1
                or len(available_dates_h) > 1
                or len(available_dates_i) > 1
            ):
                name_date = datetime.datetime.strptime(
                    date, "%Y-%m-%d"
                ) - datetime.timedelta(days=1)

                # Create start and end date for this date (as name intended) from 00:00:00 to 23:59:00 in min
                start_date_min = datetime.datetime.strptime(
                    f"{name_date.strftime('%Y-%m-%d')} 00:00:00", "%Y-%m-%d %H:%M:%S"
                )
                end_date_min = datetime.datetime.strptime(
                    f"{name_date.strftime('%Y-%m-%d')} 23:59:00", "%Y-%m-%d %H:%M:%S"
                )

                # Create pandas data frame with date range in min for merging with schedules
                date_range_min = pd.date_range(start_date_min, end_date_min, freq="min")
                date_min_df = pd.DataFrame({"Timestamp": date_range_min})

                # Create start end end date for this date (as name intended) from 00:00:00 to 23:50:00 in 10min
                start_date_10min = datetime.datetime.strptime(
                    f"{name_date.strftime('%Y-%m-%d')} 00:00:00", "%Y-%m-%d %H:%M:%S"
                )
                end_date_10min = datetime.datetime.strptime(
                    f"{name_date.strftime('%Y-%m-%d')} 23:50:00", "%Y-%m-%d %H:%M:%S"
                )

                # Create pandas data frame with date range in 10min for merging with schedules
                date_range_10min = pd.date_range(
                    start_date_10min, end_date_10min, freq="10min"
                )
                date_10min_df = pd.DataFrame({"Timestamp": date_range_10min})

                # Filter all three schedules for this date
                g_csv_date = date_min_df.merge(
                    g_csv[
                        (g_csv.index >= start_date_min) & (g_csv.index <= end_date_min)
                    ],
                    on="Timestamp",
                    how="left",
                ).set_index("Timestamp")
                h_csv_date = date_10min_df.merge(
                    h_csv[
                        (h_csv.index >= start_date_10min)
                        & (h_csv.index <= end_date_10min)
                    ],
                    on="Timestamp",
                    how="left",
                ).set_index("Timestamp")
                i_csv_date = date_min_df.merge(
                    i_csv[
                        (i_csv.index >= start_date_min) & (i_csv.index <= end_date_min)
                    ],
                    on="Timestamp",
                    how="left",
                ).set_index("Timestamp")

                pivoted_flags_date_h = pivoted_flags_h[
                    (pivoted_flags_h.index >= start_date_min)
                    & (pivoted_flags_h.index <= end_date_min)
                ]

                pivoted_flags_date_i = pivoted_flags_i[
                    (pivoted_flags_i.index >= start_date_min)
                    & (pivoted_flags_i.index <= end_date_min)
                ]

                # If all schedules are empty, rais ValueError (exact error does not matter)
                if i_csv.shape[0] == 0 or h_csv.shape[0] == 0 or g_csv.shape[0] == 0:
                    logger.warning("Multiple dates in file but not date from filename.")
                    raise ValueError
                else:
                    with pd.ExcelWriter(
                        f"./excel/{location}/{filename}.xlsx"
                    ) as writer:
                        g_csv_date.drop(["location"], axis=1).to_excel(
                            writer, sheet_name="G"
                        )
                        i_csv_date.drop(["location"], axis=1).to_excel(
                            writer, sheet_name="I"
                        )
                        h_csv_date.drop(["location"], axis=1).to_excel(
                            writer, sheet_name="H"
                        )
                        pivoted_flags_date_h.to_excel(writer, sheet_name="Flags_H")
                        pivoted_flags_date_i.to_excel(writer, sheet_name="Flags_I")
                        logger.info(
                            f"file '{filename}.xlsx' was created for location={location}"
                        )

                    g_csv["schedule"] = "G"
                    i_csv["schedule"] = "I"
                    h_csv["schedule"] = "H"

                    write_api.write(
                        os.environ.get("BUCKET"),
                        os.environ.get("ORG"),
                        # Since influxdb adds +2 hours internal (is always UTC)
                        record=i_csv_date.tz_localize(
                            "UTC"
                        ),  # .tz_convert("Europe/Berlin"),
                        data_frame_measurement_name="raw",
                        data_frame_tag_columns=["location", "schedule"],
                    )
                    logger.info(
                        f"file '{filename}.csv's I schedule has been successfully written to influxdb for location={location}"
                    )

                    write_api.write(
                        os.environ.get("BUCKET"),
                        os.environ.get("ORG"),
                        # Since influxdb adds +2 hours internal (is always UTC)
                        record=h_csv_date.tz_localize(
                            "UTC"
                        ),  # .tz_convert("Europe/Berlin"),
                        data_frame_measurement_name="raw",
                        data_frame_tag_columns=["location", "schedule"],
                    )
                    logger.info(
                        f"file '{filename}.csv's H schedule has been successfully written to influxdb for location={location}"
                    )

                    write_api.write(
                        os.environ.get("BUCKET"),
                        os.environ.get("ORG"),
                        # Since influxdb adds +2 hours internal (is always UTC)
                        record=g_csv_date.tz_localize(
                            "UTC"
                        ),  # .tz_convert("Europe/Berlin"),
                        data_frame_measurement_name="raw",
                        data_frame_tag_columns=["location", "schedule"],
                    )
                    logger.info(
                        f"file '{filename}.csv's G schedule has been successfully written to influxdb for location={location}"
                    )

                    if pivoted_flags_date_i.shape[0] > 0:
                        write_api.write(
                            os.environ.get("BUCKET"),
                            os.environ.get("ORG"),
                            # Since influxdb adds +2 hours internal (is always UTC)
                            record=pivoted_flags_date_i.tz_localize(
                                "UTC"
                            ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),,
                            data_frame_measurement_name="flags",
                            data_frame_tag_columns=["location", "schedule"],
                        )
                        logger.info(
                            f"file '{filename}.csv's I Flags schedule has been successfully written to influxdb for location={location}"
                        )
                    else:
                        logger.info(
                            f"file '{filename}.csv's I Flags schedule is empty and has not been written to influxdb for location={location}"
                        )

                    if pivoted_flags_i.shape[0] > 0:
                        write_api.write(
                            os.environ.get("BUCKET"),
                            os.environ.get("ORG"),
                            # Since influxdb adds +2 hours internal (is always UTC)
                            record=pivoted_flags_date_h.tz_localize(
                                "UTC"
                            ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),,
                            data_frame_measurement_name="flags",
                            data_frame_tag_columns=["location", "schedule"],
                        )
                        logger.info(
                            f"file '{filename}.csv's H Flags schedule has been successfully written to influxdb for location={location}"
                        )
                    else:
                        logger.info(
                            f"file '{filename}.csv's H Flags schedule is empty and has not been written to influxdb for location={location}"
                        )

                # Remove name_date from available dates
                available_dates_g.remove(name_date.strftime("%Y-%m-%d"))
                available_dates_h.remove(name_date.strftime("%Y-%m-%d"))
                available_dates_i.remove(name_date.strftime("%Y-%m-%d"))

                excel_dates = [
                    (
                        datetime.datetime.strptime(
                            re.search(".*_(.*)T.*", file).group(1), "%Y%m%d"
                        )
                        - datetime.timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                    # re.search(".*_(.*)T.*", file).group(1)
                    for file in os.listdir(f"./excel/{location}")
                    if not file == ".DS_Store"
                ]

                # Get dates that are in all three schedules available
                available_dates_all = list(
                    set(available_dates_g)
                    & set(available_dates_h)
                    & set(available_dates_i)
                )

                # Loop over all dates that are in all three schedules available
                for date in available_dates_all:
                    # Loop over all already filled dates too see if available date is missing so far
                    if date not in excel_dates:
                        # Create start and end date for this date from 00:00:00 to 23:59:00 in min
                        start_date_min = datetime.datetime.strptime(
                            f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S"
                        )
                        end_date_min = datetime.datetime.strptime(
                            f"{date} 23:59:00", "%Y-%m-%d %H:%M:%S"
                        )

                        # Create pandas data frame with date range in min for merging with schedules
                        date_range_min = pd.date_range(
                            start_date_min, end_date_min, freq="min"
                        )
                        date_min_df = pd.DataFrame({"Timestamp": date_range_min})

                        # Create start end end date for this date from 00:00:00 to 23:50:00 in 10min
                        start_date_10min = datetime.datetime.strptime(
                            f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S"
                        )
                        end_date_10min = datetime.datetime.strptime(
                            f"{date} 23:50:00", "%Y-%m-%d %H:%M:%S"
                        )

                        # Create pandas data frame with date range in 10min for merging with schedules
                        date_range_10min = pd.date_range(
                            start_date_10min, end_date_10min, freq="10min"
                        )
                        date_10min_df = pd.DataFrame({"Timestamp": date_range_10min})

                        # Filter all three schedules for this date
                        g_csv_date = date_min_df.merge(
                            g_csv[
                                (g_csv.index >= start_date_min)
                                & (g_csv.index <= end_date_min)
                            ],
                            on="Timestamp",
                            how="left",
                        ).set_index("Timestamp")
                        h_csv_date = date_10min_df.merge(
                            h_csv[
                                (h_csv.index >= start_date_10min)
                                & (h_csv.index <= end_date_10min)
                            ],
                            on="Timestamp",
                            how="left",
                        ).set_index("Timestamp")
                        i_csv_date = date_min_df.merge(
                            i_csv[
                                (i_csv.index >= start_date_min)
                                & (i_csv.index <= end_date_min)
                            ],
                            on="Timestamp",
                            how="left",
                        ).set_index("Timestamp")
                        pivoted_flags_date_h = pivoted_flags_h[
                            (pivoted_flags_h.index >= start_date_min)
                            & (pivoted_flags_h.index <= end_date_min)
                        ]
                        pivoted_flags_date_i = pivoted_flags_i[
                            (pivoted_flags_i.index >= start_date_min)
                            & (pivoted_flags_i.index <= end_date_min)
                        ]

                        # Check if not all columns are NA from g_csv_date, h_csv_date, i_csv_date and pivoted_flags_date
                        if (
                            not g_csv_date.drop(["location"], axis=1)
                            .dropna(how="all")
                            .shape[0]
                            == 0
                            or not h_csv_date.drop(["location"], axis=1)
                            .dropna(how="all")
                            .shape[0]
                            == 0
                            or not i_csv_date.drop(["location"], axis=1)
                            .dropna(how="all")
                            .shape[0]
                            == 0
                            or not pivoted_flags_date_h.dropna(how="all").shape[0] == 0
                            or not pivoted_flags_date_i.dropna(how="all").shape[0] == 0
                        ):
                            error_date_name = (
                                start_date_min + datetime.timedelta(days=1)
                            ).strftime("%Y%m%d")

                            with pd.ExcelWriter(
                                f"./excel/{location}/error_{error_date_name}T000.xlsx"
                            ) as writer:
                                g_csv_date.drop(["location"], axis=1).to_excel(
                                    writer, sheet_name="G"
                                )
                                i_csv_date.drop(["location"], axis=1).to_excel(
                                    writer, sheet_name="I"
                                )
                                h_csv_date.drop(["location"], axis=1).to_excel(
                                    writer, sheet_name="H"
                                )
                                pivoted_flags_date_h.to_excel(
                                    writer, sheet_name="Flags_H"
                                )
                                pivoted_flags_date_i.to_excel(
                                    writer, sheet_name="Flags_I"
                                )

                            g_csv["schedule"] = "G"
                            i_csv["schedule"] = "I"
                            h_csv["schedule"] = "H"

                            logger.warning(
                                f"Multiple dates in one file. file 'error_{error_date_name}T000.xlsx' has been created for location={location}. Extra postprocessing of file needed by running the script 'resolve_date_errors.py'"
                            )

                            write_api.write(
                                os.environ.get("BUCKET"),
                                os.environ.get("ORG"),
                                # Since influxdb adds +2 hours internal (is always UTC)
                                record=i_csv_date.tz_localize(
                                    "UTC"
                                ),  # .tz_convert("Europe/Berlin"),
                                data_frame_measurement_name="raw",
                                data_frame_tag_columns=["location", "schedule"],
                            )
                            logger.info(
                                f"file 'error_{error_date_name}T000.xlsx' from '{filename}.csv's I schedule has been successfully written to influxdb for location={location}"
                            )

                            write_api.write(
                                os.environ.get("BUCKET"),
                                os.environ.get("ORG"),
                                # Since influxdb adds +2 hours internal (is always UTC)
                                record=h_csv_date.tz_localize(
                                    "UTC"
                                ),  # .tz_convert("Europe/Berlin"),
                                data_frame_measurement_name="raw",
                                data_frame_tag_columns=["location", "schedule"],
                            )
                            logger.info(
                                f"file 'error_{error_date_name}T000.xlsx' from '{filename}.csv's H schedule has been successfully written to influxdb for location={location}"
                            )

                            write_api.write(
                                os.environ.get("BUCKET"),
                                os.environ.get("ORG"),
                                # Since influxdb adds +2 hours internal (is always UTC)
                                record=g_csv_date.tz_localize(
                                    "UTC"
                                ),  # .tz_convert("Europe/Berlin"),
                                data_frame_measurement_name="raw",
                                data_frame_tag_columns=["location", "schedule"],
                            )
                            logger.info(
                                f"file 'error_{error_date_name}T000.xlsx' from '{filename}.csv's G schedule has been successfully written to influxdb for location={location}"
                            )

                            if pivoted_flags_date_i.shape[0] > 0:
                                write_api.write(
                                    os.environ.get("BUCKET"),
                                    org=os.environ.get("ORG"),
                                    # Since influxdb adds +2 hours internal (is always UTC)
                                    record=pivoted_flags_date_i.tz_localize(
                                        "UTC"
                                    ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),,
                                    data_frame_measurement_name="flags",
                                    data_frame_tag_columns=["location", "schedule"],
                                )

                                logger.info(
                                    f"file 'error_{error_date_name}T000.xlsx' from '{filename}.csv's I Flags schedule has been successfully written to influxdb for location={location}"
                                )
                            else:
                                logger.info(
                                    f"file 'error_{error_date_name}T000.xlsx' from '{filename}.csv's I Flags schedule is empty and has not been written to influxdb for location={location}"
                                )

                            if pivoted_flags_date_h.shape[0] > 0:
                                write_api.write(
                                    os.environ.get("BUCKET"),
                                    org=os.environ.get("ORG"),
                                    # Since influxdb adds +2 hours internal (is always UTC)
                                    record=pivoted_flags_date_h.tz_localize(
                                        "UTC"
                                    ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),,
                                    data_frame_measurement_name="flags",
                                    data_frame_tag_columns=["location", "schedule"],
                                )

                                logger.info(
                                    f"file 'error_{error_date_name}T000.xlsx' from '{filename}.csv's H Flags schedule has been successfully written to influxdb for location={location}"
                                )
                            else:
                                logger.info(
                                    f"file 'error_{error_date_name}T000.xlsx' from '{filename}.csv's H Flags schedule is empty and has not been written to influxdb for location={location}"
                                )

            else:
                with pd.ExcelWriter(f"./excel/{location}/{filename}.xlsx") as writer:
                    g_csv.drop(["location"], axis=1).to_excel(writer, sheet_name="G")
                    i_csv.drop(["location"], axis=1).to_excel(writer, sheet_name="I")
                    h_csv.drop(["location"], axis=1).to_excel(writer, sheet_name="H")
                    pivoted_flags_h.to_excel(writer, sheet_name="Flags_H")
                    pivoted_flags_i.to_excel(writer, sheet_name="Flags_I")
                logger.info(
                    f"file '{filename}.xlsx' was created for location={location}"
                )

                g_csv["schedule"] = "G"
                i_csv["schedule"] = "I"
                h_csv["schedule"] = "H"

                # Write schedules and flag dataframe to influxdb
                write_api.write(
                    os.environ.get("BUCKET"),
                    os.environ.get("ORG"),
                    # Since influxdb adds +2 hours internal (is always UTC)
                    record=i_csv.tz_localize(
                        "UTC"
                    ),  # tz_localize("Europe/Berlin").tz_convert("UTC")
                    data_frame_measurement_name="raw",
                    data_frame_tag_columns=["location", "schedule"],
                )
                logger.info(
                    f"file '{filename}.csv's I schedule has been successfully written to influxdb for location={location}"
                )

                write_api.write(
                    os.environ.get("BUCKET"),
                    os.environ.get("ORG"),
                    # Since influxdb adds +2 hours internal (is always UTC)
                    record=h_csv.tz_localize(
                        "UTC"
                    ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),
                    data_frame_measurement_name="raw",
                    data_frame_tag_columns=["location", "schedule"],
                )
                logger.info(
                    f"file '{filename}.csv's H schedule has been successfully written to influxdb for location={location}"
                )

                write_api.write(
                    os.environ.get("BUCKET"),
                    os.environ.get("ORG"),
                    # Since influxdb adds +2 hours internal (is always UTC)
                    record=g_csv.tz_localize(
                        "UTC"
                    ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),
                    data_frame_measurement_name="raw",
                    data_frame_tag_columns=["location", "schedule"],
                )

                logger.info(
                    f"file '{filename}.csv's G schedule has been successfully written to influxdb for location={location}"
                )

                if pivoted_flags_i.shape[0] > 0:
                    write_api.write(
                        os.environ.get("BUCKET"),
                        os.environ.get("ORG"),
                        # Since influxdb adds +2 hours internal (is always UTC)
                        record=pivoted_flags_i.tz_localize(
                            "UTC"
                        ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),,
                        data_frame_measurement_name="flags",
                        data_frame_tag_columns=["location", "schedule"],
                    )

                    logger.info(
                        f"file '{filename}.csv's I Flags schedule has been successfully written to influxdb for location={location}"
                    )
                else:
                    logger.info(
                        f"file '{filename}.csv's I Flags schedule is empty and has not been written to influxdb for location={location}"
                    )

                if pivoted_flags_h.shape[0] > 0:
                    write_api.write(
                        os.environ.get("BUCKET"),
                        os.environ.get("ORG"),
                        # Since influxdb adds +2 hours internal (is always UTC)
                        record=pivoted_flags_h.tz_localize(
                            "UTC"
                        ),  # tz_localize("Europe/Berlin").tz_convert("UTC"),,
                        data_frame_measurement_name="flags",
                        data_frame_tag_columns=["location", "schedule"],
                    )

                    logger.info(
                        f"file '{filename}.csv's H Flags schedule has been successfully written to influxdb for location={location}"
                    )
                else:
                    logger.info(
                        f"file '{filename}.csv's H Flags schedule is empty and has not been written to influxdb for location={location}"
                    )

        except Exception as e:
            print(e)
            logger.error(
                f"G,H and I schedule of file '{filename}.csv' could not be written to influxdb for location={location}"
            )

logger.info("------------------------------------------------------")
# Remove the temporary files that have been created as intermediate products
os.remove(filepath)
os.remove(filepath.replace("dumped", "dumped_head"))
os.remove(filepath.replace("dumped", "original").replace(".csv", ".DBD"))
os.remove(filepath.replace("dumped", "raw/H"))
os.remove(filepath.replace("dumped", "raw/G"))
os.remove(filepath.replace("dumped", "raw/I"))
