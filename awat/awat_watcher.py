#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :awat_watcher.py
@Time     :2023/11/15 14:39:11
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
import pathlib
import re
from subprocess import Popen
import os
import datetime
import pandas as pd
import signal
import subprocess


def round_seconds(obj):
    if obj.microsecond >= 500_000:
        obj += datetime.timedelta(seconds=1)
    return obj.replace(microsecond=0)


# Configure Observer

directory_name = "awat"
directory = f"./{directory_name}/"


class Watcher:
    def __init__(self, directory, handler=FileSystemEventHandler()):
        self.observer = Observer()
        self.handler = handler
        self.directory = directory

    def run(self):
        self.observer.schedule(self.handler, self.directory, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
        self.observer.join()


# Configure handler


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and re.match(
            ".*rawdata\\d{1}_\\d+.*", event.src_path
        ):
            file = str(pathlib.Path(event.src_path))
            filename = file.split("/")[-1]

            index = re.search(".*rawdata(\\d{1})_.*", file).group(1)
            location = re.search("^(.*)_rawdata.*", filename).group(1)
            time_start = re.search(".*rawdata\\d{1}_(.*).dat$", filename).group(1)

            write_lines = []

            with open(
                f"{directory}input.dat",
                "r",
            ) as f:
                lines = f.readlines()
                for line_index, line in enumerate(lines):
                    if re.match(".*\\.dat.*", line):
                        write_lines.append(
                            re.sub("^.*\\.dat.*", file.split("/")[-1], line)
                        )
                    elif re.match(".*-\\d{1}.*", line):
                        write_lines.append(f"{location}-{index}\n")
                    elif not index == 0 and re.match(
                        ".*time for output.*", lines[line_index - 1]
                    ):
                        write_lines.append(f"{time_start}\n")
                    else:
                        write_lines.append(line)

            with open(
                f"{directory}input.dat",
                "w",
            ) as f:
                f.writelines(write_lines)

            # Execute AWAT filter

            # Indicate how long AWAT3.exe will take
            temp_rawdata = pd.read_csv(filename, sep="\t")
            pause_seconds = (temp_rawdata.shape[0] / 13000) * 8

            pro = subprocess.Popen(
                ["wine AWAT3.exe"],
                stdout=subprocess.PIPE,
                shell=True,
                preexec_fn=os.setsid,
            )
            time.sleep(pause_seconds)
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

            # Clean output main
            """ out_main = {
                "time": [],
                "System_raw": [],
                "System_smooth": [],
                "Outfl": [],
                "EVAP": [],
                "PREC": [],
            }

            with open("./OUT_MAIN.dat", "r") as f:
                lines = f.readlines()
                for line in lines[2:]:
                    line_split = line.split()
                    out_main["time"].append(float(line_split[0]))
                    out_main["System_raw"].append(float(line_split[1]))
                    out_main["System_smooth"].append(float(line_split[2]))
                    out_main["Outfl"].append(float(line_split[3]))
                    out_main["EVAP"].append(float(line_split[4]))
                    out_main["PREC"].append(float(line_split[5]))

            out_main = pd.DataFrame(out_main)

            start_time = round_seconds(datetime.datetime(1899, 12, 30, 0, 0, 0) + datetime.timedelta(
                days=out_main.iloc[0, 0]
            ))
            end_time = round_seconds(datetime.datetime(1899, 12, 30, 0, 0, 0) + datetime.timedelta(
                days=out_main.iloc[out_main.shape[0] - 1, 0]
            ))
            date_range = pd.date_range(start=start_time, end=end_time, freq="min")
            out_main.iloc[:, 0] = date_range
            pathlib.Path(f"./balance_post/{location}/out_main").mkdir(
                parents=True, exist_ok=True
            )

            out_main["Outfl"] = [0] + out_main["Outfl"].diff().tolist()[1:]
            out_main["EVAP"] = [0] + out_main["EVAP"].diff().tolist()[1:]
            out_main["PREC"] = [0] + out_main["PREC"].diff().tolist()[1:]

            days = out_main["time"].dt.strftime("%Y-%m-%d").unique().tolist()[1:-1]
             
            for day in days:
                start_date = datetime.datetime.strptime(
                    f"{day} 00:00:00", "%Y-%m-%d %H:%M:%S"
                )
                end_date = datetime.datetime.strptime(
                    f"{day} 23:59:00", "%Y-%m-%d %H:%M:%S"
                )
                out_main_single = out_main[
                    (out_main["time"] >= start_date) & (out_main["time"] <= end_date)
                ]

                out_main_single.to_csv(
                    f"./balance_post/{location}/out_main/{index}_{start_date.strftime('%Y_%m_%d')}.csv",
                    index=False,
                ) """

            # Clean daily output

            out_day = {
                "time": [],
                "D_EVAP": [],
                "D_PREC": [],
                "D_DRAIN": [],
            }

            with open("./outthin_1d.dat", "r") as f:
                lines = f.readlines()
                for line in lines[2:]:
                    line_split = line.split()
                    out_day["time"].append(float(line_split[0]))
                    out_day["D_EVAP"].append(float(line_split[1]))
                    out_day["D_PREC"].append(float(line_split[2]))
                    out_day["D_DRAIN"].append(float(line_split[3]))

            out_day = pd.DataFrame(out_day)

            start_time = round_seconds(
                datetime.datetime(1899, 12, 30, 0, 0, 0)
                + datetime.timedelta(days=out_day.iloc[0, 0])
                - datetime.timedelta(days=1)
            )
            end_time = round_seconds(
                datetime.datetime(1899, 12, 30, 0, 0, 0)
                + datetime.timedelta(days=out_day.iloc[out_day.shape[0] - 1, 0])
                - datetime.timedelta(days=1)
            )

            date_range = pd.date_range(start=start_time, end=end_time, freq="D")
            out_day.iloc[:, 0] = date_range
            pathlib.Path(f"./balance_post/{location}/outthin/").mkdir(
                parents=True, exist_ok=True
            )

            days = out_day["time"].unique().tolist()[1:]

            for day in days:
                out_day_single = out_day[out_day["time"] == day]
                out_day_single.to_csv(
                    f"./balance_post/{location}/outthin/{index}_{day.strftime('%Y_%m_%d')}.csv",
                    index=False,
                )

            other_rawdata = [
                file
                for file in os.listdir("./")
                if re.match(".*rawdata\\d{1}_\\d+.*", file) and not file == filename
            ]
            stay_files = [
                "automation.py",
                "awat_watcher.py",
                "AWAT3.exe",
                "input.dat",
            ] + other_rawdata

            for file in os.listdir("./"):
                if file not in stay_files and os.path.isfile(file):
                    os.remove(file)


if __name__ == "__main__":
    w = Watcher(directory, MyHandler())
    w.run()
