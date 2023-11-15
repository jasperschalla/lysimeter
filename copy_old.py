#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :copy_old.py
@Time     :2023/11/15 14:39:47
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""

import shutil
import os
import datetime
import re
import time

src_path = "./real/"
dest_path = "./processing/watch/"

locations = [
    location for location in os.listdir(src_path) if not location == ".DS_Store"
]
locations = ["FE1", "FE3", "FE2"]

for location in locations:
    location_files = [
        file
        for file in os.listdir(os.path.join(src_path, location))
        if not file == ".DS_Store"
    ]

    file_order = []

    for file_path in location_files:
        file_date_groups = re.search("^\\d+_(\\d{8})T.*", file_path)
        file_date = datetime.datetime.strptime(file_date_groups.group(1), "%Y%m%d")
        file_order.append(file_date)
    file_order.sort()

    files = []
    for file_path in file_order:
        date_str = datetime.datetime.strftime(file_path, "%Y%m%d")
        file_name = [i for i in location_files if re.match(f".*{date_str}.*", i)][0]
        files.append(file_name)

    for file in files:
        shutil.copy(
            os.path.join(src_path, location, file),
            os.path.join(dest_path, location, file),
        )

    time.sleep(1)
