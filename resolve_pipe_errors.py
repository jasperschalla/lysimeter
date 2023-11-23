#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :resolve_errors.py
@Time     :2023/11/15 14:40:16
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""

import os
import re
import subprocess
import shutil

src_path = "./"

errors = []

with open(os.path.join(src_path, "lysi_data.log"), "r") as f:
    lines = f.readlines()
    for line in lines:
        if re.match(".*ERROR.*", line):
            line_relevant = line.split(":")[-1]
            line_groups = re.search(".*file '(.*).csv'.*location=(.*)$", line_relevant)
            file = line_groups.group(1)
            location = line_groups.group(2)

            errors.append((f"{file}.DBD", location))

for file, location in errors:
    file_src_path = os.path.join(src_path, "watch", location, file)
    file_dest_path = os.path.join(src_path, "original", file)
    shutil.copy(file_src_path, file_dest_path)
    process = subprocess.Popen(f"bash resolve_errors.sh {location} {file}", shell=True)
    process.wait()
