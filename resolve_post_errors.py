import os
import subprocess
import re
from datetime import datetime, timedelta

src_path = "./"

locations = [
    folder
    for folder in os.listdir(os.path.join(src_path, "excel"))
    if not folder == ".DS_Store"
]


for location in locations:
    location_path = os.path.join(src_path, "excel", location)
    files = [file for file in os.listdir(location_path) if not file == ".DS_Store"]
    for file in files:
        file_date = datetime.strftime(
            datetime.strptime(re.search(".*_(\\d{8}).*", file).group(1), "%Y%m%d")
            - timedelta(days=1),
            "%Y_%m_%d",
        )
        file_preprocessed = [
            re.search("\\d{1}_(\\d{4}_\\d{2}_\\d{2}).*", file).group(1)
            for file in os.listdir(
                os.path.join(src_path, "additional_filled", location)
            )
            if not file == ".DS_Store"
        ]

        if file_date not in file_preprocessed:
            print(file_date, location)
            process = subprocess.Popen(
                [
                    f"python postprocess.py {os.path.join(location_path,file)} {location}"
                ],
                shell=True,
            )
            process.wait()
