import os
import re
import logging
from datetime import datetime
import subprocess

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

# List with locations
locations = ["FE1", "FE2", "FE3", "RB1", "RB2", "GW"]

for location in locations:
    # List all files of excel directory that have 'date_error' in their name
    files = [f for f in os.listdir(f"./excel/{location}") if re.match(".*error.*", f)]

    for file in files:
        filename = re.search(".*error_(.*)T.*", file).group(1)
        logger.info(
            f"./dumped/error_{filename}T000.csv and ./dumped_head/error_{filename}T000.csv was created for location={location}"
        )
        logger.info(
            f"./raw/I/error_{filename}T000.csv was created for location={location}"
        )
        logger.info(
            f"./raw/G/error_{filename}T000.csv was created for location={location}"
        )
        logger.info(
            f"./raw/H/error_{filename}T000.csv was created for location={location}"
        )
        logger.info(
            f"file 'error_{filename}T000.xlsx' was created for location={location}"
        )
        logger.info(
            f"file 'error_{filename}T000.csv's I schedule has been successfully written to influxdb for location={location}"
        )
        logger.info(
            f"file 'error_{filename}T000.csv's G schedule has been successfully written to influxdb for location={location}"
        )
        logger.info(
            f"file 'error_{filename}T000.csv's H schedule has been successfully written to influxdb for location={location}"
        )
        logger.info(
            f"file 'error_{filename}T000.csv's Flags schedule has been successfully written to influxdb for location={location}"
        )
        logger.info("------------------------------------------------------")

        new_filename = f"000_{filename}T000.xlsx"
        os.rename(f"./excel/{location}/{file}", f"./excel/{location}/{new_filename}")
        process = subprocess.Popen(
            [
                f"python postprocess.py {f'./excel/{location}/{new_filename}'} {location}"
            ],
            shell=True,
        )
        process.wait()
