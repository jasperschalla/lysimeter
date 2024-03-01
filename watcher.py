#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File     :watcher.py
@Time     :2023/11/15 14:40:32
@Author   :Jasper Schalla
@Contact  :jasper.schalla@web.de
"""

import time

# from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
import logging
import pathlib
import shutil
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

# Configure Observer

directory_name = "watch"
directory = f"./{directory_name}/"


class Watcher:
    def __init__(self, directory, handler=FileSystemEventHandler()):
        self.observer = Observer()
        self.handler = handler
        self.directory = directory

    def run(self):
        self.observer.schedule(self.handler, self.directory, recursive=True)
        self.observer.start()
        logger.info("watcher running in {}".format(self.directory))
        logger.info("-------------------------------------------------------------")
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
        self.observer.join()
        logger.warning("watcher terminated")


# Configure handler


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            file = pathlib.Path(event.src_path)
            filepath_start = str(file).split(directory_name)[0]
            filepath_end = str(file).split(directory_name)[-1].split("/")[-1]
            filepath = filepath_start + directory_name + "/" + filepath_end
            extension = str(file).split(".")[-1]
            # logger.info(event.src_path)

            if extension == "DBD":
                location = (
                    str(event.src_path).split(f"{directory_name}/")[-1].split("/")[0]
                )
                shutil.copy(str(file), str(filepath).replace("watch", "original"))
                process = subprocess.Popen(f"bash convert.sh {location}", shell=True)
                process.wait()


if __name__ == "__main__":
    w = Watcher(directory, MyHandler())
    w.run()
