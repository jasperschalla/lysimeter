FROM python:latest

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["python","./awat_watcher.py"]