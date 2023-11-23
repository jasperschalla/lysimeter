FROM python:latest

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY watcher.py ./
COPY convert.sh ./
COPY dump_dbd.cpp ./
COPY preprocess.py ./
COPY postprocess.py ./
COPY resolve_pipe_errors.py ./
COPY resolve_pipe_errors.sh ./
COPY resolve_post_errors.py ./
COPY resolve_date_errors.py ./

RUN mkdir ./original ./dumped ./dumped_head ./raw

CMD ["python","./watcher.py"]