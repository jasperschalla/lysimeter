FROM python:3.11

WORKDIR /usr/app/src

#RUN addgroup --gid 917 imk_ifu_tereno
#RUN adduser --disabled-password --gecos '' --uid 15223 --gid 917 s_lysipipe

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

#USER s_lysipipe

CMD ["python","./watcher.py"]