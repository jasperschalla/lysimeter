FROM python:3.9-slim
#FROM python:latest

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY favicon_pipe.ico ./
COPY lysi_pipe_text.png ./
COPY lysi_pipe.py ./

EXPOSE 8502

RUN mkdir ~/.streamlit
COPY config.toml ~/.streamlit/

CMD ["streamlit","run","./lysi_pipe.py","--server.port=8502","--server.address=0.0.0.0"]