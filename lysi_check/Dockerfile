FROM python:3.9-slim
#FROM python:latest

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY favicon_check.ico ./
COPY lysi_check_text.png ./
COPY lysi_check.py ./

EXPOSE 8501

RUN mkdir ./penmon 
COPY penmon ./penmon
RUN mkdir ~/.streamlit
COPY config.toml ~/.streamlit/

CMD ["streamlit","run","./lysi_check.py","--server.port=8501","--server.address=0.0.0.0"]