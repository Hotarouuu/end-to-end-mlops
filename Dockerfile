FROM python:3.11

WORKDIR /app

RUN mkdir -p /app/logs

ENV CONFIG=./config/model1.yaml

COPY . /app

RUN pip install .
