#!/bin/bash
#Set the base image
FROM python:3.7-stretch

ARG city
ENV CITY_ENV=${city}
ARG n_samples
ENV SAMPLES_ENV=${n_samples}
ARG n_epochs
ENV EPOCHS_ENV=${n_epochs}

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt


#creates work dir   

#copy python script to the container folder app
COPY LSTM.py /app/LSTM.py
COPY 9958.csv /app/9958.csv
RUN chmod +x /app/LSTM.py


CMD python /app/LSTM.py $CITY_ENV $SAMPLES_ENV $EPOCHS_ENV
