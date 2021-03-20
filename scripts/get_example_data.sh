#!/bin/sh

# data_20210320.zip

if [ ! -d "data" ]; then
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1cboy6pBr8PQW1PqEyRl3lDSjGUPMFoTN' -O data.zip  && \
    unzip data.zip                                  && \
    rm data.zip
fi
