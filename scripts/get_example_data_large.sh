#!/bin/sh

# data_20210315.zip

if [ ! -d "data" ]; then
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1ZJT2aE3JJDOkAAUYKVMSzmZY483KGbVi' -O data.zip  && \
    unzip data.zip                                  && \
    rm data.zip
fi




