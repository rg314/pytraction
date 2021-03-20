#!/bin/sh

# data_20210320.zip

if [ ! -d "data" ]; then
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1DsPuqAzI7CEH-0QN-DWHdnF6-to5HdFe' -O data.zip && \
    unzip data.zip                                  && \
    rm data.zip
fi