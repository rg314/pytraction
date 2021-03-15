#!/bin/sh



if [ ! -d "data" ]; then
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=12hfScdSItJUS6nEua3LjHFaazew8SQa2' -O data.zip  && \
    unzip data.zip                                  && \
    rm data.zip
fi




