#!/bin/sh


if [ ! -d "data" ]; then
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=19-_UZ3CQmhtiiYQEjOihOY0hWtNzJawx' -O data.zip  && \
    unzip data.zip                                  && \
    rm data.zip
fi



