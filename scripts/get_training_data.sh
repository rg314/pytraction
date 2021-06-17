#!/bin/sh

if [ ! -d "data" ]; then
  mkdir data
fi

cd data                            && \

if [ ! -d "BF-C2DL-HSC" ]; then
    curl -o BF-C2DL-HSC.zip -k https://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip && \
    unzip BF-C2DL-HSC.zip                                  && \
    rm BF-C2DL-HSC.zip
fi


if [ ! -d "BF-C2DL-MuSC" ]; then
    curl -o BF-C2DL-MuSC.zip -k https://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip && \
    unzip BF-C2DL-MuSC.zip                                  && \
    rm BF-C2DL-MuSC.zip
fi


if [ ! -d "DIC-C2DH-HeLa" ]; then
    curl -o DIC-C2DH-HeLa.zip -k https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip && \
    unzip DIC-C2DH-HeLa.zip                                  && \
    rm DIC-C2DH-HeLa.zip
fi



if [ ! -d "Fluo-C2DL-Huh7" ]; then
    curl -o Fluo-C2DL-Huh7.zip -k https://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip && \
    unzip Fluo-C2DL-Huh7.zip                                  && \
    rm Fluo-C2DL-Huh7.zip
fi


if [ ! -d "Fluo-C2DL-MSC" ]; then
    curl -o Fluo-C2DL-MSC.zip -k https://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip && \
    unzip Fluo-C2DL-MSC.zip                                  && \
    rm Fluo-C2DL-MSC.zip
fi


if [ ! -d "Fluo-N2DH-GOWT1" ]; then
    curl -o Fluo-N2DH-GOWT1.zip -k https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip && \
    unzip Fluo-N2DH-GOWT1.zip                                  && \
    rm Fluo-N2DH-GOWT1.zip
fi


if [ ! -d "Fluo-N2DL-HeLa" ]; then
    curl -o Fluo-N2DL-HeLa.zip -k https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip && \
    unzip Fluo-N2DL-HeLa.zip                                  && \
    rm Fluo-N2DL-HeLa.zip
fi

if [ ! -d "PhC-C2DH-U373" ]; then
    curl -o PhC-C2DH-U373.zip -k https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip && \
    unzip PhC-C2DH-U373.zip                                  && \
    rm PhC-C2DH-U373.zip
fi

if [ ! -d "PhC-C2DL-PSC" ]; then
    curl -o PhC-C2DL-PSC.zip -k https://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip && \
    unzip PhC-C2DL-PSC.zip                                  && \
    rm PhC-C2DL-PSC.zip
fi


if [ ! -d "Fluo-N2DH-SIM" ]; then
    curl -o Fluo-N2DH-SIM.zip -k https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip && \
    unzip Fluo-N2DH-SIM.zip                                  && \
    rm Fluo-N2DH-SIM.zip
    mv Fluo-N2DH-SIM+ Fluo-N2DH-SIM
fi


cd ..
