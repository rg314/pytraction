FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime


RUN apt-get update
RUN apt install -y git


RUN mkdir pytraction
WORKDIR /pytraction
ENV HOME /pytraction

RUN git clone https://github.com/rg314/pytraction.git
RUN pip install pip install pytraction/


USER root
LABEL name={NAME}
LABEL version={VERSION}
