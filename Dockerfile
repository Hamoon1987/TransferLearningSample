# Specify the base image
FROM ubuntu:18.04 as base

LABEL description="Setting a development Docker Container"
LABEL author="Hamoon"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y build-essential python3 python3-pip python-dev sudo
RUN apt-get -y install git
RUN mkdir -p /TransferLearningSample
COPY . /TransferLearningSample
WORKDIR /TransferLearningSample


RUN pip3 -q install pip --upgrade
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/jaddoescad/ants_and_bees.git
