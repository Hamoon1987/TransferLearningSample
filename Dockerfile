# Specify the base image
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

LABEL description="Setting a development Docker Container"
LABEL author="Hamoon"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y build-essential python3 python3-pip python-dev sudo
RUN apt-get -y install git
RUN mkdir -p /tmp
COPY requirements.txt /tmp/
RUN pip3 -q install pip --upgrade
RUN pip3 install -r /tmp/requirements.txt