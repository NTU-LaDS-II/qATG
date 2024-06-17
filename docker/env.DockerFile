FROM ubuntu:22.04
LABEL maintainer="NTU-LADS-II"

RUN apt update
RUN apt install -y python3.10 python3-pip
RUN pip3 install qatg