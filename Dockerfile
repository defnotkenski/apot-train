FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

COPY . .

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3.10 python3-pip
RUN apt-get install -y nvidia-container-toolkit

RUN pip3.10 install --no-cache-dir -r requirements.txt
