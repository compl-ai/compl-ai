FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04

# Such that we don't get prompted for the geographic location
ARG DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA to get Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa

# Update package lists again and install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

COPY ./ /app/

RUN apt-get update && \ 
	apt-get install -y unzip \
	curl \
	git \
	python3.10 \
    wget \
	pip && \
	rm -rf /var/lib/apt/lists/*


RUN git submodule update --init --recursive

RUN pip3 install poetry==1.7.1 gdown
RUN poetry install --no-root 

# This will disable poetry to take the already existing .venv folder as the virtual environment
RUN poetry config virtualenvs.in-project false

RUN gdown https://drive.google.com/uc?id=1eXsu20ycISCrZJa8vd1PvbU7u8dyHCeu
RUN unzip benchmark_data.zip && rm benchmark_data.zip

