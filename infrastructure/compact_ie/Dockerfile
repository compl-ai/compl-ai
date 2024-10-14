FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && \ 
	apt-get install -y wget \
	unzip && \
	rm -rf /var/lib/apt/lists/*


RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Copy the contents of the CompactIE module into the container
COPY external/ChatProtect/ /app/ChatProtect

WORKDIR /app/ChatProtect/
RUN unzip -o CompactIE.zip -d CompactIE
WORKDIR /app/ChatProtect/CompactIE


# Download model weights
RUN wget https://zenodo.org/record/6804440/files/ce_model?download=1 && \
	mv ce_model?download=1 save_results/models/constituent/ce_model && \
	wget https://zenodo.org/record/6804440/files/cl_model?download=1 && \
	mv cl_model?download=1 save_results/models/relation/cl_model

# Create a new Conda environment and activate it
RUN conda create -y -n CompactIE pip python=3.6 pytorch=1.9.0 -c pytorch

# Install Python dependencies
RUN conda run -n CompactIE pip install transformers==4.2.2 configargparse==1.2.3 bidict==0.20.0 PyYAML==6.0.1

# Expose port
EXPOSE 39881

# Set the default command to run when the container starts
CMD ["conda", "run", "-n", "CompactIE", "python", "api.py", "--config_file", "config.yml"]
