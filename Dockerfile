# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory
WORKDIR /app

COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# INSTRUCTIONS FOR BUILDING:
# 1. Clone the CataPro repository to your local machine:
#    git clone https://github.com/zchwang/CataPro.git
#
# 2. Place this Dockerfile in the root of your cloned 'CataPro' directory.
#
# 3. Run the docker build command from the root of the 'CataPro' directory:
#    docker build -t catapro_image .
#    (Note: The models are NOT copied into the image. They will be mounted at runtime.)

# Copy only the application code into the image
COPY . .

# Set the working directory for inference
WORKDIR /app/inference

# Set the entrypoint to the prediction script
ENTRYPOINT ["python", "predict.py"]

# The container is now ready.
# See the instructions on how to run it with external models mounted as volumes.
