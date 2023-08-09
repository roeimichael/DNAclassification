FROM nvidia/cuda:11.8-base

WORKDIR /usr/src/app

# Copy the local code to the container
COPY . .

# Install PyTorch, torchvision, and torchaudio with CUDA 11.8 support
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install any other necessary dependencies
RUN pip install -r requirements.txt

# Command to run your script
CMD ["python", "modelTraining.py"]
