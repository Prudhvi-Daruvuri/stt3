FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as gpu

WORKDIR /app

RUN apt-get update -y && \
  apt-get install -y python3 python3-pip libcudnn8 libcudnn8-dev libcublas-12-4 portaudio19-dev git

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

# Install NeMo toolkit for Parakeet backend support
RUN pip3 install nemo_toolkit[asr]==2.3.0

COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install -r /app/requirements-gpu.txt

# Copy the enhanced RealtimeSTT with multi-backend support
COPY RealtimeSTT /app/RealtimeSTT
COPY RealtimeSTT_server /app/RealtimeSTT_server

# Expose ports for WebSocket control and data connections
EXPOSE 8011 8012

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV KMP_DUPLICATE_LIB_OK "TRUE"

# Set working directory to server folder
WORKDIR /app/RealtimeSTT_server

# Default command runs faster_whisper backend, can be overridden
CMD ["python3", "stt_server.py", "--backend", "faster_whisper"]

# --------------------------------------------

FROM ubuntu:22.04 as cpu

WORKDIR /app

RUN apt-get update -y && \
  apt-get install -y python3 python3-pip portaudio19-dev git

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy the enhanced RealtimeSTT with multi-backend support
COPY RealtimeSTT /app/RealtimeSTT
COPY RealtimeSTT_server /app/RealtimeSTT_server

# Expose ports for WebSocket control and data connections
EXPOSE 8011 8012

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV KMP_DUPLICATE_LIB_OK "TRUE"

# Set working directory to server folder
WORKDIR /app/RealtimeSTT_server

# Default command runs faster_whisper backend (CPU only for this image)
CMD ["python3", "stt_server.py", "--backend", "faster_whisper", "--device", "cpu"]
