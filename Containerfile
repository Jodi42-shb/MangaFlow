FROM docker.io/nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- System + Python 3.11 ----
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils \
    git wget \
    tesseract-ocr \
    ffmpeg \
    poppler-utils \
    imagemagick \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# make python3 point to 3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python3

# ---- App ----
WORKDIR /app
COPY . /app

RUN python3 -m ensurepip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r req.txt

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

