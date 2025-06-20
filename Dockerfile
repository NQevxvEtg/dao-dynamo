# Stage 1: Build environment
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS build-env

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies and Python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common git wget build-essential ninja-build && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv gdb && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3 -m pip install -U pip

# Install torch stack
RUN python3 -m pip install torch torchvision torchaudio

# --- BUILD AND PREPARE LIBRARIES ---
# 1. Install causal-conv1d dependency
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d && \
    python3 -m pip install /tmp/causal-conv1d && \
    rm -rf /tmp/causal-conv1d

# 2. Prepare the full BlackMamba source, with patches and compiled extensions
RUN git clone https://github.com/Zyphra/BlackMamba.git /app/blackmamba_src
WORKDIR /app/blackmamba_src
# Apply patches to the source
COPY patches/selective_scan_interface.py /app/blackmamba_src/ops/selective_scan_interface.py
RUN sed -i 's/model_state_dict = loaded\["model"\]/model_state_dict = loaded.get("model", loaded)/g' /app/blackmamba_src/mamba_model.py
# Use pip to compile the extensions in place and install dependencies
RUN python3 -m pip install .

# Install pip-tools for compiling requirements
RUN python3 -m pip install pip-tools

# 3. Install other user requirements
WORKDIR /app
COPY requirements.in .
RUN pip-compile --output-file requirements.txt requirements.in && \
    python3 -m pip install -r requirements.txt

# ---- Final Image ----
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS final

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python 3.11 runtime
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev gdb && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy installed packages from build-env
COPY --from=build-env /usr/local/lib/python3.11/dist-packages/ /usr/local/lib/python3.11/dist-packages/
COPY --from=build-env /usr/local/bin /usr/local/bin

# Copy the entire BlackMamba source tree (which includes the compiled extensions)
COPY --from=build-env /app/blackmamba_src /app/blackmamba_src

# Set the PYTHONPATH to find the BlackMamba source files
ENV PYTHONPATH=/app/blackmamba_src:${PYTHONPATH}

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]