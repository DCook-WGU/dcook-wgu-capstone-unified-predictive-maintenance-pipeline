# Base image: plain Ubuntu
FROM ubuntu:22.04

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# 1. System tools required for Miniconda and building Python deps
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# 2. Install Miniconda manually under /opt/miniconda
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

WORKDIR /opt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH for all subsequent steps
ENV PATH="/opt/miniconda/bin:${PATH}"

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# 3. Build args for environment name and env file path
#    These are passed from docker-compose:
#      ENV_NAME: capstone
#      ENV_FILE: /app/environment.yml
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

ARG ENV_NAME=capstone
ARG ENV_FILE=/app/environment.yml

# Expose env name at runtime (used by entrypoint.sh)
ENV ENV_NAME=${ENV_NAME}

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# 4. Copy only environment.yml first, so env build is cached
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Build context is ./app, so this copies app/environment.yml into /app
COPY environment.yml /app/environment.yml

# Accept Anaconda TOS (needed for some conda distributions)
# and create the conda environment from environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create --name "${ENV_NAME}" -f "${ENV_FILE}" && \
    conda clean -afy

# Use bash with conda available for subsequent RUN commands
SHELL ["/bin/bash", "-lc"]

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# 5. Copy the full app source code
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

WORKDIR /app

# This copies everything from ./app (build context) into /app in the image
COPY . .

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# 6. Export lockfiles (environment + pip requirements)
#    This is nice for reproducibility and your capstone writeup.
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

RUN conda env export --name "${ENV_NAME}" > /app/environment.lock.yml && \
    conda run -n "${ENV_NAME}" pip freeze > /app/requirements.txt

# Make /app importable (so src/ modules resolve cleanly)
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Initialize conda for interactive shells
RUN conda init bash

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# 7. Entrypoint
#    entrypoint.sh should typically:
#      - source conda
#      - activate $ENV_NAME
#      - exec "$@"
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash", "-l"]