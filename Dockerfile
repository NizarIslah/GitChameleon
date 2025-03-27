FROM ubuntu:20.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    PYTHONPATH="/app"


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libffi-dev \ 
    libsqlite3-dev \
    gfortran \           
    libopenblas-dev \ 
    xz-utils && rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN curl -fsSL https://pyenv.run | bash

# Add pyenv initialization to the shell environment
RUN echo 'eval "$(pyenv init --path)"' >> /root/.profile
RUN echo 'eval "$(pyenv init -)"' >> /root/.bashrc

RUN /bin/bash -c "source /root/.profile && pyenv install 3.9.19"
RUN /bin/bash -c "source /root/.profile && pyenv install 3.10.14"
RUN /bin/bash -c "source /root/.profile && pyenv install 3.7.17"

WORKDIR /app

ENTRYPOINT ["/bin/bash"]