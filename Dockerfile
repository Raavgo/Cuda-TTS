FROM nvidia/cuda:11.5.1-devel-ubuntu18.04

#Get Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		netbase \
		wget \
        git \
        libsndfile1-dev \
        python3.8-dev

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN mkdir -p model/phoneme_cache\
    && mkdir -p output 

RUN git clone https://github.com/coqui-ai/TTS
RUN cd TTS && make install

#Layer that change frequently
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./dorothy_and_wizard_oz data
COPY train.py /TTS/TTS/bin/

CMD CUDA_VISIBLE_DEVICES="0" python /TTS/TTS/bin/train.py
