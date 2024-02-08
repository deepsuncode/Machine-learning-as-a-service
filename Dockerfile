FROM --platform=arm64 jupyter/base-notebook:python-3.8.6

WORKDIR /home/jovyan/work

COPY code/requirements.txt .

RUN python -m pip install --no-cache -r requirements.txt
