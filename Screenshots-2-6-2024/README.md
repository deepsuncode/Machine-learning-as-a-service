# CHANGELOG (3rd Feb 2024)

## System configuration

* Hardware - Apple M2 Pro Chip
* OS - MacOS 14.3
* Python version - v3.8.18
* Pip - v23.0.1

## Docker

Added docker file - because I wasn't able to successfully run code in `mlass_train.py` which uses sklearn_extensions package. As this package is not actively maintained and uses much older version

### Steps to run the code in docker

1. Build the image - `docker image build -t mlaas .`
2. Start the container - `docker container run --rm -it -v .:/home/jovyan/work mlaas sh`
3. CD into code directory - `cd code`
4. Start training model - `python mlaas_train.py`
5. Start testing model - `python mlaas_test.py`
