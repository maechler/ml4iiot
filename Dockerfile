FROM python:3.6

WORKDIR /usr/src/ml4iiot

COPY ml4iiot /usr/src/ml4iiot/ml4iiot
COPY requirements.txt /usr/src/ml4iiot/requirements.txt
COPY setup.py /usr/src/ml4iiot/setup.py

RUN mkdir -p /usr/src/ml4iiot/config
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

ENV ML4IIOT_RUNNER_CONFIG_FILE_PATH=/usr/src/ml4iiot/config/runner_config.yaml

CMD [ "sh", "-c", "python /usr/src/ml4iiot/ml4iiot/cli_runner.py -c ${ML4IIOT_RUNNER_CONFIG_FILE_PATH}" ]