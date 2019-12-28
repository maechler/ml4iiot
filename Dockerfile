FROM python:3.6

WORKDIR /usr/src/ml4iiot

COPY ml4iiot .
COPY requirements.txt .
COPY setup.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

ENV ML4IIOT_RUNNER_CONFIG_FILE_PATH=config/runner_config.yaml

CMD [ "sh", "-c", "python ./ml4iiot/cli_runner.py -c ${ML4IIOT_RUNNER_CONFIG_FILE_PATH}" ]