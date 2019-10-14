# Machine Learning 4 IIoT

## Development

### Setup virtualenv

```
pip install virtualenv;
python -m virtualenv env;
source env/bin/activate;
```

### Install dependencies

```
pip install -r requirements.txt;
pip install -e .;
```

### Testing

Run unit tests with: `python -m unittest discover tests/*`

### Leave virtualenv

- `deactivate`

## CLI runner

```
python ml4iiot/cli_runner.py -c config/your_config.yaml
```

## Performance

Resample the training data in advance and use integer timestamps instead of formatted date strings to speed up trainings. 
Use the following commands to profile your code:

```
python -m cProfile -o out/cli_runner.profile ml4iiot/cli_runner.py -c config/your_config.yaml
```

```
snakeviz out/cli_runner.profile 
```

## Docker

TODO

## Pipeline configuration

A pipeline always consists of an input adapter, an output adapter as well as an algorithm in between. 

```yaml
pipeline:
  input:
    class: ml4iiot.input.csvinput.CsvInput
    config:
      windowing_strategy:
        class: ml4iiot.input.windowing.timebased.TimeBasedWindowingStrategy
        config:
          window_size: 720s
          stride_size: 360s
          batch_size: 1000
          resample:
            enabled: True
            target_sampling_rate: 10s
            method: interpolate
            interpolation_method: linear
      delimiter: ','
      csv_file: /some/path/to/your/data.csv
      index_column: datetime
      columns:
        datetime:
          type: datetime
          datetime_format: 'iso'
        sensor_a_value: float
        sensor_b_value: float
  output:
    class: ml4iiot.output.stdoutput.StdOutput
  algorithm:
    class: ml4iiot.algorithms.stochastic.average.Average
    config:
      columns:
        - sensor_a_value
        - sensor_b_value
```

### Inputs
- [ml4iiot.input.csvinput.CsvInput](#CSVInput) 
- [ml4iiot.input.kafkainput.KafkaInput](#KafkaInput)

#### Windowing strategies

- [ml4iiot.input.timebased.TimeBasedWindowingStrategy](#TimeBasedWindowingStrategy) 
- [ml4iiot.input.countbased.CountBasedWindowingStrategy](#CountBasedWindowingStrategy) 

### Outputs
- [ml4iiot.output.compoundoutput.CompoundOutput](#CompoundOutput)
- [ml4iiot.output.stdoutput.StdOutput](#StdOutput)
- [ml4iiot.output.plotoutput.PlotOutput](#PlotOutput)
- [ml4iiot.output.metricoutput.MetricOutput](#MetricOutput)
- [ml4iiot.output.kafkaoutput.KafkaOutput](#KafkaOutput)

### Algorithms
- [ml4iiot.algorithms.stochastic.average.Average](#Average)
- [ml4iiot.algorithms.stochastic.average_low_high_pass.AverageLowHighPass](#AverageLowHighPass)
- [ml4iiot.algorithms.stochastic.mad.Mad](#Mad)
- [ml4iiot.algorithms.autoencoder.shallowautoencoder.ShallowAutoencoder](#ShallowAutoencoder)

### CSVInput

### KafkaInput  
