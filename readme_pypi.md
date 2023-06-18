# Dysweep

Dysweep is a Python library enhancing the functionalities of the [Weights and Biases sweep library](https://docs.wandb.ai/guides/sweeps). It allows entire experiments to be executed using a configuration dictionary (YAML/JSON).

## Features

- **Checkpointing for the Sweep Server**: Dysweep introduces checkpointing that allows resuming certain runs, useful when only a small fraction of runs fail, eliminating the need to re-run the entire sweep.

- **Running Sweeps Over Hierarchies**: Dysweep supports hierarchically structured parameters, thereby eliminating the need for hard-coding the selection between different classes.

Dysweep is inspired by [DyPy](https://github.com/vahidzee/dypy), offering a versatile configuration set that empowers defining experiments at any layer of abstraction.

## Applications

Dysweep aids in large-scale hyperparameter tuning across various models/methods and running models over different configurations and datasets. It provides a systematic way to define a sweep in WandB, allowing parallel execution of experiments.

