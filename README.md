# Dysweep

![PyPI](https://img.shields.io/badge/PyPI-Compatible-green?style=for-the-badge&logo=PyPI)

Dysweep is an innovative Python library designed to extend and enhance the functionalities of the [Weights and Biases (WandB) sweep library](https://docs.wandb.ai/guides/sweeps). Dysweep is built with the belief that an entire experiment should be executable through a configuration dictionary, whether it's formatted as a YAML or JSON file.

## Features

Dysweep introduces two major enhancements:

### Checkpointing for the Sweep Server

Dysweep introduces checkpointing for the sweep server, which becomes extremely beneficial when dealing with preemptions or specific bugs that can interrupt the sweep process. This feature ensures that even if a sweep is running on a machine that may preempt the tasks or if certain configurations encounter specific bugs, the sweep process can resume from a checkpoint directory. Unlike the original WandB sweep, where a lost configuration is ignored by the WandB agent function, Dysweep overlays an API on top of this agent function, thereby enabling certain runs to resume. This is especially useful when only a small fraction of runs fail, thus eliminating the need to re-run the entire sweep.

### Running Sweeps Over Hierarchies

Perhaps the most significant capability of Dysweep is its ability to run sweeps over hierarchically structured parameters. The original WandB sweep configuration is limited to flat parameter sets. However, deep learning experiments often demand more complex, nested sets of configurations. Dysweep enables this, effectively eliminating the need for hard-coding the selection between different classes with primitive methods. Instead, you can define a new YAML that automatically selects between class types and initialization arguments, streamlining the setup process and making it more robust.

Dysweep is inspired by [DyPy](https://github.com/vahidzee/dypy), a library used for deep learning experimentation, and mirrors its vision of facilitating fully generic configuration YAML files that encapsulate code snippets. Hence, Dysweep offers a versatile configuration set that empowers you to define experiments at any layer of abstraction.

## Applications

Dysweep is envisioned to particularly facilitate the following applications:

1. **Large Scale Hyper Parameter Tuning**: Dysweep is geared towards conducting large scale hyper parameter tuning not just within the confines of a specific model, but across various models and methods. This functionality paves the way for a more comprehensive and detailed study of the effects of hyperparameters.

2. **Running Models Over Different Configurations and Datasets**: Once a model is ready, Dysweep enables it to run over a multitude of configurations and datasets. It provides a systematic way to define a sweep in WandB, allowing every experiment to run in parallel across different machines. This significantly eases the process of large-scale computing and data gathering for a particular model.

## Installation

You can install the Dysweep library through [PyPi](https://pypi.org/project/dysweep/) using the following command:

```shell
pip install dysweep
```

## Tutorial and Use-Cases

We selected a standard task in deep learning - image classification - and utilized various convolutional models and datasets to demonstrate the broad capabilities of Dysweep. We subjected this problem to multiple configurations through our pipeline. For a hands-on understanding of the process, you can refer to our detailed Jupyter notebook available [here](./tutorials/image_classification.ipynb).

