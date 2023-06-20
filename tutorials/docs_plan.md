# Docs plan

Do an up-to-down approach, whereby you first introduce the function `dysweep_run_resume` which does one of the following:

## Explain the base_config argument

We will run through a basic image classification task throughout the documentation. Assume 
This is straight-forward. Explain the simple version of the example in this section. 
Dysweep uses the following abstraction:
start off with a base configuration, now upsert and tweak all the different parts of the configuration.
(Explain the philosophy a little bit)

## The Sweep Configuration
Only provide a high-level. Start-off by not introducing all the additional hierarchical features. 
Simply explain a flat sweep configuration example.


## Explain how to define sweep configurations

Start-off by introducing a simple `sweep=True` with the alias and values.

Link to an advanced tutorial on how to define sweep configurations.

### Aliases

### Values

### Upsert
Explain the `dy_upsert` function and explain why it is needed.

### (Advanced) List operations
### (Advanced) Dynamic YAMLs
Following the vision of the DyPy library, we would like our YAML files to be informative, but as generic as possible. The would give us full-flexibility to implement any changes needed in the configurations layer instead of the functions layer.

Explain the simple `dy_eval(function)`.

Then explain the advanced `dy_eval` also the more advanced:
```YAML
dy_eval:
    expression: |
        import numpy as np

        def pow2(x):
            return x ** 2

        def func(configuration_until_now: dict):
            ranged = np.arange(10)
            mult_by_ind = ranged * np.linspace(0, 1, 10)
            return pow2(mult_by_ind)

    function_of_interest: func
```
Link to the `DyPy` Library here.

Mention that a full example of the functionalities is explained in the Full example section.

## Creating a New Sweep

## Running an Agent

## Resuming Lost Runs

## Re-running

## Coupling with `jsonargparse`

## Full example

A classification task, on MNSIT digits as well as CIFA10. Use two different architectures that are pre-maid. 

Also include one architecture that is handmade and code everything in Pytorch.

For the dataset configurations, include tunable knobs for batch_size and standardization.

For trainer include tunable knobs for the
1. Number of steps
2. The optimizer being used along with the class and arguments.

For the custom architecture create a fully generic model. 




