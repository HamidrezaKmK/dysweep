"""
This script contains all the scripts that are accessible in the command-line for the package. Namely,

1. Creating a sweep using a yaml configuration file directly.

2. Running configurations on specific machines or resuming using the command line directly.
"""
from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.actions import ActionConfigFile
from dysweep import dysweep_run_resume, ResumableSweepConfig
import importlib
import sys

def create_sweep():
    """
    This is a function designed to run using the command-line to simplify the creation of a sweep.
    
    You can define a configuration using a configuration file `config.yaml` and then run this function using the command-line.
    
    ```bash
    create_sweep -c config.yaml
    ```
    
    The configuration file is a standard ResumanbleSweepConfig configuration file that contains the following fields:
    
    1. `base_config`: The base configuration that is being used to create the sweep.
    
    2. `sweep_configuration`: The hierarchical configuration used to update and insert (upsert) 
        the base configuration.
    
    3. `project`: The name of the project that is being used to create the sweep.
    
    4. `entity`: The WandB entity that is being used to create the sweep.
    
    It also includes additional information from `ResumanbleSweepConfig` that you can check out.
    
    An example:
    
    ```bash
    create_sweep --config config.yaml --project <my_project> --entity <my_entity>
    ```
    
    After running the function, the sweep identifier would be displayed which can be used across 
    multiple machines to run the sweep configurations.
    """
    
    parser = ArgumentParser()
    parser.add_class_arguments(
        ResumableSweepConfig,
        fail_untyped=False,
        sub_configs=True,
    )
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    args = parser.parse_args()

    dysweep_run_resume(args)

def run_resume_sweep():
    """
    Using this function, you can run or resume a sweep run using the command line.
    You have to simply define your configuration in a yaml file and use the `--config` argument to
    load the configuration using `jsonargparse`.
    
    After obtaining a specific configuration and a checkpoing directory, these will be called on a
    particular function you have implemented. If for example, you have a function `main` in a file
    denoted by `path.to.my.package`, then you can run the following command:
    
    ```bash
    run_resume_sweep --config config.yaml --package path.to.my.package --function main
    ```
    
    """
    parser = ArgumentParser()
    parser.add_class_arguments(
        ResumableSweepConfig,
        fail_untyped=False,
        sub_configs=True,
    )
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    
    parser.add_argument(
        "-p", "--package", type=str, help="Path to a package containing a main function."
    )
    parser.add_argument(
        "-f", "--function", type=str, help="Name of the main function."
    )
    
    args = parser.parse_args()
    # print the root of importlib
    
    # set the root directory of importlib the same as the directory that is running this function
    sys.path.append('.')
    
    # from args.package import args.function using importlib
    module = importlib.import_module(args.package)
    func = getattr(module, args.function)
    
    # Remove args.package and args.function from args
    delattr(args, "package")
    delattr(args, "function")
    
    # Run and resume using the arguments and function
    dysweep_run_resume(args, func)
   