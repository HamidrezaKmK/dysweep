"""
This script contains all the scripts that are accessible in the command-line for the package. Namely,

1. Creating a sweep using a yaml configuration file directly.

2. Running configurations on specific machines or resuming using the command line directly.
"""
from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.actions import ActionConfigFile
from jsonargparse.actions import Action
from dysweep import dysweep_run_resume, ResumableSweepConfig
import importlib
import sys
import functools

class CustomAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        
def create_sweep():
    """
    This is a function designed to run using the command-line to simplify the creation of a sweep.
    
    You can define a configuration using a configuration file `config.yaml` and then run this function using the command-line.
    
    ```bash
    dysweep_create -c config.yaml
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
    dysweep_create --config config.yaml --project <my_project> --entity <my_entity>
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
    
    sys.path.append('.')
    
    dysweep_run_resume(args)


def parse_dict(value):
    if isinstance(value, dict):
        return value
    elif isinstance(value, str):
        try:
            items = value.split()
            parsed_dict = {}
            for item in items:
                key, val = item.split(':')
                try:
                    val = int(val)
                except ValueError:
                    pass
                parsed_dict[key] = val
            return parsed_dict
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid dictionary format. Use key:value pairs separated by spaces.")
    else:
        raise ValueError("invalid type of parseable object:", type(value))

def run_resume_sweep():
    """
    Using this function, you can run or resume a sweep run using the command line.
    You have to simply define your configuration in a yaml file and use the `--config` argument to
    load the configuration using `jsonargparse`.
    
    After obtaining a specific configuration and a checkpoing directory, these will be called on a
    particular function you have implemented. If for example, you have a function `main` in a file
    denoted by `path.to.my.package`, then you can run the following command:
    
    ```bash
    dysweep_run_resume --config config.yaml --package path.to.my.package --function main
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
    

    parser.add_argument('--run_additional_args', action=CustomAction, type=parse_dict, nargs='+')

    args = parser.parse_args()
    run_args = {}
    if hasattr(args, 'run_additional_args') and args.run_additional_args:
        for run_arg_ in args.run_additional_args:
            run_args.update(run_arg_)
        delattr(args, 'run_additional_args')

    
    # set the root directory of importlib the same as the directory that is running this function
    sys.path.append('.')
    
    # from args.package import args.function using importlib
    module = importlib.import_module(args.package)
    func = getattr(module, args.function)
    func = functools.partial(func, **run_args)

        
    # Remove args.package and args.function from args
    delattr(args, "package")
    delattr(args, "function")
    
    # Run and resume using the arguments and function
    dysweep_run_resume(args, func)
   
