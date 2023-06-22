from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.actions import ActionConfigFile
from dysweep import dysweep_run_resume, ResumableSweepConfig
import importlib
import sys

def create_sweep():
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
   