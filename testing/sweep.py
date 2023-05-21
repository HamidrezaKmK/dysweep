from dysweep import dysweep_run_resume, ResumableSweepConfig
from jsonargparse import ArgumentParser
import os
from pathlib import Path
from pprint import pprint
from jsonargparse.actions import ActionConfigFile


def run(conf, checkpoint_dir):
    print("Checkpoint: ", checkpoint_dir)
    pprint(conf)
    # sleep for 60 seconds
    import time
    time.sleep(60)


if __name__ == "__main__":
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
    args.default_root_dir = Path.cwd() / args.default_root_dir
    # if the path does not exists then create it
    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    dysweep_run_resume(args, run)
