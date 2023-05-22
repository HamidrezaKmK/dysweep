from dataclasses import dataclass
import typing as th
from pathlib import Path
from .wandbX import sweep, agent
import functools
from random_word import RandomWords
import json
import shutil
import os
import traceback

SPLIT = '-'


@dataclass
class ResumableSweepConfig:
    default_root_dir: th.Union[Path, str]
    custom_checkpoint_dir: th.Optional[th.Union[Path, str]] = None
    sweep_configuration: th.Optional[dict] = None
    base_config: th.Optional[dict] = None
    #
    entity: th.Optional[str] = None
    project: th.Optional[str] = None
    count: th.Optional[int] = None
    resume: bool = False
    run_name: th.Optional[str] = None
    checkpoint_interval: int = 3600
    sweep_id: th.Optional[th.Union[int, str]] = None
    #
    use_lightning_logger: bool = False


def check_non_empty(checkpoint_dir):
    all_subdirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    return len(all_subdirs) > 0


def dysweep_run_resume(conf: ResumableSweepConfig, function: th.Callable):

    if conf.sweep_id is not None:
        # This means that we are running a sweep that
        # already exists. We either resume something that
        # has been crashed before, or we start something new.
        if conf.custom_checkpoint_dir is not None:
            checkpoint_dir = Path(
                conf.custom_checkpoint_dir) / conf.custom_checkpoint_dir
        else:
            checkpoint_dir = Path(conf.default_root_dir) / \
                f"checkpoints-{conf.sweep_id}"
        # create the checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Assume that function is well-formed.
        # in that case, modified_function will handle all the resumption
        # logic so that function can be run as if it was a normal function.
        def modified_function():
            try:
                # list and sort all of the checkpoint subdirectories by the order
                # they were created.
                all_subdirs = [
                    d for d in checkpoint_dir.iterdir() if d.is_dir()]
                all_subdirs = sorted(
                    all_subdirs, key=lambda x: int(x.name.split(SPLIT)[0]))

                # Change the run-name by adding something random to it
                # if the conf.run_name is None just assign it to None
                # and wandb will handle the rest.
                if conf.run_name is not None:
                    r = RandomWords()
                    w = r.get_random_word()
                    run_name = conf.run_name + '-' + w
                else:
                    run_name = conf.run_name

                if conf.resume:
                    # Resuming mode
                    if len(all_subdirs) == 0:
                        raise ValueError(
                            f"The checkpoint directory {checkpoint_dir} is empty, so you can't resume from it.")
                    # get the name of the directory that we are trying to resume which
                    # contains the resume_id in its name.
                    dir_name = all_subdirs[0].name
                    resume_id = SPLIT.join(dir_name.split(SPLIT)[1:])

                    if conf.use_lightning_logger:
                        # Create a WandbLogger with the resume_id
                        from lightning.pytorch.loggers import WandbLogger
                        logger = WandbLogger(
                            project=conf.project,
                            entity=conf.entity,
                            name=run_name,
                            id=resume_id,
                            resume="must",
                        )
                    else:
                        # Call the init function of wandb with the resume_id
                        import wandb
                        wandb.init(
                            project=conf.project,
                            entity=conf.entity,
                            name=run_name,
                            id=resume_id,
                            resume="must",
                        )

                    # Load the configuration that was already used for running
                    # the function before.
                    with open(checkpoint_dir / dir_name / "sweep_config.json", "r") as f:
                        sweep_config = json.load(f)

                    # Change the name of the directory by pushing it to the end of the queue
                    mx = int(all_subdirs[-1].name.split(SPLIT)[0])
                    new_dir_name = f"{mx+1}{SPLIT}{resume_id}"
                    shutil.copytree(checkpoint_dir / dir_name,
                                    checkpoint_dir / new_dir_name)
                    shutil.rmtree(checkpoint_dir / dir_name)

                    # create a new checkpoint directory for the inner function
                    new_checkpoint_dir = checkpoint_dir / new_dir_name
                else:
                    # if the run_id doesn't exist, then create a new run
                    # and create the subdirectory
                    if conf.use_lightning_logger:
                        from lightning.pytorch.loggers import WandbLogger
                        logger = WandbLogger(
                            project=conf.project,
                            entity=conf.entity,
                            name=run_name,
                        )
                        experiment_id = logger.experiment.id
                        sweep_config = logger.experiment.config
                    else:
                        import wandb
                        wandb.init(
                            project=conf.project,
                            entity=conf.entity,
                            name=run_name,
                        )
                        experiment_id = wandb.run.id
                        sweep_config = wandb.config

                    new_dir_name = f"{len(all_subdirs)+1}{SPLIT}{experiment_id}"

                    os.makedirs(checkpoint_dir / new_dir_name)

                    # dump a json in checkpoint_dir/run_id containing the sweep config
                    with open(checkpoint_dir / new_dir_name / "sweep_config.json", "w") as f:
                        json.dump(sweep_config, f)

                new_checkpoint_dir = checkpoint_dir / new_dir_name
            except Exception as e:
                print(traceback.format_exc())
                raise e

            if conf.use_lightning_logger:
                ret = function(sweep_config, logger, new_checkpoint_dir)
            else:
                ret = function(sweep_config, new_checkpoint_dir)

            # remove the entire new_checkpoint_dir if the function has finished
            # running.
            shutil.rmtree(new_checkpoint_dir)

            return ret
        if conf.resume:
            for _ in range(conf.count):
                if check_non_empty(checkpoint_dir):
                    modified_function()
                else:
                    break
        else:
            agent(conf.sweep_id, function=modified_function,
                  entity=conf.entity, project=conf.project, count=conf.count)
    else:
        sweep(conf.base_config, conf.sweep_configuration,
              entity=conf.entity, project=conf.project)
