from dataclasses import dataclass
import typing as th
from pathlib import Path
from .wandbX import sweep, agent, hierarchical_config
import functools
from random_word import RandomWords
import json
import shutil
import os
import traceback
import inspect
import threading
from pprint import pprint

SPLIT = '-'

@dataclass
class ResumableSweepConfig:
    default_root_dir: th.Optional[th.Union[Path, str]] = None
    custom_checkpoint_dir: th.Optional[th.Union[Path, str]] = None
    sweep_configuration: th.Optional[dict] = None
    base_config: th.Optional[dict] = None
    #
    entity: th.Optional[str] = None
    project: th.Optional[str] = None
    count: th.Optional[int] = None
    resume: bool = False
    rerun_id: th.Optional[str] = None
    run_name: th.Optional[str] = None
    sweep_id: th.Optional[th.Union[int, str]] = None
    #
    use_lightning_logger: bool = False
    #
    method: th.Optional[str] = 'grid'
    metric: th.Optional[str] = 'dysweep_default'
    goal: th.Optional[str] = 'minimize'
    sweep_name: th.Optional[str] = 'dysweep'

def check_non_empty(checkpoint_dir):
    all_subdirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    return len(all_subdirs) > 0
 
def get_max(all_subdirs):
    if len(all_subdirs) == 0:
        return 0
    identifiers = [int(d.name.split(SPLIT)[0]) for d in all_subdirs]
    return max(identifiers)

def dysweep_run_resume(
    conf: th.Optional[ResumableSweepConfig] = None,
    function: th.Optional[th.Callable] = None,
    default_root_dir: th.Optional[th.Union[Path, str]] = None,
    custom_checkpoint_dir: th.Optional[th.Union[Path, str]] = None,
    sweep_configuration: th.Optional[dict] = None,
    base_config: th.Optional[dict] = None,
    entity: th.Optional[str] = None,
    project: th.Optional[str] = None,
    count: th.Optional[int] = None,
    resume: th.Optional[bool] = None,
    rerun_id: th.Optional[str] = None,
    run_name: th.Optional[str] = None,
    sweep_id: th.Optional[th.Union[int, str]] = None,
    use_lightning_logger: th.Optional[bool] = None,
    method: th.Optional[str] = None,
    metric: th.Optional[str] = None,
    goal: th.Optional[str] = None,
    sweep_name: th.Optional[str] = None,
):
    """
    This is a multi-purpose function that does either one of the following functionalities:

    1. Creates a new sweep:
        This happens when sweep_id is not given. In which case, it will
        create a new sweep object according to the project and entity
        values that are given to the method.
    2. Calls the wandb.agent to get a specific run configuration:
        In this case, it will receive a specific configuration from the W&B
        sweep server. Then, using our API, changes that configuration into a standard
        hierarchical configuration that you can work with.
        It later passes the hierarchical configuration as a dictionary alongside
        the logger to the `function` provided.
    3. Resumes one of the left-off runs:
        It might be the case that the specific run is killed. This can be caused
        by either cluster preemption or by there being a specific bug that creates
        an exception in the code or kills it. In this case, under the checkpoint_dir
        (you can also specify it using `custom_checkpoint_dir`) you can see a summary
        of all the checkpoints.

    4. Re-runs one of the configurations:
        In this case, a specific run_id is given. This run_id should have been ran on the
        machine where re-running occurs. Now the function will retrieve the configuration
        that the run was previously executed with, retrieves any checkpoints if existant
        and `resume` is set to `True`.

    Instead of defining every single input to this function, you may also pass in a ResumableSweepConfig
    object instead. We included this feature because the intended usage of this package is coupled with
    `jsonargparse` where all the configurations are written in YAML format. (Check the examples section in the
    document).

    Args:
        conf:
            The entire configuration given as a ResumableSweepConfig object.
        function:
            This is a function that *YOU* have implemented and has the following signature:
            
            1. function(config, checkpoint_dir)
                Takes in the configuration and the checkpoint_dir associated with the run.
                You can store anything you want in the `checkpoint_dir` and then implement your
                own checkpoint retrieval procedure so that `function` can resume too.
                
            2. function(config, logger, checkpoint_dir)
                If you are using Pytorch Lightning, the W&B logger is wrapped in a WandbLogger object
                which has some external properties. We have also made the code Lightning comaptible.
                If `use_lightning_logger` is set to `True` in the configurations, then the function
                takes in the logger as well.
            
        default_root_dir: optional(Path or str)
            This is a required argument, you can specify a root directory in which all the checkpoint logs
            and everything related to the library is being stored.
        custom_checkpoint_dir: optional(Path or str)
            For checkpoints in specific, you can also define a custom_checkpoint_dir under the root_dir where
            the checkpoints will be stored.
        sweep_configuration: dict
            This is a hierarchical sweep_configuration as explained in the docs. Here you can pass
            a nested dictionary that contains all the information you need for running a sweep.
        base_config: dict
            This is the base configuration that the sweep will start off with.
        entity: str
            The entity where the sweep and runs are defined.
        project: str
            The project in which the sweep is supposed to reside in or be placed in if it is
            being instantiated.
        count: optional(int)
            Count only works when calling the agent to get a new set of configurations
            from the sweep server. For example, with count=2, the function will be run
            only on 2 new configurations.
            If you have a machine that stays alive for a long time, set the `count` value
            to be large; however, if you have a lot of machines that stay alive for not too
            long, it would be better to set the `count=1` and run a lot of these machines.
        resume: optional(bool)
            When set to True, it will try to use checkpoints. For example, when rerun_id is given
            it will also find the checkpoint_dir associated with that identifier and pass it to
            function. If rerun_id is not given, it will search for any runs in the `checkpoint_dir`
            that has failed and pass that to the function.
        rerun_id: optional(str)
            When re-running a specific run that you know it has failed, you can use this argument to specify.
        run_name: optional(str)
            If you want to set a specific names for your runs in the wandb UI, then you can specify here.
            Otherwise, it will default to a random word.
        sweep_id: optional(str)
            This argument determines whether you are using the function to instantiate a new sweep or are you
            willing to run agents or re-run specific a pre-defined sweep server.
        method: optional(str)
            The method used for the sweep.
        metric: optional(str)
            The metric used for the sweep.
        sweep_name: optional(str)
            The sweep name
        goal: optional(str)
            The goal used for the sweep.
        use_lightning_logger: optional(bool) = False
            When set to True, it will pass an additional argument `logger` to `function` that contains the
            lightning logger wrapper.
    Returns:
        It returns either one of the following:
        
        1. Returns the `sweep_id` if a sweep is instantiated.
        
        2. Returns the output of the function if the configuration only tries to run a single experiment.
        
        3. Otherwise, it will return None.
        
    """
    # if configuration is not given, then create it from the
    # input arguments
    if conf is None:
        conf = ResumableSweepConfig(
            default_root_dir=default_root_dir,
            custom_checkpoint_dir=custom_checkpoint_dir,
            sweep_configuration=sweep_configuration,
            base_config=base_config,
            entity=entity,
            project=project,
            count=count,
            resume=resume,
            rerun_id=rerun_id,
            run_name=run_name,
            sweep_id=sweep_id,
            use_lightning_logger=use_lightning_logger,
            method=method,
            metric=metric,
            sweep_name=sweep_name,
            goal=goal,
        )
    else:
        # if for any argument x, the value of x is not the default value
        # then overwrite conf.x with x
        if default_root_dir is not None:
            conf.default_root_dir = default_root_dir
        if custom_checkpoint_dir is not None:
            conf.custom_checkpoint_dir = custom_checkpoint_dir
        if sweep_configuration is not None:
            conf.sweep_configuration = sweep_configuration
        if base_config is not None:
            conf.base_config = base_config
        if entity is not None:
            conf.entity = entity
        if project is not None:
            conf.project = project
        if count is not None:
            conf.count = count
        if resume is not None:
            conf.resume = resume
        if rerun_id is not None:
            conf.rerun_id = rerun_id
        if run_name is not None:
            conf.run_name = run_name
        if sweep_id is not None:
            conf.sweep_id = sweep_id
        if use_lightning_logger is not None:
            conf.use_lightning_logger = use_lightning_logger
        if method is not None:
            conf.method = method
        if metric is not None:
            conf.metric = metric
        if sweep_name is not None:
            conf.sweep_name = sweep_name
        if goal is not None:
            conf.goal = goal
        
        
    if conf.project is None:
        raise ValueError("project should be given to the dysweep run and resume.")
    if function is None and sweep_id is not None:
        raise ValueError("function should be given to the dysweep run and resume when sweep_id is given.")

    if conf.sweep_id is not None:
        if conf.default_root_dir is None:
            conf.default_root_dir = './dysweep_logs'
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
            """
            This function handles extracting the logger, configuration, and checkpoint_dir
            and calls `function` internally.
            """
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

                if conf.resume or conf.rerun_id:
                    if not conf.rerun_id:
                        # find the first run that contains checkpoints
                        if len(all_subdirs) == 0:
                            raise ValueError(
                                f"The checkpoint directory {checkpoint_dir} is empty, so you can't resume from it.")
                        # get the name of the directory that we are trying to resume which
                        # contains the resume_id in its name.
                        dir_name = all_subdirs[0].name
                        experiment_id = SPLIT.join(dir_name.split(SPLIT)[1:])
                    else:
                        experiment_id = conf.rerun_id

                    # Using experiment_id, either get the actual configuration
                    # from the running directory or from the stored json file
                    old_dir_name = None
                    config_dir = checkpoint_dir / f"{experiment_id}-config.json"
                    if not os.path.exists(config_dir):
                        # The path exists and the run has been completed before
                        config_dir = None
                        for d in all_subdirs:
                            id_ = SPLIT.join(d.name.split(SPLIT)[1:])
                            if id_ == experiment_id:
                                old_dir_name = d.name
                                config_dir = checkpoint_dir / d.name / "sweep_config.json"

                    if not os.path.exists(config_dir):
                        raise FileNotFoundError(f"{config_dir} not found! Make sure the rerun_id is actually ran before.")

                    new_dir_name = f"{get_max(all_subdirs) + 1}{SPLIT}{experiment_id}"

                    # Load the configuration that was already used for running
                    # the function before.
                    with open(config_dir, "r") as f:
                        sweep_config = json.load(f)

                    if old_dir_name is not None:
                        # Change the name of the directory by pushing it to the end of the queue
                        shutil.copytree(checkpoint_dir / old_dir_name,
                                        checkpoint_dir / new_dir_name)
                        shutil.rmtree(checkpoint_dir / old_dir_name)

                    # create a new checkpoint directory for the inner function
                    new_checkpoint_dir = checkpoint_dir / new_dir_name

                    # Retrieve the logger
                    if conf.use_lightning_logger:
                        # Create a WandbLogger with the experiment_id
                        from lightning.pytorch.loggers import WandbLogger
                        logger = WandbLogger(
                            project=conf.project,
                            entity=conf.entity,
                            name=run_name,
                            id=experiment_id,
                        )
                    else:
                        # Call the init function of wandb with the experiment_id
                        import wandb
                        wandb.init(
                            project=conf.project,
                            entity=conf.entity,
                            name=run_name,
                            id=experiment_id,
                        )

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
                        sweep_config = hierarchical_config(
                            logger.experiment.config)
                    else:
                        import wandb
                        run_ = wandb.init(
                            project=conf.project,
                            entity=conf.entity,
                            name=run_name,
                        )
                        experiment_id = run_.id

                        sweep_config = hierarchical_config(wandb.config)

                    new_dir_name = f"{get_max(all_subdirs)+1}{SPLIT}{experiment_id}"

                    os.makedirs(checkpoint_dir / new_dir_name)

                    # dump a json in checkpoint_dir/run_id containing the sweep config
                    with open(checkpoint_dir / new_dir_name / "sweep_config.json", "w") as f:
                        json.dump(sweep_config, f, indent=4, sort_keys=True)

                    new_checkpoint_dir = checkpoint_dir / new_dir_name
            except Exception as e:
                print(traceback.format_exc())
                raise e

            if conf.use_lightning_logger:
                # check the function signature matches
                # the one we expect.
                # in which there are two arguments with the first one
                # named config and the second one named checkpoint_dir

                # get the signature of the function
                sig = inspect.signature(function)
                # get the parameters of the function
                params = sig.parameters
                # check that the function has two parameters
                if len(params) != 3 or list(params.keys())[0] != "config" or list(params.keys())[1] != "logger" or list(params.keys())[2] != "checkpoint_dir":
                    raise ValueError(
                        "the function passed to `dysweep_run_resume` should have the exact following parameters in order: (config, logger, checkpoint_dir)")
                try:
                    ret = function(sweep_config, logger, new_checkpoint_dir)
                except Exception as e:
                    # write exception into an err-log.txt file in the checkpoint_dir
                    with open(new_checkpoint_dir / "err-log.txt", "w") as f:
                        f.write(traceback.format_exc())
                    raise e
            else:
                # check the function signature matches
                # the one we expect.
                # in which there are two arguments with the first one
                # named config and the second one named checkpoint_dir

                # get the signature of the function
                sig = inspect.signature(function)
                # get the parameters of the function
                params = sig.parameters
                # check that the function has two parameters
                if len(params) != 2 or list(params.keys())[0] != "config" or list(params.keys())[1] != "checkpoint_dir":
                    raise ValueError(
                        "the function passed to `dysweep_run_resume` should have the exact following parameters in order: (config, checkpoint_dir)")
                try:
                    ret = function(sweep_config, new_checkpoint_dir)
                except Exception as e:
                    # write exception into an err-log.txt file in the checkpoint_dir
                    with open(new_checkpoint_dir / "err-log.txt", "w") as f:
                        f.write(traceback.format_exc())
                    raise e

            # remove the entire new_checkpoint_dir if the function has finished
            # running.
            shutil.copyfile(new_checkpoint_dir / "sweep_config.json",
                            checkpoint_dir / f"{experiment_id}-config.json")
            # clean up after yourself to avoid excess memory consumption
            shutil.rmtree(new_checkpoint_dir)
            # finish the wandb run so that later .init calls can resume different ones
            wandb.finish()

            return ret


        if conf.resume and not conf.rerun_id:
            # In this case, we will sequantially resume
            # any run that is remaining with the limit of `count`
            if conf.count > 1:
                for _ in range(conf.count):
                    if check_non_empty(checkpoint_dir):
                        # run modified_function in a separate thread and wait for it to finish
                        # before running the agent.
                        # this is to ensure that the function is running before the agent
                        # starts.
                        modified_function_thread = threading.Thread(
                            target=modified_function)
                        modified_function_thread.start()
                        modified_function_thread.join()

                    else:
                        break
            else:
                # A single run is performed
                return modified_function()
        elif conf.rerun_id:
            return modified_function()
        else:
            agent(conf.sweep_id, function=modified_function,
                  entity=conf.entity, project=conf.project, count=conf.count)
    else:
        # check if conf.sweep_configuration complies to the
        # standard sweep format or not.
        all_keys = list(conf.sweep_configuration.keys())
        if 'name' not in all_keys or 'method' not in all_keys or 'metric' not in all_keys or 'parameters' not in all_keys:
            # enter defaults
            conf.sweep_configuration = {
                'method': conf.method,
                'metric': {
                    'name': conf.metric,
                    'goal': conf.goal,
                },
                'parameters': conf.sweep_configuration
            }

        try:
            sweep_id = sweep(conf.base_config, conf.sweep_configuration,
                             entity=conf.entity, project=conf.project)
        except Exception as e:
            print("Exception at creation of sweep:")
            print(traceback.format_exc())
            raise e
        finally:
            import wandb
            wandb.finish()
        return sweep_id