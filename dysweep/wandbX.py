
import typing as th
import wandb
from pprint import pprint
import os
from .utils import standardize_sweep_config, destandardize_sweep_config, upsert_config
from wandb.sdk.wandb_run import Run, _run_decorator
from wandb.sdk import wandb_config
import warnings
import copy
from pprint import pprint

METADATA_RUN_NAME_PREFIX = "HIERARCHICAL_SWEEP_"

base_config: th.Optional[dict] = None
compression: th.Optional[dict] = None


def hierarchical_config(conf):
    if base_config is None or compression is None:
        return conf
    # make a deep copy of the base config and upsert it with the current config
    return upsert_config(copy.deepcopy(base_config), destandardize_sweep_config(conf, compression))


def sweep(
    base_config: dict,
    sweep_config: dict,
    entity: th.Optional[str] = None,
    project: th.Optional[str] = None,
) -> str:
    """
    Create a run under the current project and entity, and save the
    base_config as an artifact. Then, compress the sweepConfiguration
    and turn it into a standard SweepConfig. Finally, pass the SweepConfig
    to wandb.sweep and return the sweep_id.

    I ended up with this implementation, because wandb has not yet released the
    Public API, so I had to use the artifacts capability for it to work.
    """
    # import these three variables from .utils: 
    # compression_mapping = {}
    # value_compression_mapping = {}
    # remaining_bunch = {}
    from .utils import compression_mapping, value_compression_mapping, remaining_bunch
    compression_mapping.clear()
    value_compression_mapping.clear()
    remaining_bunch.clear()
    
    # (1) change the sweep_config to a standard sweep_config
    sweep_standard, compression = standardize_sweep_config(sweep_config)
    sweep_metadata = {'base_config': base_config, 'compression': compression}
    sweep_id = wandb.sweep(sweep_standard, entity=entity, project=project)

    # (2) create a run and save the base_config as an artifact
    wandb.init(entity=entity, project=project,
               name=f"{METADATA_RUN_NAME_PREFIX}{sweep_id}",
               config=sweep_metadata,
               notes="This run contains the metadata for the sweep."
               "\nDrawn from the hierarchical sweep package, at:"
               "\n\thttps://github.com/HamidrezaKmK/hierarchical-sweep",
               tags=["hierarchical_sweep"])

    wandb.finish()

    return sweep_id


def agent(sweep_id, function=None, entity=None, project=None, count=None):
    """
    First, run the agent on the sweep_id. 
    Then call the same run that contains the artifact and use it for decompression
    of the sweep configuration. Then, decorate the function so that 
    when getting the sweep_configuration from the server, it will first
    decompress it and then do whatever it did with the config.
    """
    # (1) get the artfact from the run based on the run_name
    # iterate over all the runs
    # if the run_name starts with METADATA_RUN_NAME_PREFIX
    # then get the artifact from that run
    # and save it in the mapping_run_id_to_sweep_id

    # get the run_path from entity and project
    run_path = ""
    if entity is not None:
        run_path = os.path.join(run_path, entity)
    if project is not None:
        run_path = os.path.join(run_path, project)

    # List all the runs in the project
    all_runs = wandb.Api().runs(path=run_path)

    # find the run with the metadata for the sweep we are looking for
    sweep_run = None
    for run in all_runs:
        if run.name.startswith(METADATA_RUN_NAME_PREFIX) and \
                run.name[len(METADATA_RUN_NAME_PREFIX):] == sweep_id:
            sweep_run = run
            break

    if sweep_run is None:
        raise ValueError(
            f"Could not find the run with artifacts associated with: {sweep_id}\n"
            "Make sure you have the id correct!")

    global base_config, compression
    base_config = sweep_run.config['base_config']
    compression = sweep_run.config['compression']

    try:
        wandb.sdk.wandb_run.Run.hierarchical_config = property(
            lambda self: hierarchical_config(self.config)
        )
    except Exception as e:
        warnings.warn(
            "Could not set the hierarchical_config property on wandb_run.Run\n"
            "Use the hierarchical_config function instead.")

    return wandb.agent(sweep_id, function=function,
                       entity=entity, project=project, count=count)
