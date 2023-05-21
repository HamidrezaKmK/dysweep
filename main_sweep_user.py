from hierarchical_sweep import wandbX
import wandb
import sys
from pprint import pprint
import traceback


def f():
    try:
        wandb.init(project="hierarchical_sweep")
        conf = wandb.config
        pprint(conf)
    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == "__main__":
    sweep_id = sys.argv[1]
    wandbX.agent(sweep_id, function=f, project="hierarchical_sweep", count=1)
