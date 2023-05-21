import traceback
from pprint import pprint
import wandb
from dysweep import wandbX
import sys
sys.path.append("../")


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
