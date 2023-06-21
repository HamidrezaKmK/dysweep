
import sys
sys.path.append("../")
from dysweep import dysweep_run_resume

base_config = {
    'data': {
        "batch_size": 64,
        "num_workers": 2,
    },
    "model": {
        "class_path": "torchvision.models.resnet50",
        "init_args": {
            "pretrained": False,
            "num_classes": 10
        }
    },
    "trainer": {
        "epoch_count": 10,
        "optimizer": {
            "class_path": "torch.optim.SGD",
            "init_args": {
                "lr": 0.001
            }
        }
    }
}
get_transform_from_conf = """
def func(conf):
    dataset_type = conf['data']['dataset_class'].split('.')[-1]
    if dataset_type == 'CIFAR100':
        return {
            'mean': [0.5071, 0.4865, 0.4409],
            'std': [0.2673, 0.2564, 0.2762]
        }
    elif dataset_type == 'CIFAR10':
        return {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        }
"""

get_num_classes_from_conf = """
def func(conf):
    dataset_type = conf['data']['dataset_class'].split('.')[-1]
    if dataset_type == 'CIFAR100':
        return 100
    elif dataset_type == 'CIFAR10':
        return 10
"""

sweep_config = {
    'name': 'dataset-sweep-dy-eval',
    'method': 'grid',
    'metric': {
      'name': 'loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'data': {
            'dataset_class': {
                'sweep': True,
                'values': [
                    'torchvision.datasets.CIFAR100',
                    'torchvision.datasets.CIFAR10'
                ]
            }
        },
        'dy__upsert': [
            {
                'data': {
                    "train_transforms": [
                        {
                            "class_path": "torchvision.transforms.RandomHorizontalFlip",
                        },
                        {
                            "class_path": "torchvision.transforms.RandomCrop",
                            "init_args": {
                                "size": 32,
                                "padding": 4
                            }
                        },
                        {
                            "class_path": "torchvision.transforms.ToTensor",
                        },
                        {
                            "class_path": "torchvision.transforms.Normalize",
                            # set the transform according to the dataset dynamically
                            "init_args": {
                                "dy__eval": {
                                    "expression": get_transform_from_conf,
                                    "function_of_interest": "func"
                                }
                            }
                        }
                    ],
                    "test_transforms": [
                        {
                            "class_path": "torchvision.transforms.ToTensor",
                        },
                        {
                            "class_path": "torchvision.transforms.Normalize",
                            # set the transform according to the dataset dynamically
                            "init_args": {
                                "dy__eval": {
                                    "expression": get_transform_from_conf,
                                    "function_of_interest": "func"
                                }
                            }
                        }
                    ],
                },
                'model' : {
                    'init_args': {
                        'num_classes': {
                            'dy__eval': {
                                'expression': get_num_classes_from_conf,
                                'function_of_interest': 'func',
                            }
                        }
                    }
                }
            }
        ]
    }
}


if __name__ == "__main__":
    dysweep_run_resume(
        base_config=base_config,
        sweep_configuration=sweep_config,
        project='testing',
    )