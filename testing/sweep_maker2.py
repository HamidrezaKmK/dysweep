
import sys
sys.path.append("../")
from dysweep import dysweep_run_resume

base_config = {
    'data': {
        "dataset_class": "torchvision.datasets.CIFAR10",
        "batch_size": 64,
        "num_workers": 2,
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
                "init_args": {
                    "mean": [0.4914, 0.4822, 0.4465],
                    "std": [0.2023, 0.1994, 0.2010]
                }
            }
        ],
        "test_transforms": [
            {
                "class_path": "torchvision.transforms.ToTensor",
            },
            {
                "class_path": "torchvision.transforms.Normalize",
                "init_args": {
                    "mean": [0.4914, 0.4822, 0.4465],
                    "std": [0.2023, 0.1994, 0.2010]
                }
            }
        ],
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

sweep_config = {
    'name': 'normalization-sweep',
    'method': 'grid',
    'metric': {
      'name': 'loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'data': {
            'dy__upsert': [
                {
                    'sweep': True,
                    'sweep_identifier': 'norm_no_norm',
                    'sweep_alias': ['without_norm', 'with_norm'],
                    'values': [
                        # remove the last transform from both train and test transforms
                        {
                            'train_transforms': {
                                'dy__list__operations': [
                                    {'dy__remove': -1}
                                ]
                            },
                            'test_transforms': {
                                'dy__list__operations': [
                                    {'dy__remove': -1}
                                ]
                            }
                        },
                        # leave as is
                        {}
                    ]   
                }
            ]
        },
    }
}

if __name__ == "__main__":
    dysweep_run_resume(
        base_config=base_config,
        sweep_configuration=sweep_config,
        project='testing',
    )