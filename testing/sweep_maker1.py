
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
    'name': 'my-sweep',
    'method': 'grid',
    'metric': {
      'name': 'loss',
      'goal': 'minimize'
    },
    'parameters': {
        'data': {
            'batch_size': {
                'sweep': True,
                'values': [32, 64, 128]
            },
        },
        'trainer': {
            'epoch_count': {
                'sweep': True,
                'values': [10, 20],
            },
            'optimizer': {
                'sweep': True,
                'sweep_alias': [
                    'adam-wd-0.1',
                    'adam-wd-0.01',
                    'sgd',
                ],
                'values': [
                    {
                        'class_path': 'torch.optim.AdamW',
                        'init_args': {
                            'weight_decay': 0.1,
                        },
                    },
                    {
                        'class_path': 'torch.optim.AdamW',
                        'init_args': {
                            'weight_decay': 0.01,
                        },
                    },
                    {
                        'class_path': 'torch.optim.SGD',
                    },
                ]
            },
            'dy__upsert': [
                {
                    'sweep': True,
                    'sweep_identifier': 'lr',
                    'sweep_alias': [
                        'lr-0.001',
                        'lr-0.01',
                    ],
                    'values': [
                        {
                            'optimizer': {
                                'init_args': {'lr': 0.001},
                            },
                        },
                        {
                            'optimizer': {
                                'init_args': {'lr': 0.01},
                            },
                        },
                    ]
                }
            ]
        }
    }
}

if __name__ == "__main__":
    dysweep_run_resume(
        base_config=base_config,
        sweep_configuration=sweep_config,
        project='testing',
    )