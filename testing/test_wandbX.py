

base_config = {
    'data': {
        'dataset': 'cifar10',
        'batch_size': 128,
        'chiz': [1, 2, 3]
    },
    'model': {
        'type': 'resnet',
        'depth': 20,
    },
    'optimizer': {
        'type': 'sgd',
        'lr': 0.1,
    },
}

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'model': {
            'type': {
                'sweep': True,
                'sweep_alias': ['type1', 'type2'],
                'values': ['resnet', 'densenet'],
            }
        },
        'data': {
            'chiz': {
                'sweep': True,
                'sweep_alias': ['chiz1', 'chiz2'],
                'values': [
                    {
                        'sweep_list_operations': [
                            {'sweep_overwrite': [0, 10]},
                        ]
                    },
                    {
                        'sweep_list_operations': [
                            {'sweep_overwrite': [1, 11]},
                        ]
                    },
                ]
            }
        },
        'upsert': [
            {
                'optimizer': {
                    'type': {
                        'dy_eval': {
                            'expression': 'def func(config):\n    return config["model"]["type"] == "resnet"',
                            'function_of_interest': 'func',
                        }
                    }
                }
            },
        ]
    }
}

expected_configurations = [
    {'data': {'batch_size': 128, 'chiz': [1, 11, 3], 'dataset': 'cifar10'},
     'model': {'depth': 20, 'type': 'densenet'},
     'optimizer': {'lr': 0.1, 'type': False}},
    {'data': {'batch_size': 128, 'chiz': [10, 11, 3], 'dataset': 'cifar10'},
     'model': {'depth': 20, 'type': 'resnet'},
     'optimizer': {'lr': 0.1, 'type': True}},
    {'data': {'batch_size': 128, 'chiz': [10, 2, 3], 'dataset': 'cifar10'},
     'model': {'depth': 20, 'type': 'densenet'},
     'optimizer': {'lr': 0.1, 'type': False}},
    {'data': {'batch_size': 128, 'chiz': [10, 2, 3], 'dataset': 'cifar10'},
     'model': {'depth': 20, 'type': 'resnet'},
     'optimizer': {'lr': 0.1, 'type': True}},
]
