import traceback
from pprint import pprint
import wandb
import sys
sys.path.append("../")
from dysweep import dysweep_run_resume, ResumableSweepConfig
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm
import dypy as dy
import yaml
from jsonargparse import ArgumentParser, ActionConfigFile
from pprint import pprint
import os
import pickle

def func(config, checkpoint_dir):
    print("Running on the following config:")
    pprint(config)
    print("----------")
    
    cfg = config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define transformations for the train set
    train_transform = transforms.Compose([
        dy.eval(x['class_path'])(**(x['init_args'] if 'init_args' in x and x['init_args'] else {})) for
            x in cfg['data']['train_transforms']
    ])

    # Define transformations for the test set
    test_transform = transforms.Compose([
        dy.eval(x['class_path'])(**(x['init_args'] if 'init_args' in x and x['init_args'] else {})) for
            x in cfg['data']['test_transforms']
    ])

    # Load datasets
    train_set = dy.eval(cfg['data']['dataset_class'])(root='./data', train=True, download=True, transform=train_transform)
    test_set = dy.eval(cfg['data']['dataset_class'])(root='./data', train=False, download=True, transform=test_transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
    )

    # Load the ResNet model
    model = dy.eval(cfg['model']['class_path'])(**cfg['model']['init_args']).to(device)
    
    # load the checkpoint from checkpoint_dir/last.ckpt
    if os.path.exists(os.path.join(checkpoint_dir, "last.ckpt")):
        print("Loading checkpoint from", os.path.join(checkpoint_dir, "last.ckpt"), "...")
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "last.ckpt")))
    if os.path.exists(os.path.join(checkpoint_dir, "last.pkl")):
        print("Loading epoch number ...")
        with open(os.path.join(checkpoint_dir, "last.pkl"), "rb") as f:
            pkl = pickle.load(f)
        epoch = pkl['epoch']
    else:
        epoch = 0
        
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = dy.eval(
            cfg['trainer']['optimizer']['class_path']
        )(model.parameters(), **cfg['trainer']['optimizer']['init_args'])

    try:
        epoch_count = cfg['trainer']['epoch_count']
        ###########################################
        model.train()
        while epoch < epoch_count: 
            print(f"Epoch [{epoch + 1}/{epoch_count}]")
            wandb.log({"epoch": epoch})
            for i, data in tqdm(enumerate(train_loader, 0)):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                wandb.log({"loss": loss.item()})
            ###########################################
            # compute the test accuracy
            correct = 0
            for i, data in tqdm(enumerate(test_loader, 0)):
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                # get the maximum logit
                _, predicted = torch.max(outputs.data, 1)
                # check if the prediction is correct
                correct += (predicted == labels).sum().item()

            wandb.log({"accuracy": correct / len(test_set)})
            
            epoch += 1
            
            # save the model checkpoint
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last.ckpt"))
            # save the epoch number in a pickle file
            with open(os.path.join(checkpoint_dir, "last.pkl"), "wb") as f:
                pickle.dump({'epoch': epoch}, f)
            
            
    finally:
        wandb.finish()


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
    dysweep_run_resume(
        conf=args,
        function=func,
        project='testing',
    )