import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, val_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.validate()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-ds', '--dataset', default=None, type=str,
                        help='Path to the dataset, that should be validated on.')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # args.config = "/home/rog/training_results/trained_models/50_train_lovasz_softmax/config.json"
    # args.resume = "/home/rog/training_results/trained_models/50_train_lovasz_softmax/best_model.pth"

    availble_gpus = list(range(torch.cuda.device_count()))
    args.device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')


    config = json.load(open(args.config))
    config['val_loader']['args']['split'] = [args.dataset.split("/")[-1]]
    config['val_loader']['args']['data_dir'] = args.dataset.replace(f"/{config['val_loader']['args']['split'][0]}", "")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.type

    main(config, args.resume)