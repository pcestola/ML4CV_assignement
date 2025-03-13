import re
import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import network as network

from typing import Tuple
from torch.utils.data import DataLoader

from lib.train import train, PrototypeHead
from lib.data import ComposeSync, ToTensorSync, NormalizeSync, RandomCropSync, RandomHorizontalFlipSync, SegmentationDataset
from lib.loss import FocalLoss, CE_EntropyMinimization, Focal_EntropyMinimization

# region UTILS
def fix_random(seed: int):
    """
        Fix all possible sources of randomness.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def find_next_folder_name(parent_directory, prefix="train_") -> str:
    """
        Find the next folder name with the given prefix.
    """
    
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
    
    max_ci = -1
    
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                ci = int(match.group(1))
                max_ci = max(max_ci, ci)

    next_ci = max_ci + 1
    new_folder_name = f"{prefix}{next_ci}"
    return os.path.join(parent_directory, new_folder_name)

def setup_logger(log_dir:str, log_file:str, time:bool=False) -> logging.Logger:
    """
        Create a logger and return it.
    """
    log_path = os.path.join(log_dir, log_file)

    if time:
        format = '%(asctime)s - %(message)s'
    else:
        format = '%(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=format,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("TrainingLogger")

def setup_train_folder(dir:str, resume:int=None, time:bool=False) -> Tuple[str,str,logging.Logger]:
    """
        Create a new training folder.
    """
    if resume:
        path = os.path.join(dir,f'train_{resume}')
    else:
        path = find_next_folder_name(dir, 'train_')
        os.makedirs(path, exist_ok=True)
    
    logger = setup_logger(path,'train.log',time)

    ckpts_dir = os.path.join(path,'ckpts')
    os.makedirs(ckpts_dir, exist_ok=True)

    return path, ckpts_dir, logger
#endregion

if __name__ == '__main__':
    
    fix_random(0)

    parser = argparse.ArgumentParser(description="Training script for the DeepLabV3+ model.")
    
    # CUDA parameters
    parser.add_argument('--device', type=str, default='cuda:0')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    # Loss parameters
    parser.add_argument('--loss', type=str, default='ce', choices=['ce','fl','ce+h','fl+h'])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--gamma_focal', type=float, default=2.0)
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet101', 'mobilenet'])
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'voc'])
    parser.add_argument('--head', type=str, default='distance', choices=['distance', 'standard'])
    parser.add_argument('--class_weights', action='store_true')

    args = parser.parse_args()

    results_dir = os.path.join(os.getcwd(),'results')
    os.makedirs(results_dir, exist_ok=True)
    train_dir, ckpts_dir, logger = setup_train_folder(results_dir, False, True)

    for name, value in args.__dict__.items():
        logger.info(f"{name:<{15}} : {value}")
    
    
    # region DATASET
    path_train = os.path.join(os.getcwd(), 'data', 'train')
    path_train_images =  os.path.join(path_train, 'images', 'training', 't1-3')
    path_valid_images =  os.path.join(path_train, 'images', 'validation', 't4')
    path_train_masks =  os.path.join(path_train, 'annotations', 'training', 't1-3')
    path_valid_masks =  os.path.join(path_train, 'annotations', 'validation', 't4')
    
    if args.crop:
        train_transforms = ComposeSync([
            RandomCropSync((args.crop,args.crop)),
            RandomHorizontalFlipSync(p=0.5),
            ToTensorSync(),
            NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = ComposeSync([
            RandomHorizontalFlipSync(p=0.5),
            ToTensorSync(),
            NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    valid_transforms = ComposeSync([
        ToTensorSync(),
        NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Training dataset
    dataset_train = SegmentationDataset(
        images_dirs = path_train_images,
        masks_dirs = path_train_masks,
        transforms = train_transforms
    )

    # Validation dataset
    dataset_valid = SegmentationDataset(
        images_dirs = path_valid_images,
        masks_dirs = path_valid_masks,
        transforms = valid_transforms
    )

    logger.info("Datasets")
    logger.info(f"Training images: {len(dataset_train)}")
    logger.info(f"Validation images: {len(dataset_valid)}\n")

    # Dataloader
    num_workers = 2
    train_batch_size = args.batch
    valid_batch_size = 16

    def seed_worker(worker_id):
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)
      random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(0)

    dataloader_train = DataLoader(dataset_train,
                                batch_size=train_batch_size,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=generator)

    dataloader_valid = DataLoader(dataset_valid,
                                batch_size=valid_batch_size,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=generator)

    logger.info("Dataloaders")
    logger.info(f"Training batches: {len(dataloader_train)}")
    logger.info(f"Validation batches: {len(dataloader_valid)}\n")
    # endregion


    # region MODEL
    # Load the model
    if args.dataset == 'cityscapes':
        model = network.modeling.__dict__[f'deeplabv3plus_{args.backbone}'](num_classes=19, output_stride=16)
    elif args.dataset == 'voc':
        model = network.modeling.__dict__[f'deeplabv3plus_{args.backbone}'](num_classes=21, output_stride=16)
    
    path = os.path.join(os.getcwd(), 'ckpts', f'deeplabv3plus_{args.backbone}_{args.dataset}.pth')
    model.load_state_dict(torch.load(path, map_location=args.device)['model_state'])
    logger.info(f'Loaded weights from: {path}\n')

    # Change the head
    if args.head == 'standard':
        head = nn.Conv2d(256, 13, kernel_size=(1,1), stride=(1,1))
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
    if args.head == 'distance':
        head = PrototypeHead(256, 13)

    model.classifier.classifier[3] = head

    model.to(args.device)
    logger.info(f"Model loaded on {args.device} with head: {type(head)}\n")
    # endregion


    # region TRAINING
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.classifier.parameters(),'lr': args.lr},
        {'params': model.backbone.parameters(),'lr': args.lr * 0.1}
    ], weight_decay=1e-4)
    
    # Class weights
    inv_freq = [3.223254919052124, 8.324562072753906, 80.07819366455078, 214.23089599609375, 5441.0986328125, 80.23406219482422, 74.35269927978516, 3.00116229057312, 14.900472640991211, 11.333002090454102, 404.04559326171875, 29.11901092529297, 995.0892333984375]
    if args.class_weights:
        weights = torch.tensor(inv_freq, device=args.device, requires_grad=False)
    else:
        weights = None

    # Loss
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif args.loss == 'fl':
        criterion = FocalLoss(weights=weights)
    elif args.loss == 'ce+h':
        criterion = CE_EntropyMinimization(gamma=args.gamma, weights=weights)
    elif args.loss == 'fl+h':
        criterion = Focal_EntropyMinimization(gamma=args.gamma, weights=weights)

    criterion = criterion.to(args.device)

    train(model,
          criterion,
          optimizer,
          args.epochs,
          dataloader_train,
          dataloader_valid,
          args.device,
          'weights',
          ckpts_dir,
          logger)
    # endregion