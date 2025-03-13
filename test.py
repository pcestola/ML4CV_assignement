import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import network as network

from torch.utils.data import DataLoader

from lib.train import PrototypeHead
from lib.data import ComposeSync, ToTensorSync, NormalizeSync, SegmentationDataset
from lib.test import msp_score, max_logit_score, entropy_score, energy_score, calculate_aupr, calculate_miou

# region UTILS
def fix_random(seed: int) -> None:
    """
        Fix all possible sources of randomness
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_logger(log_dir:str, log_file:str, time:bool=False) -> logging.Logger:
    
    log_path = os.path.join(log_dir, log_file)

    if time:
        format = '%(asctime)s - %(message)s'
    else:
        format = '%(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
# endregion

if __name__ == '__main__':
    
    fix_random(0)

    parser = argparse.ArgumentParser(description="Testing script for the DeepLabV3+ model.")
    
    # CUDA parameters
    parser.add_argument('--device', type=str, default='cuda:0')
    # Test parameters
    parser.add_argument('--file', type=str)
    parser.add_argument('--head', type=int, default='distance', choices=['distance','standard'])
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet101','mobilenet'])
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes','voc'])

    args = parser.parse_args()

    results_dir = os.path.join(os.getcwd(),'results')
    test_dir = os.path.join(results_dir, args.file)
    ckpt_dir = os.path.join(test_dir, 'ckpts')
    logger = setup_logger(test_dir, 'test.log')

    for name, value in args.__dict__.items():
        logger.info(f"{name:<{15}} : {value}")
    logger.info("end\n")
    
    # region DATASET
    path_test = os.path.join(os.getcwd(), 'data', 'test')
    path_test_images_t5 =  os.path.join(path_test, 'images', 'test', 't5')
    path_test_images_t6 =  os.path.join(path_test, 'images', 'test', 't6')
    path_test_masks_t5 =  os.path.join(path_test, 'annotations', 'test', 't5')
    path_test_masks_t6 =  os.path.join(path_test, 'annotations', 'test', 't6')

    # Trasformazioni
    transforms = ComposeSync([
        ToTensorSync(),
        NormalizeSync(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset_test = SegmentationDataset(
        images_dirs = [path_test_images_t5, path_test_images_t6],
        masks_dirs = [path_test_masks_t5, path_test_masks_t6],
        transforms = transforms
    )

    logger.info("Dataset")
    logger.info(f"Test images: {len(dataset_test)}\n")

    # Dataloader
    num_workers = 2
    batch_size = 8

    def seed_worker(worker_id):
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)
      random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(0)

    dataloader_test = DataLoader(dataset_test,
                                batch_size=batch_size,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True)

    logger.info("Dataloader")
    logger.info(f"Test batches: {len(dataloader_test)}\n")
    # endregion


    # region MODEL    
    # Model
    model = network.modeling.__dict__[f'deeplabv3plus_{args.backbone}'](num_classes=19, output_stride=16)

    # Head
    if args.head == 'standard':
        head = nn.Conv2d(256, 13, kernel_size=(1, 1), stride=(1, 1))
    if args.head == 'distance':
        head = PrototypeHead(256, 13)

    model.classifier.classifier[3] = head

    # Load weights
    weight_path = os.path.join(ckpt_dir,'weights')
    state_dict = torch.load(weight_path,map_location=args.device)
    model.load_state_dict(state_dict['model_state_dict'])

    model.eval()
    model.to(args.device)
    logger.info(f'Loaded weights from: {weight_path}')
    logger.info(f"Model loaded on {args.device}\n")
    #endregion
    

    # region TEST
    logger.info("Testing dataset")
    msp_aupr = calculate_aupr(dataloader_test, model, msp_score, args.device)
    logger.info(f'MSP AUPR:\t{msp_aupr}')

    max_logit_aupr = calculate_aupr(dataloader_test, model, max_logit_score, args.device)
    logger.info(f'MAXLOG AUPR:\t{max_logit_aupr}')

    entropy_aupr = calculate_aupr(dataloader_test, model, entropy_score, args.device)
    logger.info(f'ENTROPY AUPR:\t{entropy_aupr}')

    energy_aupr = calculate_aupr(dataloader_test, model, energy_score, args.device)
    logger.info(f'ENERGY AUPR:\t{energy_aupr}\n')
    
    miou = calculate_miou(dataloader_test, model, args.device)
    classes = [0,1,2,3,5,6,7,8,9,10,11,12]
    string = 'IoU per class\n'
    for i, x in enumerate(miou):
        string += f'class {classes[i]}: {x}\n'
    logger.info(string)
    logger.info(f'mIoU: {miou.mean()}')
    logger.info(f'mIoU_no_zero: {miou[miou!=0].mean()}')
    logger.info(f'minIoU: {miou.min()}')
    logger.info(f'maxIoU: {miou.max()}\n\n')
    #endregion