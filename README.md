# Semantic Segmentation with Anomaly Detection on StreetHazards
This project implements semantic segmentation combined with anomaly detection for autonomous driving scenarios. A DeepLabv3+ based architecture is employed to segment known classes (e.g., roads, cars, pedestrians) while detecting unknown objects or anomalies in the scene. Evaluation is performed on the [StreetHazards dataset](https://paperswithcode.com/dataset/streethazards), a synthetic benchmark designed for anomaly segmentation, with the aim of achieving high segmentation accuracy on familiar objects and reliably flagging unexpected elements through prediction uncertainty analysis.

## Repository Structure

```plaintext
.
├── main.ipynb          # Main Jupyter Notebook: end-to-end workflow (training, testing, ablation study)
├── lib/                # Custom functions and classes
│   ├── data.py         # Data loading and preprocessing utilities
│   ├── train.py        # Training routines and model architectures
│   ├── test.py         # Testing routines and evaluation metrics
│   └── utils.py        # Utility functions for visualization and logging
├── ckpts/              # Model checkpoints and result files
│   ├── weights.pt      # Model weights (saved after executing main.ipynb)
│   ├── tuning.csv      # Hyperparameter tuning results
│   └── ablation.csv    # Ablation study results
├── network/            # External repository code (DeepLabv3+ implementation)
├── data/               # Dataset folder
├── train.py            # Original training script for standalone execution
├── test.py             # Original testing script for standalone execution
└── README.md           # This documentation file

```
The primary workflow is defined in `main.ipynb`, which integrates all stages: environment setup, data loading, model building, training and evaluation. Custom modules supporting the pipeline are located in the `lib` directory, while external code in the `network` folder and pre-trained weights are sourced from this [repository](https://git01lab.cs.univie.ac.at/est-gan/deeplabv3plus-pytorch).

## Installation and Requirements
This project is designed to run on Google Colab via the complete `main.ipynb` file, which automatically installs all required dependencies and downloads the necessary files. No manual setup is necessary.

## Custom Library Overview
- `./lib/train` contains routines for training, including model architecture definitions, loss functions, logging, and checkpointing.

- `./lib/test` provides evaluation functions. This includes scoring functions, AUPR and mIoU.

- `./lib/data` manages data loading and preprocessing for segmentation tasks. Includes a custom dataset class and synchronized augmentations classes to ensure consistency between image and mask transformations.

- `./lib/utils` offers utility functions for visualization (e.g., displaying segmentation outputs, plotting performance metrics, and generating comparison plots).

## Notebook Structure
The `main.ipynb` notebook contains a comprehensive explanation of the project, including all necessary details about the work performed and the reasoning behind each step. It serves as the primary guide for understanding the project and its implementation. The notebook is organized into the following sections:
- **Setup**: mounting the drive, setting paths, importing libraries, and downloading weights.
- **Dataset**: overview and preparation of the dataset, including download.
- **Model**: construction of the model.
- **Training**: execution of the training loop.
- **Test**: evaluation of model performance using mIoU and AUPR metrics.
- **Ablation Study**: additional experiments and analyses.

However, if needed, the scripts originally used for conducting the training and testing processes (`train.py` and `test.py`) are also provided. These scripts require additional configuration and installation of dependencies. Please refer to the final section of this README for further instructions on using these files.

## Setup
1. Upload the entire project folder to Google Drive.
2. Inside `main.ipynb`, two cells are provided to download and extract the train and test datasets. If the datasets are already available on Google Drive, simply move them into the `data` folder and skip the cells. The cell for downloading the train set is disabled by default, as it is not necessary for testing the model. If both datasets are downloaded, the final directory structure should be as follows:
   ```plaintext
   data
   ├── train
   │   ├── images
   │   └── annotations
   └── test
       ├── images
       └── annotations
   ```
3. Open the `main.ipynb` file in Google Colab.
4. Execute the first cell to mount Google Drive:

   ```python
    from google.colab import drive
    drive.mount("/content/drive")
   ```
   
5. Update the path to the project folder in the second cell:  
    ```bash
    %cd /content/drive/MyDrive/YOUR_FOLDER_PATH
    ```
6. Execute the remaining cells sequentially. Training and non-essential cells are disabled using `%%script echo skipping` to ensure smooth execution. The notebook includes cells for:  
    - Installing dependencies (e.g., torchmetrics)
    - Downloading the test dataset (≈2GB)
    - Downloading the train dataset (≈9GB)
    - Downloading pre-trained model weights for training (≈224MB)
    - Downloading trained model weights for evaluation (≈224MB)

## Expected Results
After completing the evaluation processes, the following performance is expected on the StreetHazards test set:

### Segmentation Performance

|mIoU  | Unlabelled | Building | Fence  | Other  | Pole   | Road Line | Road   | Sidewalk | Vegetation | Car    | Wall   | Traffic Sign |
|------|------------|----------|--------|--------|--------|-----------|--------|----------|------------|--------|--------|--------------|
|0.6510| 0.8906     | 0.7944   | 0.4264 | 0.3610 | 0.3570 | 0.6368    | 0.9629 | 0.7193   | 0.8660     | 0.7191 | 0.5578 | 0.5207       |

### Anomaly Detection Performance

| Entropy-based AUPR |
|--------------------|
| 17.3%              |

## Using the Original Training and Testing Scripts
Before running these scripts from Colab:

1. Install the `torchmetrics` library.

2. Download the train and test dataset as shown in the `main.ipynb` notebook.

3. Download the pre-trained weights with which to perform training, use one of the following commands, depending on the desired model:
    ```plaintext
    !gdown -c 1-9mz8Dv2_td52qDeDlsBS_AMETXbDPJE -O network/deeplabv3plus_resnet101_voc.pt            (≈224MB)
    !gdown -c 1nmER-DVgFgpbwLN3rRlx0Hf_jqYRYdvc -O network/deeplabv3plus_resnet101_cityscapes.pt     (≈224MB)
    !gdown -c 1-B5OXRUE6K4G6NESlj6Acp5oO3Gqw9Vv -O network/deeplabv3plus_mobilenet_voc.pt            (≈20MB)
    !gdown -c 1-BWSqAzgPMy_QiTBwYHH5xJabmCWUHCz -O network/deeplabv3plus_mobilenet_cityscapes.pt     (≈20MB)
    ```
    
The training script automatically create a `/results` folder and generate a file named `train_n.py` (where `n` is a sequential identifier starting from 0). Inside this folder, the trained model and a log file detailing the training progress will also be saved. Example:
```plaintext
   results
   ├── train_0
   │   ├── ckpts
   │   │   └── weights.pt
   │   └── train.log
   └── train_1
       ├── ckpts
       │   └── weights.pt
       └── train.log
   ```

To manually run the scripts, use the following commands adjusting parameters as desired:

Train: `python train.py --backbone resnet101 --head distance --dataset cityscapes --loss fl+h --gamma 0.1 --gamma_focal 2.0`

Test: `python test.py --file <train_file_name>.py --backbone resnet101 --head distance --dataset cityscapes`

