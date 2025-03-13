import torch
import logging
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm
from pathlib import Path
from typing import Callable

# Model
class PrototypeHead(nn.Module):
  """
    A prototype-based classification head.

    This module learns num_classes prototype vectors in a feature space of
    dimension num_features. During the forward pass, it computes the negative
    Euclidean distance between the input feature maps and the learned prototypes.

    Attributes:
      n (int): Number of feature dimensions.
      c (int): Number of classes.
      prototypes (nn.Parameter): A learnable tensor of shape (n, c)
  """
  def __init__(self, num_features:int, num_classes:int):
    super(PrototypeHead, self).__init__()
    self.n = num_features
    self.c = num_classes

    self.prototypes = nn.Parameter(torch.empty(self.c, self.n))
    nn.init.xavier_uniform_(self.prototypes)

  def forward(self, x):
    x = x.permute((0,2,3,1))
    x = -torch.cdist(x, self.prototypes)
    x = x.permute((0,3,1,2))

    return x

# Train
def log_training_progress(logger:logging.Logger, optimizer:torch.optim.Optimizer, epoch:int, train_loss:float, val_loss:float):
  """
  Logs training progress on a file including learning rate, epoch, training loss,
  and validation loss.

  Parameters:
    logger (logging.Logger): The logger instance to use for logging.
    optimizer (torch.optim.Optimizer): The optimizer used during training.
    epoch (int): The current epoch number.
    train_loss (float): The training loss.
    val_loss (float): The validation loss.
  """
  lr_list = [group['lr'] for group in optimizer.param_groups]
  lr_list = f"LR: {', '.join(f'{lr:<8.1e}' for lr in lr_list)}"
  logger.info(f"Epoch: {epoch:<8d}, Training Loss: {train_loss:<7.3f}, Validation Loss: {val_loss:<7.3f}"+lr_list)


def train_epoch(model: nn.Module,
                train_loader: data.DataLoader,
                device: torch.device,
                optimizer: torch.optim.Optimizer,
                criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
  """
    Trains the model for one epoch.

    Parameters:
      model (nn.Module): The model to be trained.
      train_loader (data.DataLoader): The dataloader for the training data.
      device (torch.device): The device to perform computation on.
      optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
      criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.

    Returns:
      float: The average training loss for the epoch.
  """
  model.train()
  running_loss = 0.0

  for images, masks in tqdm(train_loader, desc='Training'):
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()

  return running_loss / len(train_loader)


def validate(model: nn.Module,
             val_loader: data.DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
  """
    Validates the model on a validation dataset.

    Parameters:
      model (nn.Module): The model to be validated.
      val_loader (data.DataLoader): The dataloader for the validation data.
      device (torch.device): The device to perform computation on.
      criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.

    Returns:
      float: The average validation loss.
  """
  model.eval()
  running_loss = 0.0

  with torch.no_grad():
      for images, masks in tqdm(val_loader, desc='Validation'):
          images = images.to(device, non_blocking=True)
          masks = masks.to(device, non_blocking=True)

          outputs = model(images)
          loss = criterion(outputs, masks)

          running_loss += loss.item()

  return running_loss / len(val_loader)


def training_loop(num_epochs: int,
                  model: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  train_loader: data.DataLoader,
                  val_loader: data.DataLoader,
                  device: torch.device,
                  run_name: str,
                  checkpoints_dir: str,
                  logger: logging.Logger,
                  log_step: int = 15):
  """
    Runs the training loop for a specified number of epochs, including logging
    and checkpoint saving.

    Parameters:
      num_epochs (int): The number of epochs to train.
      model (nn.Module): The model to be trained.
      optimizer (torch.optim.Optimizer): The optimizer used for training.
      scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
      criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.
      train_loader (data.DataLoader): The dataloader for training data.
      val_loader (data.DataLoader): The dataloader for validation data.
      device (torch.device): The device for computation.
      run_name (str): The name of the training run.
      checkpoints_dir (str): The directory path where checkpoints are saved.
      logger (logging.Logger): The logger for logging training progress.
      log_step (int, optional): The frequency (in epochs) at which to log training progress.
  """
  checkpoints_path = Path(checkpoints_dir)
  checkpoints_path.mkdir(parents=True, exist_ok=True)
  best_val_loss = float('inf')

  for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
    val_loss = validate(model, val_loader, device, criterion)

    if epoch % log_step == 0:
      if logger != None:
        log_training_progress(logger, optimizer, epoch, train_loss, val_loss)
      else:
        print(f"Epoch: {epoch:4}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
      }, checkpoints_path / f"{run_name}.pt")


def train(model: nn.Module,
          criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          optimizer: torch.optim.Optimizer,
          epochs: int,
          train_loader: data.DataLoader,
          val_loader: data.DataLoader,
          device: torch.device,
          run_name: str,
          checkpoints_dir: str,
          logger: logging.Logger=None):
    """
      Sets up and initiates the training process.

      Parameters:
        model (nn.Module): The model to be trained.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        epochs (int): The total number of training epochs.
        train_loader (data.DataLoader): The data loader for training data.
        val_loader (data.DataLoader): The data loader for validation data.
        device (torch.device): The device for computation.
        run_name (str): The name identifier for the training run.
        checkpoints_dir (str): The directory where model checkpoints will be saved.
        logger (logging.Logger, optional): A logger instance for logging training progress.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    print("Start training")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Epochs: {epochs}")
    print(f"Run name: {run_name}\n")

    training_loop(epochs, model, optimizer, scheduler, criterion, train_loader, val_loader, device, run_name, checkpoints_dir, logger)

# Loss
class CE_Entropy(nn.Module):
  """
    A class that computes the cross-entropy loss combined with an entropy
    minimization term.

    Attributes:
      entropy_coefficient (float): The weight coefficient for the entropy term.
  """
  def __init__(self, entropy_coefficient=0.1, weights=None):
    super(CE_Entropy, self).__init__()
    self.ce_loss = nn.CrossEntropyLoss(weight=weights)
    self.entropy_coefficient = entropy_coefficient

  def forward(self, logits, targets):

    ce_loss = self.ce_loss(logits, targets)
    
    entropy = torch.softmax(logits,dim=1).clip(min=1e-10)
    entropy = -torch.sum(entropy*torch.log(entropy),dim=1).mean()

    return ce_loss + self.entropy_coefficient*entropy


class FocalLoss(nn.Module):
  """
    A class that implements the focal loss for handling class imbalance.

    Attributes:
      gamma (float): The focusing parameter of the loss.
      weights (Tensor or None): Optional tensor for class weights.
      output_probabilities (bool): If True, the forward method returns both the loss and the probabilities.
  """
  def __init__(self, gamma=2.0, weights=None, output_probabilities:bool=False):
    super(FocalLoss, self).__init__()
    self.gamma = gamma
    self.weights = weights
    self.output_probabilities = output_probabilities

  def forward(self, logits, target):

    pt = torch.softmax(logits,dim=1).clip(min=1e-10)
    pt = pt.gather(1, target.unsqueeze(1))
    pt = pt.squeeze(1)

    logpt = torch.log(pt)

    if self.weights is not None:
      loss = - self.weights[target] * (1 - pt) ** self.gamma * logpt
    else:
      loss = - (1 - pt) ** self.gamma * logpt

    if self.weights is not None:
      loss = loss.sum() / self.weights[target].sum()
    else:
      loss = loss.mean()

    if self.output_probabilities:
      return loss, pt
    else:
      return loss


class FocalLoss_Entropy(nn.Module):
  """
    A class that combines focal loss with an entropy term.

    Attributes:
      gamma_focal (float): The focusing parameter of the loss.
      entropy_coefficient (float): The weight coefficient for the entropy term.
      weights (Tensor or None): Optional tensor of class weights.
  """
  def __init__(self, gamma_focal:float=2.0, entropy_coefficient:float=0.1, weights=None):
    super(FocalLoss_Entropy, self).__init__()
    self.focal_loss = FocalLoss(weights=weights, output_probabilities=True)
    self.entropy_coefficient = entropy_coefficient

  def forward(self, logits, targets):

    focal_loss, probs = self.focal_loss(logits, targets)

    entropy = -torch.sum(probs*torch.log(probs),dim=1).mean()

    return focal_loss + self.entropy_coefficient*entropy
