import torch
from tqdm import tqdm
from torchmetrics import JaccardIndex, AveragePrecision

def msp_score(logits:torch.Tensor):
  """
  Computes the maximum softmax probability score.

  Parameters:
    logits (torch.Tensor): The input tensor containing model outputs.
  """
  score, _ = torch.max(torch.softmax(logits,dim=1),dim=1)
  return -score


def max_logit_score(logits:torch.Tensor):
  """
    Computes the maximum logit score.

    Parameters:
      logits (torch.Tensor): The input tensor containing model outputs.
  """
  score, _ = torch.max(logits,dim=1)
  return -score


def entropy_score(logits:torch.Tensor):
  """
    Computes the entropy score over the softmax probabilities of logits.

    Attributes:
      logits (torch.Tensor): The input tensor containing model outputs.
  """
  probs = torch.softmax(logits,dim=1).clamp(min=1e-10)
  return -torch.sum(probs * torch.log(probs), dim=1)


def energy_score(logits:torch.Tensor):
  """
    Computes the energy score using the log-sum-exp of logits.

    Attributes:
      logits (torch.Tensor): The input tensor containing raw model outputs.
  """
  return -torch.logsumexp(logits, dim=1)


def calculate_aupr(dataloader, model, score_function, device, thresholds=None):
  """
  Calculates the Area Under the Precision-Recall curve (AUPR) of a model over
  a dataloader using a provided score function.

  Attributes:
    dataloader (iterable): The dataloader providing batches of images and masks.
    model (torch.nn.Module): The model used to generate predictions.
    score_function (callable): The function used to compute the score from logits.
    device (torch.device): The device on which the computations are performed.
    thresholds (int): The number of threshold for AUPR. None for automatic bins.

  Returns:
    torch.Tensor: The computed average precision metric as a scalar tensor.
  """
  model.eval()

  eval_AUPR = AveragePrecision(
      task='binary',
      thresholds=thresholds
  ).to(device)

  with torch.no_grad():
    for image, mask in tqdm(dataloader, desc="Processing", leave=False):

      image = image.to(device, non_blocking=True)
      mask = mask.to(device, non_blocking=True)

      score = score_function(model(image)).reshape((-1))
      label = (mask == 13).to(torch.int32).reshape((-1))

      eval_AUPR.update(score, label)
    
  aupr = eval_AUPR.compute().cpu()

  del image, mask, score, label, eval_AUPR
  torch.cuda.empty_cache()

  return aupr


def calculate_miou(dataloader, model, device, ignore_anomaly:bool=True):
  """
    Calculates the mean Intersection over Union (mIoU) metric for
    segmentation tasks.

    Attributes:
      dataloader (iterable): The data loader providing batches of images and masks.
      model (torch.nn.Module): The model used for generating predictions.
      device (torch.device): The device on which computations are performed.
      ignore_anomaly (bool): If True, the anomaly class is ignored during metric calculation.

    Returns:
      torch.Tensor: The computed mIoU metric per class.
  """
  model.eval()

  eval_IoU = JaccardIndex(
    task='multiclass',
    num_classes=14,
    ignore_index=13 if ignore_anomaly else None,
    average="none",
    zero_division=0
  ).to(device)

  with torch.no_grad():
    for images, masks in tqdm(dataloader, desc="Processing", leave=False):
        
      images = images.to(device, non_blocking=True)
      masks = masks.to(device, non_blocking=True)

      predictions = torch.argmax(model(images), dim=1)

      eval_IoU.update(predictions, masks)
  
  iou = eval_IoU.compute().cpu()
  del images, masks, predictions, eval_IoU
  torch.cuda.empty_cache()

  return iou