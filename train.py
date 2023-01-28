from typing import List, Tuple
import torch
import numpy as np
from tqdm import tqdm

from config import device

def validation(model: torch.nn.modules.module,
               val_loader:torch.utils.data.dataloader,
               loss_fn: torch.nn.modules.loss) -> float:
  '''
  compute validation loss with corresponding model and loss function
  return validation loss
  '''

  model.eval()
  total_loss = 0 
  with torch.no_grad():
    for data, targets in tqdm(val_loader, desc= 'Validation'):
      data = data.to(device)
      targets = targets.to(device)

      scores = model(data)

      loss = loss_fn(scores,targets)
      total_loss += loss.item()
  model.train()
  return total_loss / len(val_loader)


def training(model: torch.nn.modules.module,
             train_dataset: torch.utils.data.dataloader,
             val_dataset: torch.utils.data.dataloader,
             criterion: torch.nn.modules.loss,
             optimizer:torch.optim,
             scheduler=None,
             epochs: int = 5) -> Tuple[List]:


  '''
  train model with given train set and optimizer
  return train and validation losses
  '''
  train_losses = []
  val_losses = []
  model.train()

  for epoch in tqdm(range(epochs) ):

    for  data, targets in tqdm(train_dataset,desc = f'Epoch {epoch + 1}'):
      # convert data and target to device
      data = data.to(device)
      targets = targets.to(device)

      # forward pass
      scores = model(data)
      loss = criterion(scores,targets)

      # backward pass
      optimizer.zero_grad()
      loss.backward()
      # update weights
      optimizer.step()
      
    if scheduler is not None:
      scheduler.step()

    #validation
    val_loss = validation(model,val_dataset,criterion)
    train_losses.append(loss.item())
    val_losses.append(val_loss)
    print(f'Train Loss: {np.mean(train_losses):.2f}, ',end = '')
    print(f'Validation Loss: {np.mean(val_losses):.2f}')

  return train_losses, val_losses



def inference(model: torch.nn.modules.module,
              test_dataset: torch.utils.data.dataloader) -> float:

  '''
  inference model on test set
  return test accuracy
  '''

  n_samples = 0
  n_correct = 0
  model.eval()

  with torch.no_grad():
    for data, targets in tqdm(test_dataset):
      data = data.to(device)
      targets = targets.to(device)

      scores = model(data)
      _, preds = scores.max(1)
      n_correct += (preds == targets).sum().item()
      n_samples += targets.size(0)

    return n_correct / n_samples