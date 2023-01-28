from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import device


def compute_confusion_matrix(model,test_dataset)-> None:
  
  model.eval()
  predicted = []
  labels = []
  with torch.no_grad():
    for data, targets in(test_dataset):
      data = data.to(device)
      
      scores = model(data)
      _, preds = scores.max(1)
      predicted = np.concatenate((predicted,preds.cpu().numpy()))
      labels = np.concatenate((labels,targets.cpu().numpy()))
      
  model.train()

  confusion_mat = confusion_matrix(labels,predicted,normalize = 'true')
  fig, ax = plt.subplots(figsize=(15,15))
  ConfusionMatrixDisplay(confusion_mat, display_labels=test_dataset.dataset.classes).plot(xticks_rotation= 'vertical', ax = ax,colorbar = False, cmap =plt.cm.Blues)
  # plt.savefig('confusion_matrix.jpg')
  