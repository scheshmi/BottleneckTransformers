
from model import Block, BoTNet
from config import *
from dataset import get_dataset

import torch.optim as optim
from torch import nn
from train import training


def main():

    train_loader, test_loader = get_dataset('/content/dataset/seg_train/seg_train','/content/dataset/seg_test/seg_test')

    num_classes = len(train_loader.dataset.classes)

    botnet = BoTNet(Block, [3, 4, 6, 3], in_channels=3, num_classes=num_classes, shape=(224,224), heads=4)
    botnet.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(botnet.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)


    train_losses, val_losses = training(botnet,train_loader,test_loader,criterion ,optimizer,lr_scheduler,epochs= 20)




if __name__ == "__main__":
    main()

