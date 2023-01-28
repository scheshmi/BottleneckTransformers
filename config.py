
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 32
learning_rate = 1e-3
momentum = 0.9
EPOCHS = 15