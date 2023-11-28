import torch

learning_rate = .001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 5
num_epochs = 5
num_workers = 1
image_scaling = 8