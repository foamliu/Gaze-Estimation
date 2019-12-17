import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
im_size = 224
channel = 3

# Training parameters
batch_size = 32
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_samples = 42813
num_train = int(num_samples * 0.9)

DATA_DIR = 'data'
IMG_DIR = 'data/imgs'
