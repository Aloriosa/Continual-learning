import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gzip
import pickle
import sys
from copy import deepcopy
import vcl_model
import coreset
import utils
import run_vcl


class PermutedMnistGenerator():
  def __init__(self, max_iter=10):

    f = gzip.open('/content/drive/My Drive/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    self.X_train = np.vstack((train_set[0], valid_set[0]))
    self.Y_train = np.hstack((train_set[1], valid_set[1]))
    self.X_test = test_set[0]
    self.Y_test = test_set[1]
    self.max_iter = max_iter
    self.cur_iter = 0

  def get_dims(self):

    return self.X_train.shape[1], 10

  def next_task(self):

    if self.cur_iter >= self.max_iter:
        raise Exception('Number of tasks exceeded!')
    else:
        np.random.seed(self.cur_iter)
        perm_inds = list(range(self.X_train.shape[1])) #
        np.random.shuffle(perm_inds)

        # Retrieve train data
        next_x_train = deepcopy(self.X_train)
        next_x_train = next_x_train[:,perm_inds]
        next_y_train = np.eye(10)[self.Y_train]

        # Retrieve test data
        next_x_test = deepcopy(self.X_test)
        next_x_test = next_x_test[:,perm_inds]
        next_y_test = np.eye(10)[self.Y_test]

        self.cur_iter += 1

        return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = 100
batch_size = 256
no_epochs = 5
single_head = True
num_tasks = 10

# Run vanilla VCL
np.random.seed(1)

coreset_size = 0
data_gen = PermutedMnistGenerator(num_tasks)
vcl_result = run_vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print('VCL', vcl_result)


# Run random coreset VCL
np.random.seed(1)

coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
rand_vcl_result = run_vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print('Random coreset VCL', rand_vcl_result)

# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
utils.plot(vcl_avg, rand_vcl_avg)

