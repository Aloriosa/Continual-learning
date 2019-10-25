import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np


class BayesLinear(nn.Module):
  
  def __init__(self, input_size, output_size):
    super(BayesLinear, self).__init__()
    
    self.inp = input_size
    self.out = output_size
    
    self.w_mu = nn.Parameter(1e-6 + torch.zeros(self.out, self.inp))
    self.w_var = nn.Parameter(0.1 * torch.ones(self.out, self.inp))
    self.w = torch.distributions.normal.Normal(self.w_mu, 1)
    
    self.w_prior_mu = torch.Tensor([0.])
    self.w_prior_var = torch.Tensor([1.])    
    
  def forward(self, x, sampling=False, calculate_log_probs=False):
    
    if self.training or sampling:
      w = self.w.sample()
    else:
      w = self.w_mu
        
    return F.linear(x, w)
  
  def loss_layer(self, x):
    
    cte_term = -0.5 * np.log(2 * np.pi)
    det_sig_term = -torch.log(self.w_var)
    inner = (x - self.w_mu) / self.w_var
    dist_term = -0.5 * (inner**2)    
    out = (cte_term + det_sig_term + dist_term).sum()
    
    return out
  
  def kl(self):
    
    const_term = -0.5 * self.inp * self.out
    log_std_diff = 0.5 * torch.sum(self.w_prior_var - self.w_var)
    mu_diff_term = 0.5 * torch.sum((torch.exp(self.w_var) + (self.w_prior_mu - self.w_mu)**2) / self.w_prior_var)    
    kl_layer = const_term + log_std_diff + mu_diff_term
    
    return kl_layer
        

class VCL(nn.Module):
  
  def __init__(self, input_size, hidden_size, output_size):
    super(VCL, self).__init__()
    
    self.inp = input_size
    self.hid = hidden_size
    self.out = output_size
    
    self.bfc1 = BayesLinear(self.inp, self.hid)
    self.bfc2 = BayesLinear(self.hid, self.hid)
    self.bfc3 = BayesLinear(self.hid, self.out)
    
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()
    
  def forward(self, x, sampling=False):
    
    x = self.relu(self.bfc1(x, sampling=sampling))
    x = self.relu(self.bfc2(x, sampling=sampling))
    out = self.softmax(self.bfc3(x, sampling=sampling))
    
    return out
  
  def logpred(self, predictions, targets):
    
    error = nn.CrossEntropyLoss(reduction='mean') 
    log_lik = - torch.mean(error(predictions, torch.max(targets, 1)[1])) 
    
    return log_lik
  
  def kl_calc(self):
    
    return self.bfc1.kl() + self.bfc2.kl() + self.bfc3.kl()  
  
  def train(self, x, y, batch_size, n_epochs, task_id, display_epoch=20):
  
    print('Task is:', task_id)
    optim = torch.optim.Adam(self.parameters(), lr=0.001)
    N = x.shape[0]
    if batch_size > N:
      batch_size = N
    
    costs = []

    for epoch in range(n_epochs):

      perm_inds = list(range(x.shape[0]))
      np.random.shuffle(perm_inds)
      cur_x = x[perm_inds]
      cur_y = y[perm_inds]     
      avg_cost = 0.
      total_batch = int(np.ceil(N * 1.0 / batch_size))
      
      for i in range(total_batch):

        start_ind = i*batch_size
        end_ind = np.min([(i+1)*batch_size, N])
        batch_x = cur_x[start_ind:end_ind, :]
        batch_y = cur_y[start_ind:end_ind, :]        
        batch_x = Variable(torch.from_numpy(batch_x))
        batch_y = Variable(torch.from_numpy(batch_y))       
        optim.zero_grad()   
        outputs = self.forward(batch_x)       
        log_pred = self.logpred(outputs, batch_y)
        kl = self.kl_calc()
        loss = kl / batch_y.shape[0] - log_pred 
        loss.backward()
        optim.step()
        avg_cost += loss.item() / total_batch

      if epoch % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost =", \
                "{:.9f}".format(avg_cost))

      costs.append(avg_cost)

    print("Optimization Finished!")

    return costs  
  
