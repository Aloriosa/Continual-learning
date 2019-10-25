import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
code_size= 100
NTrainPointsMNIST = 60000

class BaseNet(object):
    def __init__(self):
        print('\nBNN:')

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def loglike(self, x, do_sum=True):
        out = -(0.5) * np.log(2*np.pi) - np.log(self.sigma.detach().cpu().numpy()) - (0.5)*((x - self.mu.detach().cpu().numpy())/(self.sigma.detach().cpu().numpy())) ** 2
        if do_sum:
            return out.sum()
        else:
            return out
            
def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    out = -(0.5) * np.log(2 * np.pi) - torch.log(sigma) - (0.5) * (((x - mu)/sigma) ** 2)
    if do_sum:
        return out.sum()
    else:
        return out
      
def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out
    
class BayesLinear_Normalq(nn.Module):
    def __init__(self, n_in, n_out, prior_class):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class
        
        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.2, 0.2))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))
        
        self.lpw = 0
        self.lqw = 0
        
    
                                   
    def forward(self, X, sample=False):
        if not self.training and not sample: # This is just a placeholder function
            output = torch.mm(X, self.W_mu)
            return output, 0, 0                        
        else:
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            # sample parameters         
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)                       
            W = self.W_mu + 1 * std_w * eps_W
            
            output = torch.mm(X, W)
            
            lqw = isotropic_gauss_loglike(W, self.W_mu, std_w)
            lpw = self.prior.loglike(W.detach().cpu().numpy())
            return output, lqw, lpw
            
            
class bayes_linear_2L(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(bayes_linear_2L, self).__init__()
        
        
        self.prior_instance = [isotropic_gauss_prior(mu=torch.zeros((input_dim, hid_dim)), 
                                                     sigma= torch.ones((input_dim, hid_dim))),
                               isotropic_gauss_prior(mu=torch.zeros((hid_dim, hid_dim)), 
                                                     sigma= torch.ones((hid_dim, hid_dim))),
                               isotropic_gauss_prior(mu=torch.zeros((hid_dim, output_dim)), 
                                                     sigma= torch.ones((hid_dim, output_dim)))]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.bfc1 = BayesLinear_Normalq(input_dim, hid_dim, self.prior_instance[0])
        self.bfc2 = BayesLinear_Normalq(hid_dim, hid_dim, self.prior_instance[1])
        self.bfc3 = BayesLinear_Normalq(hid_dim, output_dim, self.prior_instance[2])
        
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x, sample=False):
        tlqw = 0.
        tlpw = 0.
        x, lqw, lpw = self.bfc1(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.act(x)
        x, lqw, lpw = self.bfc2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.act(x)
        y, lqw, lpw = self.bfc3(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        return y, tlqw, tlpw
    
    
class Bayes_Net(BaseNet):
    def __init__(self, lr=1e-3, channels_in=1, side_in=784, hid_dim=100, cuda=True, classes=10, 
                 batch_size=128, Nbatches=0):
        super(Bayes_Net, self).__init__()
        print('Creating BNN')
        self.lr = lr
        self.channels_in = channels_in
        self.side_in=side_in
        self.hid_dim = hid_dim
        self.cuda = cuda
        self.classes = classes
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        
        self.create_net()
        self.create_opt()

    def create_net(self):
        self.model = bayes_linear_2L(input_dim=self.channels_in * self.side_in, 
                                     hid_dim=self.hid_dim, output_dim=self.classes)
        if self.cuda:
            self.model.cuda()

    def create_opt(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
        self.optimizer.zero_grad()
        if samples == 1:
            out, tlqw, tlpw = self.model(x)
            mlpdw = F.cross_entropy(out, y, reduction='sum')
            Edkl = (tlqw - tlpw)/self.Nbatches            
        elif samples > 1:
            mlpdw_cum = 0
            Edkl_cum = 0           
            for i in range(samples):
                out, tlqw, tlpw = self.model(x, sample=True)
                mlpdw_i = F.cross_entropy(out, y, reduction='sum')
                Edkl_i = (tlqw - tlpw)/self.Nbatches
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i  
            mlpdw = mlpdw_cum/samples
            Edkl = Edkl_cum/samples
        loss = Edkl + mlpdw
        loss.backward()
        self.optimizer.step()

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
        out, _, _ = self.model(x)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        acc = float(np.sum(pred.detach().cpu().numpy() == y.detach().cpu().numpy())) / float(len(y))
        return acc, pred
        
        
def solver_train_predict(batch_size, train_loader, test_loader, n_epochs, seen_tasks):
    state_dict_path = '/stored_models/solver_weights.pth'
    net = Bayes_Net(lr=0.001, channels_in=1, side_in=784, hid_dim=100, 
                    cuda=torch.cuda.is_available(), classes=10, 
                    batch_size=batch_size, Nbatches=(NTrainPointsMNIST/batch_size))

    if seen_tasks > 0:
        prev_state_dict = torch.load(state_dict_path)
        net.model.bfc1.prior.mu = prev_state_dict['bfc1.W_mu']
        net.model.bfc1.prior.sigma = prev_state_dict['bfc1.W_p']
        net.model.bfc2.prior.mu = prev_state_dict['bfc2.W_mu']
        net.model.bfc2.prior.sigma = prev_state_dict['bfc2.W_p']
        net.model.bfc3.prior.mu = prev_state_dict['bfc3.W_mu']
        net.model.bfc3.prior.sigma = prev_state_dict['bfc3.W_p']
    net.model.to(device)
    
    #train
    
    accuracy = 0.
    for i in tqdm(range(n_epochs)):
        net.set_mode_train(True)

        for x, y in train_loader:
            net.fit(x.to(device), y.to(device), samples=3)
        
    # test
        
    net.set_mode_train(False)
    for j, (x, y) in enumerate(test_loader):
        acc, _ = net.eval(x.to(device), y.to(device))
        accuracy += acc
    accuracy /= float(j+1)
    
    torch.save(net.model.state_dict(), state_dict_path)

    return net, accuracy        
