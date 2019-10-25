import numpy as np
import utils
import vcl_model
from vcl_model import VCL



def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True):
  
  in_dim, out_dim = data_gen.get_dims()
  x_coresets, y_coresets = [], []
  x_testsets, y_testsets = [], []
  
  all_acc = np.array([])
  
  for task_id in range(data_gen.max_iter):
    
    x_train, y_train, x_test, y_test = data_gen.next_task()
    x_testsets.append(x_test)
    y_testsets.append(y_test)
    
    head = 0 if single_head else task_id
    bsize = x_train.shape[0] if (batch_size is None) else batch_size
    
    if task_id == 0:
      
      ml_model = VCL(in_dim, hidden_size, output_size=10).
      ml_model.train(x_train, y_train, bsize, no_epochs, task_id)
      torch.save(ml_model.state_dict(), 'my_model.pth')
    
    else: 
    
      ml_model = VCL(in_dim, hidden_size, output_size=10)
      ml_model.load_state_dict(torch.load('my_model.pth'))
      ml_model.bfc3 = vcl_model.BayesLinear(hidden_size, 10)
      ml_model.train(x_train, y_train, bsize, no_epochs, task_id)
      torch.save(ml_model.state_dict(), 'my_model.pth')
    
    if coreset_size > 0:
      x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)
    
    acc = utils.get_scores(ml_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size)
    all_acc = utils.concatenate_results(acc, all_acc)
    
  return all_acc
