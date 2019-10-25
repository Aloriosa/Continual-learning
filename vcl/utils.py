import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from vcl_model import VCL




def merge_coresets(x_coresets, y_coresets):
  
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
        
    return merged_x, merged_y

def get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size=None):
    
    torch.save(model.state_dict(), 'model.pth')
    acc = []

    if single_head:
      if len(x_coresets) > 0:
        x_train, y_train = merge_coresets(x_coresets, y_coresets)
        bsize = x_train.shape[0] if (batch_size is None) else batch_size
        final_model = VCL(x_train.shape[1], hidden_size, y_train.shape[1])
        final_model.load_state_dict(torch.load('model.pth'))
        final_model.train(x_train, y_train, bsize, no_epochs, task_id=0)
        torch.save(model.state_dict(), 'model.pth')
      else:
        final_model = model

    for i in range(len(x_testsets)):
      if not single_head:
        if len(x_coresets) > 0:
          x_train, y_train = x_coresets[i], y_coresets[i]
          bsize = x_train.shape[0] if (batch_size is None) else batch_size
          final_model = VCL(x_train.shape[1], hidden_size, y_train.shape[1])
          final_model.load_state_dict(torch.load('model.pth'))
          final_model.train(x_train, y_train, bsize, no_epochs, task_id=i)
          torch.save(model.state_dict(), 'model.pth')
        else:
          final_model = model

      head = 0 if single_head else i
      x_test, y_test = x_testsets[i], y_testsets[i]

      x_test = Variable(torch.from_numpy(x_test))

      pred = final_model.forward(x_test)
      pred_y = np.argmax(pred.detach().cpu().numpy(), axis=1)
      y = np.argmax(y_test, axis=1)
      cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
      acc.append(cur_acc)
        
    return acc

def concatenate_results(score, all_score):
  
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
        
    return all_score

def plot(vcl, rand_vcl):
  
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    plt.figure(figsize=(7,3))
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.xticks(range(1, len(vcl)+1))
    plt.ylabel('Average accuracy')
    plt.xlabel('# tasks')
    plt.legend()
    plt.show()
