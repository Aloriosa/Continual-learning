import numpy as np



""" Random coreset selection """
def rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size):
  
    # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    idx = np.random.choice(x_train.shape[0], coreset_size, False)
    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx,:])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    
    return x_coreset, y_coreset, x_train, y_train

def update_distance(dists, x_train, current_id):
  
    for i in range(x_train.shape[0]):
        current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
        dists[i] = np.minimum(current_dist, dists[i])
        
    return dists
