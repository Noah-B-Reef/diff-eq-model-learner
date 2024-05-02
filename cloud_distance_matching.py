import numpy as np


def sumsq_inf_distance(dataset, reference):
    '''
    Sum of square distances from the dataset to a reference dataset
    in a minimal way:
    
    sum_i min_j || x[i] - y[j] ||_2
    
    i runs over rows of dataset;
    j runs over rows of reference.
    
    Datasets do not need to have the same number of data points.
    
    Works through scipy.spatial.distance_matrix and np.min.
    
    Inputs:
        dataset:   2D array; shape (m,d)
        reference: 2D array; shape (n,d)
    Outputs:
        s: float; the metric above.
    '''
    from scipy import spatial
    
    D = spatial.distance_matrix(dataset, reference)
    min_distances = np.min(D, axis=1)
    s = sum(min_distances)
    
    return s

if __name__=="__main__":
    '''
    Demo and proof of concept.
    '''
    n = 100

    # dataset 1
    t = np.linspace(0, 2*np.pi, n)
    X1 = np.vstack([np.cos(t), np.sin(t)]) + np.random.normal(0,0.1, (2,n))

    # a second dataset
    t_shuff = t[np.random.permutation(n)]
    X2 = np.vstack([np.cos(t), np.sin(t)]) + np.random.normal(0,0.1, (2,n))
    
    ### try to use a subset for demo  purposes.
    #X2 = X2[:, np.random.permutation(n)[:n//2]]

    # Arrange data points as rows 
    X1 = X1.T
    X2 = X2.T

    val = sumsq_inf_distance(X1, X2)
    
    # average per-datapoint distance.
    print('total distance: ', val)
    print('average distance per datapoint:', val/n)
    print('naive average distance: ', sum(np.linalg.norm(X1-X2, 2, axis=1))/n )
    
    ###
    from matplotlib import pyplot as plt
    
    fig,ax = plt.subplots(constrained_layout=True)
    ax.scatter(X1[:,0], X1[:,1], label='dataset')
    ax.scatter(X2[:,0], X2[:,1], label='reference')
    ax.set(xlabel='x', ylabel='y', aspect='equal')
    ax.legend()
    
    fig.show()
    
