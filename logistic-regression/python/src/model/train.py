import numpy as np

def sigmoid(x): 
    h = 1/(1+np.exp(-x))
    return h


