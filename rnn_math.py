# rnn_math.py
import numpy as np

# ==== RNN Parameters ====
Wax = np.array([[0.5, -0.2],[0.3, 0.8]])
Waa = np.array([[0.4,  0.1],[-0.3, 0.2]])
Wya = np.array([[1.0, -1.0],[0.5,  1.0]])
ba  = np.array([0.1, -0.2])
by  = np.array([0.0,  0.0])

def tanh(x): 
    return np.tanh(x)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

def rnn_step(x, a):
    a_new = tanh(Waa.dot(a) + Wax.dot(x) + ba)
    y_new = softmax(Wya.dot(a_new) + by)
    return a_new, y_new

def run_rnn_sequence(inputs):
    """
    inputs: list of input vectors (np.array)
    
    Returns a list of dictionaries per time step:
      Each dictionary contains:
        'x': input vector,
        'a_prev': previous hidden state,
        'a': current hidden state,
        'y': output,
        'Wax', 'Waa', 'Wya', 'ba', 'by' (all constant matrices/vectors)
    """
    a_prev = np.zeros(Waa.shape[0])
    history = []
    
    for t, x in enumerate(inputs, start=1):
        a_new, y_new = rnn_step(x, a_prev)
        history.append({
            't': t,
            'x': x,
            'a_prev': a_prev,
            'a': a_new,
            'y': y_new,
            'Wax': Wax,
            'Waa': Waa,
            'Wya': Wya,
            'ba': ba,
            'by': by,
        })
        a_prev = a_new
    return history
