import numpy as np
class sigmoid(object):
  def __init__(self):
      pass
  def Avance(x):
      y=1/(1+np.exp(-x))
      return y
  def Gradiente(x):
      dx=-np.exp(-1)/np.power(1+np.exp(-x),2)
      return dx

class relu(object):
  def __init__(self):
    pass
  def Avance(x):
    y = np.zeros(x.shape)
    for i in range(x.shape[1]):
      if(x[0, i] > 0):
        y[0, i] = x[0, i]
      else:
        y[0, i] = 0
    return x
  def Gradiente(x):
    y = np.zeros(x.shape)
    for i in range(x.shape[1]):
      if(y[0, i] > 0):
        y[0, i] = 1.0
      else:
        y[0, i] = 0
    return y   
    
