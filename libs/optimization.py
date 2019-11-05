import numpy as np
class emc(object):
  def __init__(self):
    pass
  def Avance(x,xd):
    y = 0.5*np.power(x - xd,2)
    return y
  def Gradiente(x,xd):
    dx = x-xd
    return dx
