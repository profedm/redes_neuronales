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
    