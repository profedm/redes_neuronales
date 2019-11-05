import numpy as np

class FCL(object):
    def __init__(self,n,N,mu,std):
        #Inicializan atributos
        self.w = np.random.normal(loc=mu,scale=std,size=(n,N))
        self.b = np.random.normal(loc=mu,scale=std,size=N)
        self.dw = np.zeros([n,N])
        self.db = np.zeros([1,N])
        self.s = np.zeros([1,N])
    def Avance(self,x):
        #Proceso de entrada
        self.s[0,:] = np.dot(x,self.w)+self.b
        #Calcular Gradiente
        for i in range(self.dw.shape[1]):
          self.dw[:,i] = x
        self.db = np.ones([1,self.w.shape[1]])
        self.dx = self.w
        return self.s
    def Gradiente(self):
        return self.dx,self.dw,self.db
    
