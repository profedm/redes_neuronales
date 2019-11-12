import numpy as np

class FCL(object):
    def __init__(self,n,N,mu,std):
        #Inicializan atributos
        self.w = std*np.random.randn(n,N)
        self.b = np.zeros([1,N])
        self.dw = np.zeros([n,1])
        self.db = np.zeros([1,N])
        self.s = np.zeros([1,N])
    def Avance(self,x):
        #Proceso de entrada
        #print('w:' + str(self.w))
        #print('b:' + str(self.b))
        #print('x:' + str(x))
        #print('r: ' + str(np.dot(x,self.w)))
        self.s[0,:] = np.dot(x,self.w)+self.b
        #Calcular Gradiente
        #for i in range(self.dw.shape[1]):
        self.dw[:,0] = x
        self.db = np.ones([1,self.w.shape[1]])
        self.dx = self.w
        return self.s
    def Gradiente(self):
        dx = np.transpose(self.dx)
        dw = np.transpose(self.dw)
        db = np.transpose(self.db) 

        return dx, dw, db
    
