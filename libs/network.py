import numpy as np
from layers import *
from optimization import *
from activation import *

class FCN(object):
  def __init__(self,n,L,N,s,mu,std):
    #Inicializar Topologia
    self.L = []
    self.a = []
    #Flujo de informacion
    N.insert(0,n)
    N.append(s)
    print(N)
    #Generar Arreglos
    for i in range(L+1):
      #Inicializar Capa i
      self.L.append(FCL(N[i], N[i+1],mu,std))
  def Avance(self,x):
    #Inicializar flujos
    y = x
    self.a = []
    #Pasar por capas ocultas
    for i in range(len(self.L)):
      #Generar Salida
      c = self.L[i].Avance(y)
      y = sigmoid.Avance(c)
      #print(y.shape)
      #Guardar gradiente
      self.a.append(sigmoid.Gradiente(c))
    return y
  def Entrenar(self,x,yd,it,Lr):
    #Inicializar error
    e_h = np.zeros(it)
    e_a = 0.0

    for i in range(it):
      for j in range(x.shape[0]):
        #Paso hacia delante
        #print('Avance')
        #print(x.shape)
        y = self.Avance(x[j,:])
        #Calcular Error
        e = emc.Avance(y,yd[j])
        e_a+=e
        #Gradiente del error
        de = emc.Gradiente(y,yd[j])
	#Numero de capas
        p = len(self.L)
        for k in range(p):
          #Gradiente acumulado
          dL=de
          #print('Gradientes: ' + str(k))
          #Iterar recursivamente
          for l in range(k):
            #Position actual
            p_a = p-l-1
            #Obtener Gradiente
            dai = self.a[p_a]
            dxi,dwi,dbi = self.L[p_a].Gradiente()
            #Acumular Gradiente
            dL = dL*dai*dxi
          #Gradiente Capa Actual
          dak = self.a[p-k-1]
          dxk,dwk,dbk = self.L[p-k-1].Gradiente()
          #Modificar Pesos
          self.L[p-k-1].w[:,:] = self.L[p-k-1].w - Lr*np.transpose(np.dot(dL*dak,np.transpose(dwk)))
          self.L[p-k-1].b[:] = self.L[p-k-1].b - Lr*np.transpose(np.dot(dL*dak,np.transpose(dbk)))

      #Guardar historial del error
      e_h[i] = e_a/x.shape[0]
      e_a = 0.0
    return e_h
        
