import numpy as np
from layers import *
from optimization import *
from activation import *

#Regression network
class FCN_R(object):
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
        
#Classification Network
class FCN_C(object):
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

  def Guardar(self, folder, pre, it):
    #Generar nombre
    nombre = folder + pre + "_w" + str(it) + ".npy" 

    #Guardar modelo
    np.save(nombre, self.L)

    print("Modelo " + str(nombre) + " salvado en ./" + str(folder))

  def Cargar(self, folder, pre):
    #Generar nombre
    nombre = folder + pre

    #Cargar pesos
    self.L = np.load(nombre)

  def Avance(self,x):
    #Inicializar flujos
    y = x
    self.a = []
    #Pasar por capas ocultas
    for i in range(len(self.L) - 1):
      #Generar Salida
      c = self.L[i].Avance(y)
      #print('Primera capa:' + str(c))
      y = relu.Avance(c)
      #print('ReLu: ' + str(y))
      #print('Gradiente: ' + str(relu.Gradiente(c)))
      #Guardar gradiente
      self.a.append(relu.Gradiente(c))
    
    #Last layer (classification)
    #print('Entrada:' + str(y))
    c = self.L[len(self.L) - 1].Avance(y)
    #print('salida sin act' + str(c))
    self.a.append(np.ones(c.shape))
    y = softmax.Puntaje(c)

    return y

  def Entrenar(self,x,yd,it,Lr, Di, Dr, folder, pre, i_d):
    #Inicializar error
    e_h = np.zeros(it)
    e_a = 0.0

    for i in range(it):
      
      if(i%Di == 0):
        #Update learning rate
        Lr = Dr*Lr

        #Save weights
        self.Guardar(folder, pre, i)
   
      for j in range(x.shape[0]):
        #print('--------')
        #print('Entrada: ' + str(x[j, :]))
        #Paso hacia delante
        y = self.Avance(x[j,:])
        #print('Salida con activacion:' + str(y))
        #print('Posicion correcta:' + str(yd[j]))

        #Calcular Error (softmax)
        e = softmax.Error(y,yd[j])
        e_a+=e
	#Gradiente de ultima capa
        ds = softmax.Gradiente(y,yd[j])	
        #print('Gradiente en la salida:' + str(ds))


        #Numero de capas
        p = len(self.L)

        #Gradiente de capa de salida
        dxs,dws,dbs = self.L[p-1].Gradiente()
        #print(dxs.shape)
        for m in range(y.shape[1]):
          dws[m, :] = dws[m, :]*ds[0, m]
          dbs[m, :] = dbs[m, :]*ds[0, m] 

        #Ajustar pesos de capa de salida
        self.L[p-1].w[:,:] = self.L[p-1].w - Lr*(np.transpose(dws))
        self.L[p-1].b[:] = self.L[p-1].b - Lr*np.transpose(dbs)

        #Gradiente acumulado
        dL = np.dot(ds, dxs)
        #print('Gradiente en x' + str(dxs))
        #print('Gradiente en X:' + str(dL))

        #Gradiente de retropropagacion
        for k in range(p-1):
   
          #Gradiente acumulado
          de = dL      

          #Posicion actual
          p_a = p - k - 2

          #Gradientes de capa actual
          dai = self.a[p_a]
          dxi,dwi,dbi = self.L[p_a].Gradiente()

          #Acumular gradiente 
          for l in range(k):

            #Get gradients of current layer
            dah = self.a[p_a + l + 1]
            dxh,dwh,dbh = self.L[p_a + l + 1].Gradiente()
          
            #Acumular gradiente
            dL = np.dot(dL*dah, dxh)

          #Ajustar pesos de capa actual
          self.L[p_a].w[:,:] = self.L[p_a].w - Lr*(np.transpose(np.dot(dL*dai, dwi)))
          self.L[p_a].b[:] = self.L[p_a].b - Lr*np.transpose(np.dot(dL*dai, dbi))

        y = self.Avance(x[j,:])
        #print('Despues de optimizar:' + str(y))
        #print('--------')
      #Guardar historial del error
      e_h[i] = e_a/x.shape[0]
      e_a = 0.0
      if(i%i_d == 0):
        print('Epoca: ' + str(i) + ', Error:' + str(e_h[i]))
    return e_h

        
