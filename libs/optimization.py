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

class softmax(object):

  def __init__(self):
    pass

  def Puntaje(x):
    #Inicializar arreglo de salida
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
      y[i] = np.exp(x[i])/np.sum(np.exp(x))

    return y

  def Error(x, xd):
    y = softmax.Puntaje(x)
    e = -np.log(y[0, int(xd)])
    return e

  def Gradiente(x, xd):
    #Inicializar salida
    g = np.zeros(x.shape)

    #Obtener salidas
    y = softmax.Puntaje(x)

    #Puntaje de salida correcta
    y_d = y[0, int(xd)]

    #Calcular gradiente
    for i in range(x.shape[1]):
      
      #Gradiente de clase correcta
      if(i == int(xd)):
        g[0, i] = y_d - 1
      #Gradiente de clase incorrecta
      else:
        g[0, i] = y[0, i]

    return g



  
