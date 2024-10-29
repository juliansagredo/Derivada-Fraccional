#============================================
#  Evaluación de cualquier Z-spline cardinal
#============================================
#  Julián T. Sagredo
#  ESFM IPN Octubre 2024
#============================================

#=============================
#  Módulos: numpy, matplotlib
#=============================
import numpy as np
import matplotlib.pyplot as plt
#===================
#  Externos: zsmake
#===================
from zsmake import zsmake,Zspline

#==================================================
#  Función zseval: 
#    evaluación del zspline zs en un conjunto de x 
#==================================================
def zseval(zs:Zspline,x:np.array):
  m = zs.m
  cc = zs.cc
  n = len(x) 
  intx = [np.empty(0,dtype=np.int16) for i in range(2*m)]
  #=======================
  #  Índices de los datos  
  #=======================
  for i in range(n):
    j = int(np.floor(x[i]))
    if j>=-m and j<m:
      intx[j+m] = np.append(intx[j+m],i)
  #===================================
  #  Evaluación usando factorización 
  #===================================
  y = np.zeros(len(x))
  for j in range(-m,m):
    p1 = j*np.ones(m,dtype=np.int16)
    p2 = (j+1)*np.ones(m,dtype=np.int16)
    P = np.concatenate((p1,p2))
    y[intx[j+m]] = cc[j+m,2*m-1]
    #========================
    #  Pedazos del polinomio 
    #========================
    for p in range(2*m-1):
      y[intx[j+m]] = y[intx[j+m]]*(x[intx[j+m]]-P[2*m-2-p]) + cc[j+m,2*m-2-p]
  return y

#=============================
#  Prueba autónoma del código 
#=============================
if __name__ == "__main__":
   x = np.linspace(-5,5,201)
   for i in range(1,20,5):
     zs = zsmake(i)
     y = zseval(zs,x) 
     plt.plot(x,y,color="black")
   plt.show()
