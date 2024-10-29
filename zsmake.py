#=============================================
#  Construcción de coeficientes de Z-splines
#=============================================
#  Dr. Julián T. Sagredo
#  ESFM IPN
#  Octubre 2024
#=============================================

#==========================
#  Módulos: numpy y scipy
#==========================
import numpy as np
import scipy.special as sp

#=====================
#  Externos: fdmay.py
#=====================
from fdmat import fdmat 

#==================
#  Objeto Z-spline 
#==================
class Zspline:
  def __init__(self,m,cc):
    self.m = m
    self.cc = cc
  def val(self,x):
    pass
  def vali(self,x,intx):
    pass

#============================
#  Construcción del Z-spline 
#============================
def zsmake(mm):
  #======================
  #  Instanciar Z-spline 
  #======================
  m = mm+1
  q = 2*m
  cc = np.zeros((q,q))
  zs = Zspline(m,cc)

  #================================
  #  Matriz de diferencias finitas
  #================================
  A = np.zeros((m,q+1))
  B = fdmat(m) 
  for i in range(m):
    for j in range(q-1):
      A[i,j+1] = B[i,q-2-j] 

  #=========================================
  #  Polinomio de interpolación de Hermite
  #=========================================
  for j in range(-m,m):
    N = np.zeros((q,q))
    N[0:m,0] = A[0,m+j] 
    N[m:2*m,0] = A[0,m+j+1] 
    for k in range(2,m+1):
      N[0:m-k+1,k-1]   = A[k-1,m+j]  /sp.factorial(k-1) 
      N[m:2*m-k+1,k-1] = A[k-1,m+j+1]/sp.factorial(k-1)
      for i in range(m-k+2,m+1):
        N[i-1,k-1] = N[i,k-2]-N[i-1,k-2]
    for k in range(m+1,2*m+1):
      for i in range(1,2*m-k+2):
        N[i-1,k-1] = N[i,k-2]-N[i-1,k-2] 
    zs.cc[j+m,:] = N[0,:] 
  #=======================
  #  Fórmula del Z-spline 
  #=======================
  return zs 

#=================================
#  Prueba autónoma del algoritmo
#=================================
if __name__ == "__main__":
   zs = zsmake(2) 
   print(zs.cc)
