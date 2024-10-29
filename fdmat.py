#===========================================================
#  Función para generar las matrices de diferencias finitas
#===========================================================
#  Dr. Julián T. Sagredo
#  ESFM IPN   
#  Octubre 2024
#===========================================================

#=========================
#  Módulos: numpy y scipy 
#=========================
import numpy as np
import scipy.special as sp 

#===================================================================
#  Genera la matriz de diferencias finitas para 2m-1 puntos vecinos 
#===================================================================
def fdmat(m):
  #=============================
  #  Conjunto de puntos vecinos
  #=============================
  q = 2*m-1
  s = np.linspace(-m+1,m-1,q)
  #=========================
  #  Polinomios de Lagrange 
  #=========================
  P = np.zeros((q,q))
  for k in range(q):
    p1 = (0,1) 
    for l in range(k): 
      p1 = np.convolve(p1,(1,-s[l]))
    for l in range(k+1,q):
      p1 = np.convolve(p1,(1,-s[l])) 
    P[k,:] = p1[-q:None]
  P = np.transpose(P)
  #================================
  #  Matriz inversa de Vandermonde 
  #================================
  U = np.zeros((q,q))
  for k in range(q):
    U[k,:] = P[q-1-k,:] 
  W = np.zeros((q,q)) 
  for k in range(q):
    W[:,k] = ((-1.0)**(q-k-1))/(sp.factorial(q-k-1)*sp.factorial(k))
  V = W*U
  #===================================
  #  Diagonal de factoriales inversos 
  #===================================
  D = np.diag([sp.factorial(k) for k in range(q)]) 
  #================================
  #  Matriz de diferencias finitas
  #================================
  A = np.matmul(D,V) 
  return A 

#=============================
#  Prueba autónoma del código
#=============================
if __name__ == "__main__":
  #for k in range(4):
    A1 = fdmat(2) 
    print("A1 = \n",A1)
    A2 = fdmat(3)
    print("A2 = \n",A2)
