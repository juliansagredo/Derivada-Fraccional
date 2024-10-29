#=========================
#  Fórmulas de Z-splines 
#=========================
#  Julián T. Sagredo
#  ESFM IPN 
#  Octubre 2024
#=========================

#==============================
#  Módulos: numpy, matplotlib
#==============================
import numpy as np
import matplotlib.pyplot as plt
#==========================
#  Z-splines de algoritmo 
#==========================
from zsmake import zsmake
from zseval import zseval

#=====================================
# Z1 (cúbico): una derivada continua
#=====================================
def z1(x:np.array):
  y:np.array = np.piecewise(x,[np.logical_and(x>-2,x<=-1),np.logical_and(x>-1,x<=0), \
      np.logical_and(x>0,x<=1),np.logical_and(x>1,x<=2)], \
      [lambda x: 0.5*(1.0+x)*(2+x)**2, \
       lambda x: 1-x*x*(2.5 +1.5*x),  \
       lambda x: 1-x*x*(2.5 -1.5*x),  \
       lambda x: 0.5*(1.0-x)*(2-x)**2])   
  return y 

#========================================
# Z2 (quíntico): dos derivadas continuas
#========================================
def z2(x:np.array):
  u12:np.float64 = 1.0/12.0
  y:np.array = np.piecewise(x,[np.logical_and(x>-3,x<=-2),np.logical_and(x>-2,x<=-1), \
      np.logical_and(x>-1,x<=0),np.logical_and(x>0,x<=1),np.logical_and(x>1,x<=2), \
      np.logical_and(x>2,x<=3)], \
      [lambda x: 18.0+u12*x*(459.0+x*(382.5+x*(156.5+x*(31.5+2.5*x)))), \
       lambda x: -4.0+u12*x*(-225.0+x*(-367.5+x*(-272.5+x*(-94.5-12.5*x)))), \
       lambda x: 1.0+x*x*u12*(-15.0+x*(35.0+x*(63.0+x*(25.0)))), \
       lambda x: 1.0+x*x*u12*(-15.0+x*(-35.0+x*(63.0+x*(-25.0)))), \
       lambda x: -4.0+u12*x*(225.0+x*(-367.5+x*(272.5+x*(-94.5+12.5*x)))), \
       lambda x: 18.0+u12*x*(-459.0+x*(382.5+x*(-156.5+x*(31.5-2.5*x))))])
  return y

#==============================
#  Prueba autónoma del código
#==============================
if __name__ == "__main__":
  x = np.linspace(-5,5,301)
  y = z1(x) 
  zs = zsmake(1)
  y2 = zseval(zs,x)
  plt.plot(x,y,lw=5)
  plt.plot(x,y2,color="red")
  plt.show()
  y = z2(x)
  zs = zsmake(2)
  y2 = zseval(zs,x)
  plt.plot(x,y,lw=5)
  plt.plot(x,y2,color="red")
  plt.show()

