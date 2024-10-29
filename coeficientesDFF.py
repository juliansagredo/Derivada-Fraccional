#=====================================
#  Cálculo de Coeficientes para 
#  Diferencias Finitas de Derivadas
#  Fraccionales 
#=====================================
#  Julián T. Sagredo 
#  Septembre 2024
#=====================================

#=============
#  CUDA FFT
#=============
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft as sfft

#======================
#  Numpy y Matplotlib
#======================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from zsmake import zsmake, Zspline
from zseval import zseval

def coeficientesDFF(alpha=0.5,mol=[4,12],spline=3):
  # Tamaño de la molécula numérica [-A,B]  
  A = mol[0]
  B = mol[1]
  # Dominio [-L/2,L/2] 
  L = 1000
  # Nodos por unidad
  l = 32
  # Número de nodos
  N = L*l+1
  # Malla 1D
  X = np.linspace(-L/2,L/2,N)

  #=========================
  #  FFT de la función base 
  #=========================
  zs = zsmake(spline)
  f = zseval(zs,X) 
  fc = cp.complex128(f)
  sfft.register_backend(cufft)
  fhat = sfft.fft(fc)
  #===============================================
  #  Números de onda arreglados para derivada FFT
  #===============================================
  k = (2*np.pi/L)*np.arange(-N/2,N/2)
  k = np.fft.fftshift(k)
  #==============================
  #  Extracción de Coeficientes 
  #==============================
  A0 = int(L/2-A) 
  cc = np.zeros((A+B)+1)
  #===============================
  #  Derivada fraccional de sinc
  #===============================
  Daf = sfft.ifft(((1j*k)**alpha)*fhat)
  #=======================
  # Guardar coeficientes
  #=======================
  ii:int=0
  for iii in range(A0*l,(A0+A+B+1)*l,l):
    cc[ii] = Daf[iii].real
    ii += 1
  return cc

if __name__ == "__main__":
  cc = coeficientesDFF(1.0,[4,12],2)
  print(cc) 
