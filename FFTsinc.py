#=====================================
#  Transformada de Fourier en el GPU
#=====================================
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt

#===================
#  Función sinc(x) 
#===================
x = np.linspace(-10,10,10000)
y = np.sinc(x)

#============================
#  Graficar función sinc(x) 
#============================
plt.plot(x,y)
plt.show()

#===============================
#  Arreglo en formato complejo 
#===============================
a = cp.complex64(y) 

#===========================
#  Transformada de Fourier 
#===========================
scipy.fft.register_backend(cufft)
b = scipy.fft.fft(a) 
plt.plot(b.real)
plt.show()

#=======================
#  Tranformada inversa 
#=======================
norm = len(x)
bb = scipy.fft.fft(b)/norm
plt.plot(x,bb.real)
plt.show()
