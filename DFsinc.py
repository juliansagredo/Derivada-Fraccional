#=====================================
#  Derivada fraccional usando FFT
#  FFT en el GPU
#=====================================
#  Julián T. Sagredo 
#  Septembre 2024
#=====================================

#=============
#  CUDA FFT
#=============
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft

#======================
#  Numpy y Matplotlib
#======================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 

#===============
#  Función sinc
#===============
L = 1000
n = 100000
x = np.linspace(-L/2,L/2,n)
N = len(x)
f = np.sinc(x)

#=========================
#  FFT de la función sinc 
#=========================
fc = cp.complex64(f) 
scipy.fft.register_backend(cufft)
fhat = scipy.fft.fft(fc) 

#======================================
#  Números de onda arreglados para FFT
#======================================
k = (2*np.pi/L)*np.arange(-N/2,N/2)
k = np.fft.fftshift(k)

#============================================
#  Preparar la gráfica para animar derivadas
#============================================
N_plots = 200

def preparar_grafica():
  fig,ax = plt.subplots()
  color = plt.cm.viridis(np.linspace(-1, 1, N_plots+1))
  lns,=ax.plot([],[])
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('Derivada fraccional')
  plt.xlim(-10, 10)
  plt.ylim(-4, 3)
  return fig,lns,color

[fig,lns,color] = preparar_grafica()

#=====================================
#  Derivadas fraccionales espectrales 
#=====================================
np1 = N_plots+1
dx = 2.0/float(N_plots)
Frame = np.empty((np1,N)) 
for i in range(np1):
  alpha = dx*float(i)
  bb = scipy.fft.ifft(((1j*k)**alpha)*fhat)   # FFT Inversa
  Frame[i] = bb.real

#=============
#  Animación 
#=============
def animacion():
  def animate(i):
    lns.set_data(x,Frame[i])
    lns.set_color(color[i])
    return lns
  ani = FuncAnimation(fig, animate, frames=np1, interval=50)
  plt.show()
animacion()

