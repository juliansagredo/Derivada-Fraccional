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
import scipy.fft as sfft

#======================
#  Numpy y Matplotlib
#======================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 
from zsmake import zsmake, Zspline
from zseval import zseval

funcion = 3

#===============
#  Función sinc
#===============
L = 500
n = 100001
x = np.linspace(-L/2,L/2,n)
N = len(x)
if (funcion == 1):
  g = np.sinc(x)

#=============================
#  Funciones seno y Gaussiana 
#=============================
# Seno
if (funcion == 2):
  g = np.sin(2.0*np.pi*x/5.0)
# Gaussiana
if (funcion == 3):
  g = np.exp(-x*x/1.0)

#=================================
#  Z-spline de continuidad m
#=================================
#  zs = zsmake(m): coeficientes
#  y = zseval(zs,x): evaluación
#=================================
zs = zsmake(3)
if (funcion == 4):
   g = zseval(zs,x)

#=========================
#  FFT de la función sinc 
#=========================
fc = cp.complex128(g) 
sfft.register_backend(cufft)
fhat = sfft.fft(fc) 

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
  plt.ylim(-1.5, 1.5)
  return fig,lns,color

[fig,lns,color] = preparar_grafica()

#=====================================
#  Derivadas fraccionales espectrales 
#=====================================
np1 = N_plots+1
final = 1.0
dx = final/float(N_plots)
Frame = np.empty((np1,N)) 
for i in range(np1):
  alpha = dx*float(i)
  bb = sfft.ifft(((1j*k)**alpha)*fhat)   # FFT Inversa
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
h = np.gradient(g,x)
plt.plot(x,bb.real,color="blue",linewidth=5)
plt.plot(x,h,color="yellow")
plt.xlim(-10,10)
plt.grid()
plt.show()
