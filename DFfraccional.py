#=====================================
#  Diferencias Fintas Fraccionales
#  FFT en el GPU
#=====================================
#  Julián T. Sagredo 
#  Octubre 2024
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
#==================
#  Módulo zplines
#==================
from zsmake import zsmake, Zspline
from zseval import zseval
from coeficientesDFF import coeficientesDFF as cDFF

funcion = 1

#===============
#  Función sinc
#===============
L = 250
n = 2001
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

#====================
#  FFT de la función 
#====================
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
  lns2,=ax.plot([],[])
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('Derivada fraccional')
  plt.xlim(-10, 10)
  plt.ylim(-1.5, 1.5)
  return fig,lns,lns2,color

[fig,lns,lns2,color] = preparar_grafica()

#=====================================
#  Derivadas fraccionales espectrales 
#=====================================
np1 = N_plots+1
final = 1.0
dx = final/float(N_plots)
Frame = np.empty((np1,N)) 
Frame2 = np.empty((np1,N))
A = 2 
B = 50
dd = np.zeros(n)
for i in range(np1):
  alpha = dx*float(i)
  bb = sfft.ifft(((1j*k)**alpha)*fhat)   # FFT Inversa
  cc = cDFF(alpha,[A,B],2) 
  dd = (((n-1)/L)**alpha)*np.convolve(g,cc,"same")
  if (B-A)>0:
    dd[int((A+B)/2):n-int((A+B)/2)] = dd[A:n-B]
  Frame[i] = bb.real
  Frame2[i] = dd

#=============
#  Animación 
#=============
def animacion():
  def animate(i):
    lns.set_data(x,Frame[i])
    lns.set_color("purple")
    lns.set_linewidth(6)
    lns2.set_data(x,Frame2[i])
    lns2.set_color("orange")
    lns2.set_linewidth(1)
    return lns,lns2
  ani = FuncAnimation(fig, animate, frames=np1, interval=50)
  plt.show()
animacion()
h = np.gradient(g,x)
plt.plot(x,bb.real,color="blue",linewidth=10)
plt.plot(x,dd,color="red",linewidth=5)
plt.plot(x,h,color="yellow")
plt.xlim(-10,10)
plt.grid()
plt.show()
