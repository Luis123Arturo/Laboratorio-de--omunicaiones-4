import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.optimize import lsq_linear

fs, x = wavfile.read('x2_U017.wav')  # Read x2_U017.wav

# Get the size of the audio vector
lx = len(x)

fs, y = wavfile.read('y2_U017.wav')  # Read y2_U017.wav

# Plot the vectors
n = np.arange(lx)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.stem(n, x)
ax1.set_title('X[n]')
ax2.stem(n, y)
ax2.set_title('Y[n]')
plt.show()

# Calculate coefficients
lx = 5000
q = np.arange(lx)
n = np.arange(len(x))

# Solve system of simultaneous equations
c1 = np.zeros(lx-2000)
c2 = np.zeros(lx-2000)
b1 = np.zeros(lx-2000)

for i in range(2001, lx):
    c1[i-2001] = x[i-2001]
    c2[i-2001] = -y[i-751]
    b1[i-2001] = y[i] - x[i]

D = np.vstack((c1, c2)).T
R = np.linalg.lstsq(D, b1, rcond=None)[0]
print("The values for each of the coefficients are:")
print("A=", R[0])
print("B=", R[1])

# Comparison of values for positions between [2001] and [20016]
print("Comparison of values for positions between [2001] and [20016]")
print("First column: Found function Second row: Original function Third row: Noise s[n]")
m = np.zeros(16)
l = np.zeros(16)
s = np.zeros(16)

for j in range(2001, 2017):
    m[j-2001] = x[j] + R[0]*x[j-2000] + R[1]*y[j-750]
    l[j-2001] = y[j]
    s[j-2001] = l[j-2001] - m[j-2001]

M = np.transpose([m])
L = np.transpose([l])
S = np.transpose([s])
Comparacion = np.hstack((M, L, S))

b = [1, R[0]]
a = [1, R[1]]

# Fractional partial coefficients
r, p, k = signal.residuez(b, a)
r = r.reshape((-1,1))
p = p.reshape((-1,1))

HH = np.zeros((lx,), dtype=np.complex128)
for i in range(1,lx,1):
    z = i
    HH[i] = np.sum(r/(1-p*z**(-1))) + np.sum(k*np.array([1, z**(-1)]))

# Spectral analysis
f = 1/10  # Frequency of 0.1Hz
N1 = 30  # Number of samples
N2 = 120

# Spectral analysis for X
X1 = np.abs(np.fft.fft(x, N1))
X2 = np.abs(np.fft.fft(x, N2))

F1x = np.arange(N1-1)/N1
F2x = np.arange(N2-1)/N2

# Spectral analysis for Y
Y1 = np.abs(np.fft.fft(y, N1))
Y2 = np.abs(np.fft.fft(y, N2))

F1y = np.arange(N1-1)/N1
F2y = np.arange(N2-1)/N2

# Grafica de las funciones

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.stem(n, x)
ax1.set_title('X[n]')
ax2.stem(n, y)
ax2.set_title('Y[n]')
plt.show()

# Espectros para X[n] con N=30 y N=120
F1x = np.linspace(0, 1, 30)
F2x = np.linspace(0, 1, 120)

# Espectros para Y[n] con N=30 y N=120
F1y = np.linspace(0, 1, 30)
F2y = np.linspace(0, 1, 120)

# Parte real e imaginaria de H[z]
b = np.array([1, 2, 3])
a = np.array([1, -0.5, 0.1])
q = np.arange(0, 50)
HH = np.fft.fft(np.concatenate([b, np.zeros(len(q)-len(b))])) / np.fft.fft(np.concatenate([a, np.zeros(len(q)-len(a))]))

# Gráficas de los espectros
fig3, axs3 = plt.subplots(2, 2, figsize=(10, 8))
axs3[0, 0].stem(F1x, X1, '.')
axs3[0, 0].set_title('Espectro para X[n] con N=30')
axs3[0, 1].stem(F2x, X2, '.')
axs3[0, 1].set_title('Espectro para X[n] con N=120')
axs3[1, 0].stem(F1y, Y1, '.')
axs3[1, 0].set_title('Espectro para Y[n] con N=30')
axs3[1, 1].stem(F2y, Y2, '.')
axs3[1, 1].set_title('Espectro para Y[n] con N=120')

# Parte real e imaginaria de H[z]
fig, axs = plt.subplots(2, 1, figsize=(8,8))
axs[0].stem(q, np.real(HH))
axs[0].set_title('Parte real H[z]')
axs[1].stem(q, np.imag(HH))
axs[1].set_title('Parte imaginaria H[z]')
plt.show()

# Respuesta al impulso y al escalón unitario
y = signal.lfilter(b, a, np.arange(lx))

plt.figure(figsize=(12,6))
plt.plot(y)
plt.title('Respuesta al impulso y al escalon unitario')
plt.show()

# Grafica el diagrama de polos y ceros
z, p, _ = signal.tf2zpk(b, a)
plt.figure(6)
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r')
plt.scatter(np.real(p), np.imag(p), marker='x', color='b')
plt.title('Diagrama de Polos y Ceros')
plt.show()


