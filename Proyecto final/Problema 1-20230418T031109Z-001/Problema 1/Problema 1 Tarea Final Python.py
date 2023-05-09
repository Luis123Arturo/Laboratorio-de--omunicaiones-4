import numpy as np
import random
import matplotlib.pyplot as plt
from sympy import *
from scipy.signal import residue
from scipy import signal
from tabulate import tabulate



#Vector x[n]
#x = np.random.randint(low=-5, high=5, size=31).tolist()
x=[1, 0, -3, -5, -2, -5, -4, -3, 1, -3, -2, 3, 0, 0, -5, -4, 3, 3, -1, -1, 4, 3, -3, -3, 1, -4, 3, 2, 4, -5, 0]
n = np.arange(0, 31).tolist()
x[30]=0
print(x)
#Vector y[n]
y = (np.cos(x)+2).tolist()

plt.figure(1)
plt.subplot(2,1,1)
plt.stem(x)
plt.title('x[n]')

plt.subplot(2,1,2)
plt.stem(y)
plt.title('y[n]')



#función [b,a] = CalculoCoef1(x,y)
#Solución de sistema tipo AX=B
#Definición de matriz de coeficiente
A = []
A = [
    [x[10], x[9],  x[8],  x[7],  x[6],  x[5],  x[4],  x[3],  x[2],  x[1],  x[0],  -y[9],  -y[8],  -y[7],  -y[6],  -y[5],  -y[4],  -y[3],  -y[2],  -y[1],  -y[0]],
    [x[11], x[10], x[9],  x[8],  x[7],  x[6],  x[5],  x[4],  x[3],  x[2],  x[1],  -y[10], -y[9],  -y[8],  -y[7],  -y[6],  -y[5],  -y[4],  -y[3],  -y[2],  -y[1]],
    [x[12], x[11], x[10], x[9],  x[8],  x[7],  x[6],  x[5],  x[4],  x[3],  x[2],  -y[11], -y[10], -y[9],  -y[8],  -y[7],  -y[6],  -y[5],  -y[4],  -y[3],  -y[2]],
    [x[13], x[12], x[11], x[10], x[9],  x[8],  x[7],  x[6],  x[5],  x[4],  x[3],  -y[12], -y[11], -y[10], -y[9],  -y[8],  -y[7],  -y[6],  -y[5],  -y[4],  -y[3]],
    [x[14], x[13], x[12], x[11], x[10], x[9],  x[8],  x[7],  x[6],  x[5],  x[4],  -y[13], -y[12], -y[11], -y[10], -y[9],  -y[8],  -y[7],  -y[6],  -y[5],  -y[4]],
    [x[15], x[14], x[13], x[12], x[11], x[10], x[9],  x[8],  x[7],  x[6],  x[5],  -y[14], -y[13], -y[12], -y[11], -y[10], -y[9],  -y[8],  -y[7],  -y[6],  -y[5]],
    [x[16], x[15], x[14], x[13], x[12], x[11], x[10], x[9],  x[8],  x[7],  x[6],  -y[15], -y[14], -y[13], -y[12], -y[11], -y[10], -y[9],  -y[8],  -y[7],  -y[6]],
    [x[17], x[16], x[15], x[14], x[13], x[12], x[11], x[10], x[9],  x[8],  x[7],  -y[16], -y[15], -y[14], -y[13], -y[12], -y[11], -y[10], -y[9],  -y[8],  -y[7]],
    [x[18], x[17], x[16], x[15], x[14], x[13], x[12], x[11], x[10], x[9],  x[8],  -y[17], -y[16], -y[15], -y[14], -y[13], -y[12], -y[11], -y[10], -y[9],  -y[8]],
    [x[19], x[18], x[17], x[16], x[15], x[14], x[13], x[12], x[11], x[10], x[9],  -y[18], -y[17], -y[16], -y[15], -y[14], -y[13], -y[12], -y[11], -y[10], -y[9]],
    [x[20], x[19], x[18], x[17], x[16], x[15], x[14], x[13], x[12], x[11], x[10], -y[19], -y[18], -y[17], -y[16], -y[15], -y[14], -y[13], -y[12], -y[11], -y[10]],
    [x[21], x[20], x[19], x[18], x[17], x[16], x[15], x[14], x[13], x[12], x[11], -y[20], -y[19], -y[18], -y[17], -y[16], -y[15], -y[14], -y[13], -y[12], -y[11]],
    [x[22], x[21], x[20], x[19], x[18], x[17], x[16], x[15], x[14], x[13], x[12], -y[21], -y[20], -y[19], -y[18], -y[17], -y[16], -y[15], -y[14], -y[13], -y[12]],
    [x[23], x[22], x[21], x[20], x[19], x[18], x[17], x[16], x[15], x[14], x[13], -y[22], -y[21], -y[20], -y[19], -y[18], -y[17], -y[16], -y[15], -y[14], -y[13]],
    [x[24], x[23], x[22], x[21], x[20], x[19], x[18], x[17], x[16], x[15], x[14], -y[23], -y[22], -y[21], -y[20], -y[19], -y[18], -y[17], -y[16], -y[15], -y[14]],
    [x[25], x[24], x[23], x[22], x[21], x[20], x[19], x[18], x[17], x[16], x[15], -y[24], -y[23], -y[22], -y[21], -y[20], -y[19], -y[18], -y[17], -y[16], -y[15]],
    [x[26], x[25], x[24], x[23], x[22], x[21], x[20], x[19], x[18], x[17], x[16], -y[25], -y[24], -y[23], -y[22], -y[21], -y[20], -y[19], -y[18], -y[17], -y[16]],
    [x[27], x[26], x[25], x[24], x[23], x[22], x[21], x[20], x[19], x[18], x[17], -y[26], -y[25], -y[24], -y[23], -y[22], -y[21], -y[20], -y[19], -y[18], -y[17]],
    [x[28], x[27], x[26], x[25], x[24], x[23], x[22], x[21], x[20], x[19], x[18], -y[27], -y[26], -y[25], -y[24], -y[23], -y[22], -y[21], -y[20], -y[19], -y[18]],
    [x[29], x[28], x[27], x[26], x[25], x[24], x[23], x[22], x[21], x[20], x[19], -y[28], -y[27], -y[26], -y[25], -y[24], -y[23], -y[22], -y[21], -y[20], -y[19]],
    [x[30], x[29], x[28], x[27], x[26], x[25], x[24], x[23], x[22], x[21], x[20], -y[29], -y[28], -y[27], -y[26], -y[25], -y[24], -y[23], -y[22], -y[21], -y[20]]]

#Definición de matriz de resultados B
B = np.array([y[10:31]])
B = B.transpose()

# Solución de los coeficientes C  C=A^-1  *B;
# se puede escribir como A\B o inv(A)*B

C = np.dot(np.linalg.inv(A),B)

print('Los coeficientes quedan de la siguiente forma: \n \n b_0  = ' ,C[0][0])
print(' b_1  = ' ,C[1][0])
print(' b_2  = ' ,C[2][0])
print(' b_3  = ' ,C[3][0])
print(' b_4  = ' ,C[4][0])
print(' b_5  = ' ,C[5][0])
print(' b_6  = ' ,C[6][0])
print(' b_7  = ' ,C[7][0])
print(' b_8  = ' ,C[8][0])
print(' b_9  = ' ,C[9][0])
print(' b_10 = ' ,C[10][0])
print(' a_1  = ' ,C[11][0])
print(' a_2  = ' ,C[12][0])
print(' a_3  = ' ,C[13][0])
print(' a_4  = ' ,C[14][0])
print(' a_5  = ' ,C[15][0])
print(' a_6  = ' ,C[16][0])
print(' a_7  = ' ,C[17][0])
print(' a_8  = ' ,C[18][0])
print(' a_9  = ' ,C[19][0])
print(' a_10 = ' ,C[20][0])

print('Comparación de valores, \n ')


M = np.matmul(A, C)

m = np.array([y[10:31]])
m = m.transpose()

#Comparacion = np.array([M,m])
Comparacion = [["Función Encontrada","Función Original"],np.array([M,m])]


print(tabulate(Comparacion, headers=['Función encontrada','Función original']), "\n")
print("---------------------------------------------------------------------------------------------------------------")

#Encontrando H(z)
c = C.T

#Ordenando coeficientespara el denominador y el numerador

b = []
for i in range(11):
    b.append(c[0][i])
a = [1]
for j in range(11, 21):
    a.append(c[0][j])

#Transformada Z
z = symbols('z')

# Encontrar H(z)
Z = [1, z**(-1), z**(-2), z**(-3), z**(-4), z**(-5), z**(-6), z**(-7), z**(-8), z**(-9), z**(-10)]
h1 = sum([b[i]*Z[i] for i in range(11)])
h2 = sum([a[i]*Z[i] for i in range(11)])
print('La función H (sin fracciones parciales) para el sistema es:')
H = h1 / h2
print(H)

# Encontrar coeficientes de fracciones parciales
r, p, k = residue(b, a)

r = r.transpose()
p = p.transpose()

# Imprimir coeficientes de fracciones parciales
print('\nCoeficientes de fracciones parciales:')
print(f'r = {r}')
print(f'p = {p}')
print(f'k = {k}')

# Encontrar la función H con fracciones parciales
z = symbols('z')

h = sum(r / (1 - p * z**(-1))) + sum(k * [1, z**(-1)])
HH = []
for i in range(1, 32):
    z = i
    HH.append(sum(r / (1 - p * z**(-1))) + sum(k * [1, z**(-1)]))
print()
print(h)
print()
print(HH)
print()


print(a)

print(b)

# reescribiendo el vector a y b para el total de las posiciones
for i in range(11, 31):
    b[i] = b.append(0)
for j in range(10, 31):
    a[j] = a.append(0)

for i in range(11, 31):
    b[i] = 0
for j in range(10, 31):
    a[j] = 0
print(a)

print(b)


#Analisis Espectral
n = np.arange(0, 31)
f = 1/10  # frecuencia de 0.1 Hz
N1 = 30
N2 = 120

# Analisis espectral para X
# Transformadas
X1 = np.abs(np.fft.fft(x, N1))
X2 = np.abs(np.fft.fft(x, N2))

# Rango normalizado para transformadas
F1x = np.arange(0, N1) / N1
F2x = np.arange(0, N2) / N2

# Analisis Espectral Para Y
# Transformadas
Y1 = np.abs(np.fft.fft(y, N1))
Y2 = np.abs(np.fft.fft(y, N2))

# Rango normalizado  para transformadas
F1y = np.arange(0, N1)/N1
F2y = np.arange(0, N2)/N2


# Grafica de funciones
# Funciones X y Y

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))

axs[0, 0].stem(n, x)
axs[0, 0].set_title('X[n]')
axs[0, 1].stem(n, y)
axs[0, 1].set_title('Y[n]')

# Graficas del espectro
axs[1, 0].stem(F1x, X1, '.')
axs[1, 0].set_title('Espectro en frecuencia para X[n] con N=30')

axs[2, 0].stem(F2x, X2, '.')
axs[2, 0].set_title('Espectro en frecuencia para X[n] con N=120')

axs[1, 1].stem(F1y, Y1, '.')
axs[1, 1].set_title('Espectro en frecuencia para Y[n] con N=30')

axs[2, 1].stem(F2y, Y2, '.')
axs[2, 1].set_title('Espectro en frecuencia para Y[n] con N=120')

# Parte Real e imaginaria
axs[3, 0].stem(n, np.real(HH), '.')
axs[3, 0].set_title('Parte real H[z]')

axs[3, 1].stem(n, np.imag(HH), '.')
axs[3, 1].set_title('Parte imaginaria H[z]')

plt.tight_layout()


# Computing the frequency response of the filter
w, h = signal.freqz(b, a)


# Resoyesta en amplitud y frecuencia
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(w, 20 * np.log10(abs(h)))
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Frequency (rad/sample)')
angles = np.unwrap(np.angle(h))
ax2.plot(w, angles, 'g')
ax2.set_ylabel('Fase [radianes]', color='g')
ax2.set_xlabel('Frequency (rad/sample)')
fig.suptitle('Frequency Response of the Filter')
ax2.grid(True)


# Respuesta al impulso
plt.figure(4)
y = signal.lfilter(b, a, n)
plt.plot(n, y)
plt.title('Respuesta al impulso y al escalon unitario')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

# Plotting the pole-zero plot of the filter
fig, ax = plt.subplots()
zeros = np.roots(b)
poles = np.roots(a)
unit_circle = plt.Circle((0, 0), 1, fill=False, color='black')
ax.add_artist(unit_circle)
ax.scatter(zeros.real, zeros.imag, marker='o', facecolors='none', edgecolors='b', label='Zeros')
ax.scatter(poles.real, poles.imag, marker='x', color='r', label='Poles')
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.legend()
ax.set_title('Pole-Zero Plot of the Filter')

plt.show()