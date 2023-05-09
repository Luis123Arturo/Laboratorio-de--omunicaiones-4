clc
clear
pkg load signal;
pkg load optim;

[x, fs] = audioread('x2_U017.wav'); % se lee el archivo x2_U017.wav

% Almacenado en la actual carpet de trabajo.
% Calculo del tamaño del vector de audio

lx = length(x);

% se lee el archivo y2_U017.wav
% almacenando en la actual carpeta de trabajo.
% calculo del tamaño del vector de audio

[y, fs] = audioread('y2_U017.wav')

% Grafica de los vectores
n=0:lx-1;
figure(1);
subplot(2,1,1);
stem(n,x);
title('X[n]');
subplot(2,1,2);
stem(n,y);
title('Y[n]');

% funcion [b, a] = CalculoCoef2(x,y)
clc
lx=5000;
q=1:lx;
n=1:length(x);

% Determinacion de ssitema de ecuaciones simultaneas para dos variables con mayor numero de ecuaciones
for i=2001:lx-1
  c1(i)=x(i-2000);
  c2(i)=-y(i-750);
  b1(i)=y(i)-x(i);
endfor
C1=transpose(c1);
C2=transpose(c2);
B=transpose(b1);
D=[C1 C2];
lb = [-Inf; -Inf];
ub = [Inf; Inf];
R=lsqlin(D,B, [], [], [], [], lb, ub);
fprintf('El valor para cada uno de los coeficientes es:');
fprintf('\n A= ')
disp(R(1,1));
fprintf(' B= ')
disp(R(2,1));

% Comparacion evalualndo coeficientes ----
fprintf('Comparacion de valores para las posiciones entre [2001] y [20016]\n Primera columna: Funcion encontrada Segunda Fila: Funcion original Tercera Fila: Ruido s[n]');
for j=2001:2016
  m(j-2000)=x(j)+R(1,1)*x(j-2000)+R(2,1)*y(j-750);
  l(j-2000)=y(j);
  s(j-2000)=l(j-2000)-m(j-2000);
endfor
M=transpose(m);
L=transpose(l);
S=transpose(s);
Comparacion = [M L S]

b(1)=1;
b(2)=R(1,1);
a(1)=1;
a(2)=R(2,1);

fprintf('\n Coeficientes de fracciones parciales: \n');
[r, p, k]=residuez(b,a);
r=transpose(r);
p=transpose(p);

b=1:96000;
b=b*0;
a=b;
b(1)=1;
b(2000)=R(1,1);
a(1)=1;
a(750)=R(2,1);
for i=1:lx
  z=i;
  HH(i)=sum(r./(1-p*z^(-1))) + sum(k*[1 ; z^(-1)]);
endfor
HH;

% Analists espectral

f=1/10; % Frecuencia de 0.1Hz
N1=30; % numero de muestras
N2=120;

% Analisis Espectral para X
% Transformadas:
X1=abs(fft(x, N1));
X2=abs(fft(x,N2));

% Rango normalizado para transformadas:
F1x=[(0:N1-1)/N1];
F2x=[(0:N2-1)/N2];

% Analisis espectral para Y
% Transformadas
Y1=abs(fft(y,N1));
Y2=abs(fft(y,N2));

% Rango normalizado para transformadas
F1y=[(0:N1-1)/N1];
F2y=[(0:N2-1)/N2];

% Grafica de funciones

% Funciones X y Y
figure(2);
subplot(2,1,1);
stem(n,x);
title('X[n]');

subplot(2,1,2);
stem(n,y);
title('Y[n]');

% Graficas de los espectros
figure(3)
subplot(2,2,1);
stem(F1x, X1, '.');
title('Espectro para X[n] con N=30');

subplot(2,2,2);
stem(F2x,X2,'.');
title('Espectro para X[n] con N=120');

subplot(2,2,3);
stem(F1y, Y1, '.');
title('Espectro para Y[n] con N=30');

subplot(2,2,4);
stem(F2y, X2, '.');
title('Espectro para Y[n] con N=120');

% Parte real e imaginaria
figure(4)
subplot(2,1,1);
stem(q, real(HH));
title('Parte real H[z]');

subplot(2,1,2);
stem(q, imag(HH));
title('Parte imaginaria H[z]');

% Respuesta en amplitud y frecuencia
figure(5)
y=filter(b,a,q);
plot(q,y);
title('Respuesta al impulso y al escalon unitario');

% Polos y Ceros
figure(6)
zplane(b,a);
title('Diagrama de Polos y Ceros');
