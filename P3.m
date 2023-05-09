% Cargar el archivo de audio [input_signal, fs] = audioread('audio.wav');

% Diseñar el filtro RFI
fc = 1000; % frecuencia de corte bw = 500; % ancho de banda

% Calcule la frecuencia normalizada Wn = [fc-bw/2, fc+bw/2]/(fs/2);

% Diseñar el filtro Butterworth de segundo orden [b, a] = butter(2, Wn);

% Diseñar el filtro Notch de segundo orden para eliminar interferencias fn = 1200; % frecuencia de interferencia
Wn_notch = fn/(fs/2);
[b_notch, a_notch] = iirnotch(Wn_notch, 0.1);

% Combinar los dos filtros en serie b_total = conv(b, b_notch);
a_total = conv(a, a_notch);

% Aplicar el filtro RFI a la señal de audio filtered_signal_RFI = filter(b_total, a_total, input_signal);

% Diseñar el filtro RII
fc = 1000; % frecuencia de corte
gain = 20; % ganancia en la banda de paso

% Calcule la frecuencia normalizada Wn = fc/(fs/2);

% Diseñar el filtro Chebyshev de tercer orden con un polo real y dos polos complejos conjugados [b, a] = cheby1(3, gain, Wn, 'high');

% Aplicar el filtro RII a la señal de audio filtered_signal_RII = filter(b, a, filtered_signal_RFI);

% Graficar la señal de audio original t = 0:1/fs:(length(input_signal)-1)/fs; figure();
plot(t, input_signal); xlabel('Tiempo (s)'); ylabel('Amplitud'); title('Señal original');

% Graficar la señal de audio filtrada con el filtro RFI t = 0:1/fs:(length(filtered_signal_RFI)-1)/fs;
figure();
plot(t, filtered_signal_RFI); xlabel('Tiempo (s)'); ylabel('Amplitud');
title('Señal filtrada con filtro RFI');

% Graficar la señal de audio filtrada con el filtro RII t = 0:1/fs:(length(filtered_signal_RII)-1)/fs;
figure();
plot(t, filtered_signal_RII); xlabel('Tiempo (s)'); ylabel('Amplitud');
title('Señal filtrada con filtro RII');

