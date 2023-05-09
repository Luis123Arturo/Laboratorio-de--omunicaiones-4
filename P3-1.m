opcion = 0; % Inicializar opcion fuera del rango válido
senal_grabada = []; % Inicializar variable para evitar errores en caso 2

while opcion ~= 3
    disp('Seleccione una opción:')
    disp('1. Grabar señal de voz')
    disp('2. Reproducir última señal de voz grabada')
    disp('3. Salir')
    opcion = input('Ingrese el número de la opción: ');

    switch opcion
        case 1
            disp('Grabando señal de voz...')
            duracion_grabacion = input('Ingrese la duración de la grabación en segundos: ');
            frecuencia_muestreo = 8000;
            num_bits = 16;
            num_canales = 1;

            grabacion = audiorecorder(frecuencia_muestreo, num_bits, num_canales);
            recordblocking(grabacion, duracion_grabacion);
            if isvalid(grabacion) && isempty(grabacion)
                senal_grabada = getaudiodata(grabacion);
                disp('Señal de voz grabada exitosamente')
            else
                disp('Error al grabar señal de voz')
            end
        case 2
            if ~isempty(senal_grabada)
                disp('Reproduciendo señal de voz...')
                sound(senal_grabada, frecuencia_muestreo)
            else
                disp('No hay señal de voz grabada')
            end
        case 3
            disp('Saliendo del programa')
        otherwise
            disp('Opción inválida')
    end
end


