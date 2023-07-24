import argparse # esta biblioteca es para poder pasar argumentos al programa
import os # esta biblioteca es para poder acceder a las variables de entorno del sistema operativo
from pathlib import Path # esta biblioteca es para poder acceder a las rutas de los archivos

import librosa # esta biblioteca es para poder cargar los archivos de audio
import numpy as np # esta biblioteca es para poder trabajar con matrices
import soundfile as sf # esta biblioteca es para poder guardar los archivos de audio
import torch # esta biblioteca es para poder trabajar con tensores

from encoder import inference as encoder # esta biblioteca es para poder cargar el modelo de encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size # esta biblioteca es para poder cargar el modelo de encoder
from synthesizer.inference import Synthesizer # esta biblioteca es para poder cargar el modelo de synthesizer
from utils.argutils import print_args # esta biblioteca es para poder imprimir los argumentos
from utils.default_models import ensure_default_models # esta biblioteca es para poder cargar los modelos por defecto
from vocoder import inference as vocoder # esta biblioteca es para poder cargar el modelo de vocodery

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder") # si ejecutamos el programa con el argumento -e, --enc_model_fpath, el valor por defecto es saved_models/default/encoder.pt
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer") # si ejecutamos el programa con el argumento -s, --syn_model_fpath, el valor por defecto es saved_models/default/synthesizer.pt
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder") # si ejecutamos el programa con el argumento -v, --voc_model_fpath, el valor por defecto es saved_models/default/vocoder.pt
    parser.add_argument("--cpu", action="store_true", help=\
        "Si es True, el procesamiento se realiza en la CPU, incluso cuando hay una GPU disponible.") # si ejecutamos el programa con el argumento --cpu, el valor por defecto es False
    parser.add_argument("--no_sound", action="store_true", help=\
        "Si es True, el audio no se reproducirá.") # si ejecutamos el programa con el argumento --no_sound, el valor por defecto es False
    parser.add_argument("--seed", type=int, default=None, help=\
        "Valor opcional de la semilla de numeros aleatorios para que la caja de herramientas sea determinista.") # si ejecutamos el programa con el argumento --seed, el valor por defecto es None
    args = parser.parse_args() # parseamos los argumentos
    arg_dict = vars(args) # convertimos los argumentos en un diccionario
    print_args(args, parser) # imprimimos los argumentos

    # Oculta las GPU de Pytorch para forzar el procesamiento de la CPU
    if arg_dict.pop("cpu"): # si el argumento cpu es True
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # ocultamos las GPU de Pytorch para forzar el procesamiento de la CPU

    print("Ejecutar una prueba de su configuracion...\n") # imprimimos un mensaje

    if torch.cuda.is_available(): # si hay una GPU disponible
        device_id = torch.cuda.current_device() # obtenemos el id de la GPU
        gpu_properties = torch.cuda.get_device_properties(device_id) # obtenemos las propiedades de la GPU
        ## Print some environment information (for debugging purposes)
        print("Se han encontrado %d GPUs disponibles. Usando GPU %d (%s) de capacidad de calculo %d.%d con "
            "%.1fGb de memoria total.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9)) # imprimimos el numero de GPUs disponibles, el id de la GPU, el nombre de la GPU, la capacidad de computo de la GPU, la memoria total de la GPU
    else:
        print("Uso de la CPU para la inferencia.\n") # este mensaje se muestra si no hay una GPU disponible

    ## Load the models one by one.
    print("Preparar el codificador, el sintetizador y el vocoder...") # esto se imprime si hay una GPU disponible
    ensure_default_models(Path("saved_models")) # cargamos los modelos por defecto
    encoder.load_model(args.enc_model_fpath) # cargamos el modelo de encoder
    synthesizer = Synthesizer(args.syn_model_fpath) # cargamos el modelo de synthesizer
    vocoder.load_model(args.voc_model_fpath) # cargamos el modelo de vocoder

    ## Run a test
    print("Prueba tu configuracion con pequeñas entradas.")
    """
    Reenvía una onda de audio de ceros que dure 1 segundo. Observe cómo podemos obtener la
    del codificador, que puede ser diferente.
    Si no estás familiarizado con el audio digital, debes saber que se codifica como una matriz de floats
    (o a veces enteros, pero sobre todo flotantes en estos proyectos) que van de -1 a 1.
    La frecuencia de muestreo es el número de valores (muestras) grabados por segundo, se establece en
    16000 para el codificador. La creación de una matriz de longitud <sampling_rate> siempre corresponderá
    a un audio de 1 segundo.
    """
    print("\tPrueba del encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate)) # reenvía una onda de audio de ceros que dure 1 segundo

    """
    EMBEDED TEXT TO FILE PROCESS
    Crea una incrustación ficticia. Normalmente se utilizaría la incrustación que encoder.embed_utterance
    devuelve, pero aquí vamos a hacer uno nosotros mismos sólo por el bien de mostrar que es
    posible.
    """
    embed = np.random.rand(speaker_embedding_size)  # crea una incrustación ficticia de 256 dimensiones (el tamaño de la incrustación del hablante)

    embed /= np.linalg.norm(embed) # normaliza la incrustación
    # El sintetizador puede gestionar varias entradas por lotes. Aquí sólo se utiliza una entrada.
    embeds = [embed, np.zeros(speaker_embedding_size)]  # crea una lista de incrustaciones ficticias
    texts = ["test 1", "test 2"] # crea una lista de textos ficticios
    print("\tPrueba del synthesizer... (cargando el modelo de sintesis puede tardar un tiempo)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds) # sintetizamos los espectrogramas

    """
    El vocoder sintetiza una forma de onda cada vez, pero es más eficaz para las largas. 
    Podemos concatenar los espectrogramas mel en uno solo.
    """
    mel = np.concatenate(mels, axis=1)
    # El vocoder puede tomar una función callback para mostrar la generación, por ahora esta oculto
    no_action = lambda *args: None # no hace nada
    print("\tPrueba del vocoder...")
    """
    Para que esta prueba sea corta, pasaremos una longitud objetivo corta. La longitud objetivo es la longitud de 
    los segmentos wav que se procesan en paralelo. Por ejemplo, para audio muestreado a 16000 hercios, una longitud 
    objetivo de 8000 significa que el audio objetivo se cortará en trozos de 0,5 segundos que se generarán todos juntos. 
    Los parámetros aquí son absurdamente cortos, y eso tiene un efecto perjudicial en la calidad del audio. En general, 
    se recomiendan los parámetros por defecto.
    """
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action) # sintetizamos la forma de onda

    print("Todas las pruebas superadas. Ahora puedes sintetizar el habla.\n\n")

    ## Interactive speech generation
    print("Este es un ejemplo de interfaz sin GUI.\n")

    print("Bucle de generación interactivo")
    num_generated = 0 # inicializamos el numero de generaciones en 0

    while True:
        try:
            # Obtener la ruta del archivo de audio de referencia
            message = "Voz de referencia: introduzca la ruta del archivo de audio de la voz que desea clonar. (mp3, " \
                      "wav, m4a, flac, ...):\n"
            in_fpath = Path(input(message).replace("\"", "").replace("\'",
                                                                     ""))  # obtenemos la ruta del archivo de audio de referencia

            ## COMPUTING EMBENDING
            ## Procesamos el audio ingresad, se convierte a matriz numpy y datos de audio; tambien se tiene una discriminación de audio
            ## normalizando el sonido : es decir se ajustara el volumen de la señal de audio para que alcance una nivel de potencia de -30 dBFS (dentro de params)
            ## cortando los silencios largos : es decir borraremos los silencios mayores a 6 periodos de 30 ms ( q es la ventana addmitida params)

            # primero cargamos el wav utilizando la función que proporciona el codificador de altavoces

            # Los dos metodos siguientes son equivalentes:
            # - Cargar directamente desde la ruta del archivo:
            preprocessed_wav = encoder.preprocess_wav(
                in_fpath)  # cargamos el wav utilizando la función que proporciona el codificador de altavoces
            # - Si el wav ya está cargado:
            original_wav, sampling_rate = librosa.load(str(in_fpath))  # cargamos el wav utilizando librosa
            preprocessed_wav = encoder.preprocess_wav(original_wav,
                                                      sampling_rate)  # cargamos el wav utilizando la función que proporciona el codificador de altavoces
            print("Archivo cargado correctamente")  # mensaje de éxito

            """ 
            A continuación, derivamos la incrustación. Hay muchas funciones y parámetros con los que 
            interactúa el codificador de altavoces.
            """
            # Dividimos en fragmentos parciales y calculamos la incrustación promedio del audio preprocesado
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Creada la incrustacion")

            ## IMPUT TEXT
            text = input("Escriba una frase (+-20 palabras) para sintetizar:\n")

            if args.seed is not None:
                torch.manual_seed(args.seed)  # establecemos la semilla de torch
                synthesizer = Synthesizer(args.syn_model_fpath)  # recargamos el modelo de synthesizer

            # El sintetizador funciona por lotes, así que tienes que poner tus datos en una lista o matriz numpy
            texts = [text]  # creamos una lista de textos
            embeds = [embed]  # creamos una lista de incrustaciones

            # passing return_alignments=True

            ##START SYNTHETIZER STEP
            ##Tacotron
            specs = synthesizer.synthesize_spectrograms(texts, embeds)  # sintetizamos los espectrogramas
            spec = specs[0]  # obtenemos el espectrograma
            print("Creado el espectrograma mel")  # imprimimos un mensaje

            ## Generating the waveform
            print("Sintetizar la forma de onda:")

            if args.seed is not None:  # si se especifica la semilla, restablece la semilla del torch y recarga el vocoder
                torch.manual_seed(args.seed)  # establecemos la semilla de torch
                vocoder.load_model(args.voc_model_fpath)  # recargamos el modelo de vocoder

            # cuanto más largo sea el espectrograma, más eficiente en tiempo será el vocoder.
            generated_wav = vocoder.infer_waveform(spec)  # sintetizamos la forma de onda

            ## Post-generation
            # Hay un error en la biblioteca de vocoders que hace que el audio sea un poco más corto de lo que debería ser.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")  # corregimos el error

            # Recorte el exceso de silencios para compensar los huecos en los espectrogramas
            generated_wav = encoder.preprocess_wav(generated_wav)  # corregimos el error

            # Play the audio (non-blocking)
            if not args.no_sound:  # si no se especifica el argumento no_sound
                import sounddevice as sd  # importamos la libreria sounddevice que es para reproducir audio

                try:
                    sd.stop()  # detenemos la reproducción de audio
                    sd.play(generated_wav, synthesizer.sample_rate)  # reproducimos el audio
                except sd.PortAudioError as e:  # si hay un error
                    print("\nCaught exception: %s" % repr(e))
                    print(
                        "Continura sin reproduccion de audio. Suprima este mensaje con la tecla \"--no_sound\" flag.\n")
                except:
                    raise

            # Guardar audio como wav
            filename = "demo_output_%02d.wav" % num_generated  # creamos el nombre del archivo
            print(generated_wav.dtype)  # imprimimos el tipo de dato
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)  # guardamos el audio
            num_generated += 1  # aumentamos el numero de generaciones
            print("\nSalida guardada como %s\n\n" % filename)  # imprimimos un mensaje


        except Exception as e:  # si hay un error
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
