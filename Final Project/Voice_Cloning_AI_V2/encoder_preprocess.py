from encoder.preprocess import preprocess_librispeech, preprocess_voxceleb1, preprocess_voxceleb2 # Funciones para preprocesar los datos
from utils.argutils import print_args  # Funcion para imprimir los argumentos
from pathlib import Path # Es una biblioteca para trabajar con rutas de archivos y directorios
import argparse # es biblioteca es para analizar argumentos de la linea de comandos

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Preprocesa archivos de audio a partir de conjuntos de datos, los codifica como espectrogramas mel y "
                    "los escribe en el disco. Esto le permitira entrenar al codificador. "
                    "Los conjuntos de datos necesarios son al menos uno de VoxCeleb1, VoxCeleb2 y LibriSpeech. "
                    "Lo ideal sería tener los tres. Debes extraerlos tal cual  "
                    "despues de haberlos descargado y ponerlos en un mismo directorio, e.g.:\n"
                    "-[datasets_root]\n"
                    "  -LibriSpeech\n"
                    "    -train-other-500\n"
                    "  -VoxCeleb1\n"
                    "    -wav\n"
                    "    -vox1_meta.csv\n"
                    "  -VoxCeleb2\n"
                    "    -dev",
        formatter_class=MyFormatter
    )

    parser.add_argument("datasets_root", type=Path, help=\
        "Ruta al directorio que contiene los conjuntos de datos LibriSpeech/TTS y VoxCeleb.") # datasets_root: ruta al directorio que contiene los conjuntos de datos LibriSpeech/TTS y VoxCeleb.
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Ruta al directorio de salida que contendrá los espectrogramas fundidos. Si se omite, "
        "por defecto es <datasets_root>/SV2TTS/encoder/") # out_dir: ruta al directorio de salida que contendrá los espectrogramas fundidos. Si se omite, por defecto es <datasets_root>/SV2TTS/encoder/
    parser.add_argument("-d", "--datasets", type=str,
                        default="librispeech_other,voxceleb1,voxceleb2", help=\
        "Lista separada por comas con el nombre de los conjuntos de datos que desea preprocesar. Solo el tren "
        "de estos conjuntos de datos. Nombres posibles: librispeech_other, voxceleb1, voxceleb2.") # datasets: lista separada por comas con el nombre de los conjuntos de datos que desea preprocesar. Sólo el tren de estos conjuntos de datos. Nombres posibles: librispeech_other, voxceleb1, voxceleb2.
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Si omitir archivos de salida existentes con el mismo nombre. Util si este script se ha interrumpido.")
    parser.add_argument("--no_trim", action="store_true", help=\
        "Preprocesa el audio sin recortar los silencios (no recomendado).") # no_trim: preprocesa el audio sin recortar los silencios (no recomendado).
    args = parser.parse_args()

    # Verify webrtcvad is available
    if not args.no_trim:
        try:
            import webrtcvad # importamos webrtcvad que es para el filtrado de ruido
        except:
            raise ModuleNotFoundError("Paquete 'webrtcvad' no encontrado. Este paquete permite "
                "la eliminacion de ruido y se recomienda. Instalelo e intentelo de nuevo. Si la instalacion falla, "
                "utilice --no_trim para desactivar este mensaje de error.")
    del args.no_trim

    # Process the arguments
    args.datasets = args.datasets.split(",")
    if not hasattr(args, "out_dir"):    # si no tiene el atributo out_dir
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "encoder") # se le asigna la ruta datasets_root/SV2TTS/encoder
    assert args.datasets_root.exists() # si no existe la ruta datasets_root
    args.out_dir.mkdir(exist_ok=True, parents=True) # se crea el directorio out_dir

    # Preprocess the datasets
    print_args(args, parser) # imprime los argumentos
    preprocess_func = { # diccionario con los nombres de los conjuntos de datos y sus funciones de preprocesamiento
        "librispeech_other": preprocess_librispeech,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
    }
    args = vars(args) # convierte los argumentos en un diccionario
    for dataset in args.pop("datasets"): # para cada conjunto de datos
        print("Procesando %s" % dataset) # imprime el nombre del conjunto de datos
        preprocess_func[dataset](**args) # llama a la función de preprocesamiento correspondiente al conjunto de datos
